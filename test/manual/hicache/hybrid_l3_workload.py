"""Pinned, meaningful hybrid-cache continuations for manual correctness sweeps."""

import json
import math
import time

import requests
from transformers import AutoTokenizer

MODEL = "moonshotai/Kimi-Linear-48B-A3B-Instruct"
REVISION = "e1df551a447157d4658b573f9a695d57658590e9"


def known_code_cases(*, model, sizes):
    tokenizer = AutoTokenizer.from_pretrained(
        model, revision=REVISION, trust_remote_code=True
    )
    template = tokenizer.apply_chat_template(
        [
            dict(
                role="user",
                content="Remember the secret access code: 739391.\nINSERT_FILLER_HERE\nReply with the secret access code four times separated by single spaces, and nothing else.",
            )
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    before, after = template.split("INSERT_FILLER_HERE")
    suffix = tokenizer.encode(after, add_special_tokens=False)
    filler = tokenizer.encode(
        "The library keeps records of books. ", add_special_tokens=False
    )
    cases = []
    for index, size in enumerate(sizes):
        prefix = tokenizer.encode(
            before.replace("Remember", f"Record {index}. Remember"),
            add_special_tokens=False,
        )
        count = size - len(prefix) - len(suffix)
        assert count > 0
        cases.append(prefix + (filler * (count // len(filler) + 1))[:count] + suffix)
    assert len({tuple(ids[:64]) for ids in cases}) == len(cases)
    return cases


def generate(*, url, ids, count, output, name, rid=None):
    payload = dict(
        input_ids=ids,
        sampling_params=dict(temperature=0, max_new_tokens=count, ignore_eos=True),
        return_logprob=True,
    )
    if rid is not None:
        payload["rid"] = rid
    started = time.monotonic()
    response = requests.post(url + "/generate", json=payload, timeout=240)
    record = dict(
        request=payload,
        status_code=response.status_code,
        response=response.json(),
        elapsed_seconds=time.monotonic() - started,
    )
    (output / f"{name}.json").write_text(json.dumps(record, indent=2))
    response.raise_for_status()
    result = record["response"]
    assert len(result["output_ids"]) == count
    assert "739391" in result["text"], result["text"]
    scores = result["meta_info"]["output_token_logprobs"]
    assert [x[1] for x in scores] == result["output_ids"]
    assert all(math.isfinite(x[0]) for x in scores)
    return result


def parity(*, reference, actual, atol=0.20):
    assert actual["output_ids"] == reference["output_ids"]
    delta = max(
        abs(a[0] - b[0])
        for a, b in zip(
            actual["meta_info"]["output_token_logprobs"],
            reference["meta_info"]["output_token_logprobs"],
            strict=True,
        )
    )
    assert delta <= atol, delta
    return delta
