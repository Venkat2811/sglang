#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import requests

from sglang.benchmark.utils import get_processor, get_tokenizer
from sglang.test.bench_one_batch_server_internal import BenchArgs, run_one_case

DEFAULT_TIMEOUT = 600


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _summarize_metric(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "min": float(arr.min()),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "p50": _percentile(values, 50),
        "p95": _percentile(values, 95),
        "p99": _percentile(values, 99),
        "max": float(arr.max()),
    }


def _format_num(value: float, scale: float = 1.0, unit: str = "") -> str:
    if math.isnan(value):
        return "n/a"
    return f"{value * scale:10.2f}{unit}"


def _render_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(row: list[str]) -> str:
        return "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))

    sep = "  ".join("-" * width for width in widths)
    lines = [fmt_row(headers), sep]
    lines.extend(fmt_row(row) for row in rows)
    return "\n".join(lines)


def _load_tokenizer(base_url: str, dataset_name: str):
    response = requests.get(base_url + "/server_info", timeout=DEFAULT_TIMEOUT)
    response.raise_for_status()
    server_info = response.json()
    if "tokenizer_path" in server_info:
        tokenizer_path = server_info["tokenizer_path"]
    elif "prefill" in server_info:
        tokenizer_path = server_info["prefill"][0]["tokenizer_path"]
    else:
        raise RuntimeError(f"Unable to determine tokenizer path from {server_info.keys()}")

    if dataset_name == "mmmu":
        tokenizer = get_processor(tokenizer_path)
    else:
        tokenizer = get_tokenizer(tokenizer_path)
    return tokenizer


def _result_to_dict(result: Any, iteration: int, seed: int) -> dict[str, Any]:
    return {
        "iteration": iteration,
        "seed": seed,
        "run_name": result.run_name,
        "batch_size": result.batch_size,
        "input_len": result.input_len,
        "output_len": result.output_len,
        "latency": result.latency,
        "input_throughput": result.input_throughput,
        "output_throughput": result.output_throughput,
        "overall_throughput": result.overall_throughput,
        "last_ttft": result.last_ttft,
        "last_gen_throughput": result.last_gen_throughput,
        "acc_length": result.acc_length,
        "cache_hit_rate": result.cache_hit_rate,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--input-len", type=int, required=True)
    parser.add_argument("--output-len", type=int, required=True)
    parser.add_argument("--repeats", type=int, required=True)
    parser.add_argument("--warmup-repeats", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--dataset-name", default="random")
    parser.add_argument("--dataset-path", default="")
    parser.add_argument("--client-stream-interval", type=int, default=1)
    parser.add_argument("--seed-base", type=int, default=1000)
    parser.add_argument("--parallel-batch", action="store_true")
    parser.add_argument("--result-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--summary-txt", type=Path, required=True)
    args = parser.parse_args()

    args.result_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_txt.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = _load_tokenizer(args.base_url, args.dataset_name)

    for warmup_idx in range(args.warmup_repeats):
        BenchArgs.seed = args.seed_base - args.warmup_repeats + warmup_idx
        run_one_case(
            url=args.base_url,
            batch_size=args.batch_size,
            input_len=args.input_len,
            output_len=args.output_len,
            temperature=args.temperature,
            return_logprob=False,
            stream_interval=args.client_stream_interval,
            input_len_step_percentage=0.0,
            run_name="",
            result_filename="",
            tokenizer=tokenizer,
            dataset_name=args.dataset_name,
            dataset_path=args.dataset_path,
            parallel_batch=args.parallel_batch,
            cache_hit_rate=0.0,
            backend="sglang",
            model_name=None,
        )

    results: list[dict[str, Any]] = []
    with args.result_jsonl.open("w", encoding="utf-8") as fout:
        for iteration in range(args.repeats):
            seed = args.seed_base + iteration
            BenchArgs.seed = seed
            result = run_one_case(
                url=args.base_url,
                batch_size=args.batch_size,
                input_len=args.input_len,
                output_len=args.output_len,
                temperature=args.temperature,
                return_logprob=False,
                stream_interval=args.client_stream_interval,
                input_len_step_percentage=0.0,
                run_name=f"{args.run_name}-iter-{iteration:04d}",
                result_filename="",
                tokenizer=tokenizer,
                dataset_name=args.dataset_name,
                dataset_path=args.dataset_path,
                parallel_batch=args.parallel_batch,
                cache_hit_rate=0.0,
                backend="sglang",
                model_name=None,
            )
            row = _result_to_dict(result, iteration, seed)
            fout.write(json.dumps(row, sort_keys=True) + "\n")
            results.append(row)
            if (iteration + 1) % 8 == 0 or iteration + 1 == args.repeats:
                print(
                    f"[repeat-bench] completed {iteration + 1}/{args.repeats} "
                    f"batches for {args.run_name}"
                )

    latency_values = [row["latency"] for row in results]
    ttft_values = [row["last_ttft"] for row in results]
    input_tp_values = [row["input_throughput"] for row in results]
    output_tp_values = [row["output_throughput"] for row in results]
    overall_tp_values = [row["overall_throughput"] for row in results]

    total_wall_s = float(sum(latency_values))
    summary = {
        "run_name": args.run_name,
        "base_url": args.base_url,
        "repeats": args.repeats,
        "warmup_repeats": args.warmup_repeats,
        "batch_size": args.batch_size,
        "input_len": args.input_len,
        "output_len": args.output_len,
        "completed_requests": args.repeats * args.batch_size,
        "completed_responses": args.repeats * args.batch_size,
        "prompt_tokens_total": args.repeats * args.batch_size * args.input_len,
        "completion_tokens_total": args.repeats * args.batch_size * args.output_len,
        "total_wall_s": total_wall_s,
        "latency_s": _summarize_metric(latency_values),
        "ttft_s": _summarize_metric(ttft_values),
        "input_throughput_tok_s": _summarize_metric(input_tp_values),
        "output_throughput_tok_s": _summarize_metric(output_tp_values),
        "overall_throughput_tok_s": _summarize_metric(overall_tp_values),
    }

    args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    table = _render_table(
        ["Metric", "Min", "P50", "P95", "P99", "Mean", "Max"],
        [
            [
                "Latency (ms)",
                _format_num(summary["latency_s"]["min"], 1000),
                _format_num(summary["latency_s"]["p50"], 1000),
                _format_num(summary["latency_s"]["p95"], 1000),
                _format_num(summary["latency_s"]["p99"], 1000),
                _format_num(summary["latency_s"]["mean"], 1000),
                _format_num(summary["latency_s"]["max"], 1000),
            ],
            [
                "TTFT (ms)",
                _format_num(summary["ttft_s"]["min"], 1000),
                _format_num(summary["ttft_s"]["p50"], 1000),
                _format_num(summary["ttft_s"]["p95"], 1000),
                _format_num(summary["ttft_s"]["p99"], 1000),
                _format_num(summary["ttft_s"]["mean"], 1000),
                _format_num(summary["ttft_s"]["max"], 1000),
            ],
            [
                "Input TP (tok/s)",
                _format_num(summary["input_throughput_tok_s"]["min"]),
                _format_num(summary["input_throughput_tok_s"]["p50"]),
                _format_num(summary["input_throughput_tok_s"]["p95"]),
                _format_num(summary["input_throughput_tok_s"]["p99"]),
                _format_num(summary["input_throughput_tok_s"]["mean"]),
                _format_num(summary["input_throughput_tok_s"]["max"]),
            ],
            [
                "Output TP (tok/s)",
                _format_num(summary["output_throughput_tok_s"]["min"]),
                _format_num(summary["output_throughput_tok_s"]["p50"]),
                _format_num(summary["output_throughput_tok_s"]["p95"]),
                _format_num(summary["output_throughput_tok_s"]["p99"]),
                _format_num(summary["output_throughput_tok_s"]["mean"]),
                _format_num(summary["output_throughput_tok_s"]["max"]),
            ],
        ],
    )
    summary_text = "\n".join(
        [
            f"Run name:            {args.run_name}",
            f"Completed batches:   {args.repeats}",
            f"Completed requests:  {summary['completed_requests']}",
            f"Completed responses: {summary['completed_responses']}",
            f"Prompt tokens total: {summary['prompt_tokens_total']}",
            f"Output tokens total: {summary['completion_tokens_total']}",
            f"Total wall time (s): {summary['total_wall_s']:.4f}",
            "",
            table,
            "",
        ]
    )
    args.summary_txt.write_text(summary_text, encoding="utf-8")
    print(summary_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
