"""Bounded live P/D concurrency and deterministic restore-failure exercises."""

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests


def plugin_environment(output):
    root = output / "lifecycle-plugin"
    root.mkdir()
    source = Path(__file__).with_name("l3_lifecycle_hooks.py")
    (root / source.name).write_bytes(source.read_bytes())
    metadata = root / "dcp_hicache_lifecycle-0.0.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text("Name: dcp-hicache-lifecycle\nVersion: 0.0\n")
    (metadata / "entry_points.txt").write_text(
        "[sglang.srt.plugins]\nhicache_lifecycle = l3_lifecycle_hooks:register\n"
    )
    events = output / "lifecycle-events"
    events.mkdir()
    return dict(
        PYTHONPATH=os.pathsep.join([str(root), os.environ.get("PYTHONPATH", "")]),
        SGLANG_PLUGINS="hicache_lifecycle",
        DCP_L3_TEST_EVENTS_DIR=str(events),
    )


class LiveRestoreLifecycle:
    def __init__(self, test):
        self.test = test
        self.events = test.output / "lifecycle-events"

    def run(self):
        self._concurrent()
        self._interrupted("abort")
        self._retry("abort-retry", self.test.follow_refs[1], 512)
        self._interrupted("missing-state")
        # Remove the P checkpoint too so the retry has the cold control's
        # producer history. The failed request already removed D's checkpoint.
        self.test._remove_state_shard()
        self._retry("io-failure-retry", self.test.cold_follow_refs[1], 0)

    def _concurrent(self):
        test = self.test
        test._flush()
        before = self._completions()
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [
                pool.submit(test._generate, f"concurrent-follow-{i}", ids, 14)
                for i, ids in enumerate(test.follow_ids)
            ]
            for i, future in enumerate(futures):
                result = future.result(timeout=240)
                test._parity(f"concurrent-follow-{i}", test.follow_refs[i], result)
                details = result["meta_info"]["cached_tokens_details"]
                test.assertEqual(details["host"], 0)
                test.assertGreater(details["storage"], 0)
        test.report["concurrent_requests"] = 3
        test._save()
        self._assert_completions(before, [256, 512, 1024])

    def _completions(self):
        matches = re.findall(
            r"DCP L3 prefetch: tp_rank=(\d+) tokens=(\d+)",
            (self.test.output / "fresh-decode.log").read_text(),
        )
        return {
            rank: [int(n) for r, n in matches if int(r) == rank and int(n)]
            for rank in range(4)
        }

    def _assert_completions(self, before, expected):
        after = self._completions()
        for rank in range(4):
            self.test.assertEqual(after[rank][: len(before[rank])], before[rank])
            self.test.assertEqual(sorted(after[rank][len(before[rank]) :]), expected)

    def _wait(self, predicate, description, seconds=30):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            value = predicate()
            if value:
                return value
            time.sleep(0.02)
        self.test.fail(f"Timed out waiting for {description}")

    def _read(self, kind, rank):
        path = self.events / f"{self.name}-{kind}-{rank}.json"
        return json.loads(path.read_text()) if path.exists() else None

    def _arm(self, name):
        self.name = name
        self.rid = f"dcp-l3-lifecycle-{name}"
        path = self.events / "arm.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(
                dict(
                    id=name, rid=self.rid, pause_rank=3, tokens=self.test.follow_ids[1]
                )
            )
        )
        temporary.replace(path)

    def _failed_request(self):
        payload = dict(
            rid=self.rid,
            input_ids=self.test.follow_ids[1],
            sampling_params=dict(temperature=0, max_new_tokens=14, ignore_eos=True),
            return_logprob=True,
        )
        started = time.monotonic()
        response = requests.post(
            self.test.router_url + "/generate", json=payload, timeout=120
        )
        try:
            body = response.json()
        except ValueError:
            body = response.text
        record = dict(
            request=payload,
            status_code=response.status_code,
            response=body,
            elapsed_seconds=time.monotonic() - started,
        )
        (self.test.output / f"{self.name}.json").write_text(
            json.dumps(record, indent=2)
        )
        return record

    def _assert_failed(self, record):
        if self.name == "missing-state":
            outcome = self._read("read-result", 3)["mamba"]
            self.test.assertTrue(outcome)
            self.test.assertFalse(all(outcome))
        self.test.assertIn(record["status_code"], (200, 500, 503))
        if record["status_code"] == 200:
            reason = record["response"]["meta_info"]["finish_reason"]
            self.test.assertEqual(reason["type"], "abort")
        else:
            message = record["response"]["message"].lower()
            self.test.assertIn(self.rid, message)
            if self.name == "missing-state":
                self.test.assertIn("decode transfer failed", message)
            else:
                self.test.assertIn("abort", message)

    def _trigger(self):
        if self.name == "abort":
            for url in (self.test.p_url, self.test.d_url):
                response = requests.post(
                    url + "/abort_request",
                    json=dict(rid=self.rid, abort_all=False),
                    timeout=10,
                )
                response.raise_for_status()
            held = self._wait(lambda: self._read("aborted", 3), "abort while IO held")
            before = self._read("before", 3)
            self.test.assertLess(held["host_state"], before["host_state"])
        else:
            victims = [
                item
                for item in self.test._inventory()
                if item["path"].startswith("decode/")
                and self.test._is_state(item["path"], rank=3)
            ]
            self.test.assertTrue(victims)
            (self.test.output / "post-promise-removed-state.json").write_text(
                json.dumps(victims, indent=2)
            )
            for item in victims:
                (self.test.storage / item["path"]).unlink()

    def _drained(self):
        return all(
            self._read("before", rank) is not None
            and self._read("after", rank) == self._read("before", rank)
            for rank in range(4)
        )

    def _interrupted(self, name):
        self.test._flush()
        before = self._completions()
        self._arm(name)
        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(self._failed_request)
                try:
                    entered = self._wait(
                        lambda: self._read("entered", 3), "actual state-file read"
                    )
                    self.test.assertTrue(entered["keys"])
                    for rank in range(4):
                        promise = self._wait(
                            lambda r=rank: self._read("promise", r),
                            f"rank {rank} published transfer prefix",
                        )
                        self.test.assertEqual(promise["prefix"], 512)
                    self._trigger()
                finally:
                    (self.events / f"{name}.release").touch()
                self._assert_failed(future.result(timeout=120))
            # No flush/reset here: cleanup itself must return all allocations.
            self._wait(self._drained, "all-rank allocation return", seconds=45)
            # IO may finish copying bytes after cancellation. Reuse is guarded
            # by actual radix insertions and allocation snapshots, not IO ACKs.
            after = self._completions()
            for rank in range(4):
                snapshot = self._read("after", rank)
                self.test.assertEqual(snapshot["tree_device_tokens"], 0)
                self.test.assertEqual(snapshot["tree_auxiliary_slots"], 0)
                publications = [
                    json.loads(line)
                    for line in (self.events / f"publications-{rank}.jsonl")
                    .read_text()
                    .splitlines()
                ]
                self.test.assertTrue(any(not item["dropped"] for item in publications))
                self.test.assertFalse(
                    any(
                        item["interrupted_request"] == name
                        and not item["dropped"]
                        and item["key_tokens"] > 0
                        for item in publications
                    )
                )
            self.test.report["requests"] += 1
            self.test.report.setdefault("lifecycle", {})[name] = dict(
                promised_prefix=512,
                paused_rank=3,
                allocations_returned=True,
                published_tokens=0,
                io_completions={
                    rank: after[rank][len(before[rank]) :] for rank in range(4)
                },
            )
            self.test._save()
        finally:
            (self.events / "arm.json").unlink(missing_ok=True)

    def _retry(self, name, expected, cached):
        self.test._flush()
        before = self._completions()
        actual = self.test._generate(name + "-follow", self.test.follow_ids[1], 14)
        self.test._parity(name, expected, actual)
        self.test.assertEqual(actual["meta_info"]["cached_tokens"], cached)
        self._assert_completions(before, [cached] if cached else [])
