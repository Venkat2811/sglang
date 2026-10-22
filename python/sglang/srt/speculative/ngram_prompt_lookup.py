"""Prompt-lookup drafts for NGRAM: copy the continuation of the longest earlier match of the request's own suffix.

The SGLang NGRAM corpus only learns from generated tokens; agentic outputs mostly copy from the prompt (file contents,
tool output, earlier tool calls), so this index covers the request's whole prompt plus its output so far.
Per request: suffix hashes for the n-gram sizes in MATCH_SIZES. The prompt is hashed once with numpy (sorted arrays,
latest end per hash); generated tokens go to a small dict. A lookup takes the longest size whose suffix occurred
earlier (latest occurrence wins) and returns the tokens that followed it.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

MATCH_SIZES = (32, 16, 8, 6, 4, 3, 2, 1)
_MASK64 = (1 << 64) - 1
_BASE = 0x9E3779B97F4A7C15
_POW = [pow(_BASE, i, 1 << 64) for i in range(max(MATCH_SIZES) + 1)]


def _bulk_hashes(tokens: np.ndarray, n: int) -> np.ndarray:
    """Polynomial hash of every length-n window, as uint64 with wraparound; element i ends at i + n."""
    t = tokens.astype(np.uint64) + np.uint64(1)
    m = len(t) - n + 1
    h = np.zeros(m, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for j in range(n):
            h = h * np.uint64(_BASE) + t[j : j + m]
    return h


class _RequestIndex:
    __slots__ = ("seq", "n_out", "sorted_h", "sorted_end", "recent", "indexed", "h_origin", "h_rel")

    def __init__(self, prompt: Sequence[int]):
        self.seq: List[int] = list(prompt)
        self.n_out = 0
        arr = np.asarray(self.seq, dtype=np.int64)
        self.sorted_h: Dict[int, np.ndarray] = {}
        self.sorted_end: Dict[int, np.ndarray] = {}
        for n in MATCH_SIZES:
            if len(arr) < n:
                continue
            h = _bulk_hashes(arr, n)
            ends = np.arange(n, n + len(h), dtype=np.int64)
            # Stable sort keeps ends ascending inside one hash, so the last of a run is the latest occurrence.
            order = np.argsort(h, kind="stable")
            self.sorted_h[n] = h[order]
            self.sorted_end[n] = ends[order]
        self.recent: Dict[int, Dict[int, int]] = {n: {} for n in MATCH_SIZES}
        # Ends <= len(prompt) are in the sorted arrays; later ends go to `recent` as the output grows.
        self.indexed = len(self.seq) + 1
        # Window hashes only need positions >= len(prompt) - max size, so the running prefix hash starts there;
        # a polynomial window hash depends only on the window, whatever the origin.
        self.h_origin = max(0, len(self.seq) - max(MATCH_SIZES))
        self.h_rel = [0]
        for x in self.seq[self.h_origin :]:
            self.h_rel.append((self.h_rel[-1] * _BASE + x + 1) & _MASK64)

    def extend(self, tokens: Sequence[int]) -> None:
        for x in tokens:
            self.seq.append(x)
            self.h_rel.append((self.h_rel[-1] * _BASE + x + 1) & _MASK64)

    def _window_hash(self, a: int, b: int) -> int:
        o = self.h_origin
        return (self.h_rel[b - o] - self.h_rel[a - o] * _POW[b - a]) & _MASK64

    def _index_to(self, end_exclusive: int) -> None:
        for e in range(self.indexed, end_exclusive):
            for n in MATCH_SIZES:
                if e - n >= self.h_origin:
                    self.recent[n][self._window_hash(e - n, e)] = e
        if end_exclusive > self.indexed:
            self.indexed = end_exclusive

    def _latest_end(self, n: int, h: int) -> int:
        e = self.recent[n].get(h)
        if e is not None:
            return e
        sh = self.sorted_h.get(n)
        if sh is None:
            return -1
        i = int(np.searchsorted(sh, np.uint64(h), side="right")) - 1
        if i >= 0 and int(sh[i]) == h:
            return int(self.sorted_end[n][i])
        return -1

    def draft(self, k: int, min_match: int) -> Tuple[int, List[int]]:
        seq = self.seq
        p = len(seq)
        self._index_to(p)
        for n in MATCH_SIZES:
            if n < min_match or p - n < self.h_origin:
                continue
            e = self._latest_end(n, self._window_hash(p - n, p))
            if 0 < e < p and seq[e - n : e] == seq[p - n : p]:
                return n, seq[e : min(e + k, p)]
        return 0, []


class PromptLookup:
    def __init__(self, min_match: int):
        self.min_match = min_match
        self._index: Dict[str, _RequestIndex] = {}

    def sync(self, rid: str, prompt: Sequence[int], output_ids: Sequence[int], pending: Sequence[int]) -> None:
        """Bring the request's sequence to prompt + output_ids + pending (accepted, not yet in output_ids)."""
        idx = self._index.get(rid)
        if idx is None:
            idx = _RequestIndex(prompt)
            self._index[rid] = idx
        n_ids = len(output_ids)
        total = n_ids + len(pending)
        if total > idx.n_out:
            tail = list(output_ids[idx.n_out :]) if idx.n_out < n_ids else []
            tail += list(pending[max(0, idx.n_out - n_ids) :])
            idx.extend(tail)
            idx.n_out = total

    def draft(self, rid: str, k: int) -> Tuple[int, List[int]]:
        return self._index[rid].draft(k, self.min_match)

    def erase(self, rids: Sequence[str]) -> None:
        for rid in rids:
            self._index.pop(rid, None)


def chain_from_tree(tokens: np.ndarray, mask: np.ndarray, k1: int) -> List[int]:
    """The first-child path of an NGRAM draft tree (the deepest anchor's recency path).

    A node's parent is its highest-index ancestor in the mask row (fillResult numbers nodes breadth first).
    Zero padding that lands on the path is harmless: the target checks every drafted token.
    """
    m = mask.reshape(k1, k1)
    path: List[int] = []
    cur = 0
    for i in range(1, k1):
        anc = np.nonzero(m[i, :i])[0]
        if len(anc) and int(anc[-1]) == cur:
            path.append(int(tokens[i]))
            cur = i
    return path


def chain_arrays(last_tokens: Sequence[int], chains: Sequence[Sequence[int]], k1: int) -> Tuple[np.ndarray, np.ndarray]:
    """Drafts and lower-triangular masks for chains padded with token 0 to k1 - 1 drafted tokens."""
    bs = len(chains)
    tok = np.zeros((bs, k1), dtype=np.int64)
    for i, (last, c) in enumerate(zip(last_tokens, chains)):
        tok[i, 0] = last
        c = list(c)[: k1 - 1]
        tok[i, 1 : 1 + len(c)] = c
    tri = np.tril(np.ones((k1, k1), dtype=bool))
    mask = np.broadcast_to(tri, (bs, k1, k1)).copy()
    return tok.reshape(-1), mask.reshape(-1)
