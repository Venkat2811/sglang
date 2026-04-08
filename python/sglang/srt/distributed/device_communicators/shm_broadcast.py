# Adapted from https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/distributed/device_communicators/shm_broadcast.py

import logging
import os
import pickle
import json
import socket
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from multiprocessing import shared_memory
from typing import List, Optional
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup
from zmq import IPV6  # type: ignore
from zmq import SUB, SUBSCRIBE, XPUB, XPUB_VERBOSE, Context  # type: ignore

from sglang.srt.utils.network import NetworkAddress, get_local_ip_auto, get_open_port

# SGLANG_RINGBUFFER_WARNING_INTERVAL can be set to 60
SGLANG_RINGBUFFER_WARNING_INTERVAL = int(
    os.environ.get("SGLANG_RINGBUFFER_WARNING_INTERVAL", "60")
)

logger = logging.getLogger(__name__)

# ── Myelon IPC Instrumentation ──────────────────────────────────────────
_MYELON_INSTRUMENT = os.environ.get("MYELON_INSTRUMENT", "0") == "1"
_MYELON_INSTRUMENT_JSONL = os.environ.get("MYELON_INSTRUMENT_JSONL", "").strip()
_MYELON_INSTRUMENT_TAG = os.environ.get("MYELON_INSTRUMENT_TAG", "").strip()
_MYELON_INSTRUMENT_TOPN = int(os.environ.get("MYELON_INSTRUMENT_TOPN", "8"))
_perf_ns = time.perf_counter_ns


def _append_jsonl(path: str, record: dict) -> None:
    if not path:
        return
    payload = (json.dumps(record, sort_keys=True) + "\n").encode("utf-8")
    fd = os.open(path, os.O_CREAT | os.O_WRONLY | os.O_APPEND, 0o644)
    try:
        os.write(fd, payload)
    finally:
        os.close(fd)


def _payload_signature(obj) -> str:
    obj_type = type(obj)
    type_name = f"{obj_type.__module__}.{obj_type.__qualname__}"
    try:
        if isinstance(obj, list):
            head = type(obj[0]).__qualname__ if obj else "empty"
            return f"{type_name}[len={len(obj)},head={head}]"
        if isinstance(obj, tuple):
            head = type(obj[0]).__qualname__ if obj else "empty"
            return f"{type_name}[len={len(obj)},head={head}]"
        if isinstance(obj, dict):
            return f"{type_name}[len={len(obj)}]"
        if isinstance(obj, (bytes, bytearray, memoryview)):
            return f"{type_name}[len={len(obj)}]"
        if hasattr(obj, "shape"):
            return f"{type_name}[shape={getattr(obj, 'shape', None)}]"
        if hasattr(obj, "__len__") and not isinstance(obj, str):
            return f"{type_name}[len={len(obj)}]"
    except Exception:
        pass
    return type_name


class _MyelonIpcStats:
    """Nanosecond-granular IPC timing accumulator for shm_broadcast.
    Tracks: pickle, shm_write/read, zmq_send/recv, unpickle, payload sizes.
    Active only when MYELON_INSTRUMENT=1."""

    __slots__ = (
        "name",
        "pid", "hostname", "queue_config", "tag", "dumped",
        # enqueue (writer) stats
        "enq_count", "enq_total_ns", "enq_max_ns", "enq_min_ns",
        "enq_inline_count", "enq_remote_send_count",
        "pickle_total_ns", "pickle_max_ns",
        "shm_write_total_ns", "shm_write_max_ns", "shm_write_count",
        "zmq_send_total_ns", "zmq_send_max_ns", "zmq_send_count",
        "enq_bytes_total", "enq_bytes_max", "enq_bytes_min",
        "enq_overflow_count",
        # dequeue (reader) stats
        "deq_count", "deq_total_ns", "deq_max_ns", "deq_min_ns",
        "deq_inline_count", "deq_zmq_count",
        "shm_read_total_ns", "shm_read_max_ns", "shm_read_count",
        "zmq_recv_total_ns", "zmq_recv_max_ns", "zmq_recv_count",
        "unpickle_total_ns", "unpickle_max_ns",
        "deq_bytes_total", "deq_bytes_max", "deq_bytes_min",
        "deq_bytes_known_count",
        # acquire contention
        "write_wait_total_ns", "write_wait_max_ns", "write_wait_count",
        "read_wait_total_ns", "read_wait_max_ns", "read_wait_count",
        # payload classification
        "payload_type_counts",
        "payload_type_max_bytes",
        "payload_type_overflow_counts",
    )

    def __init__(self, name: str, queue_config: Optional[dict] = None):
        self.name = name
        self.pid = os.getpid()
        self.hostname = socket.gethostname()
        self.queue_config = queue_config or {}
        self.tag = _MYELON_INSTRUMENT_TAG
        self.dumped = False
        self.payload_type_counts = {}
        self.payload_type_max_bytes = {}
        self.payload_type_overflow_counts = {}
        for slot in self.__slots__:
            if slot in {
                "name",
                "pid",
                "hostname",
                "queue_config",
                "tag",
                "dumped",
                "payload_type_counts",
                "payload_type_max_bytes",
                "payload_type_overflow_counts",
            }:
                continue
            if "min" in slot:
                setattr(self, slot, 2**63)
            else:
                setattr(self, slot, 0)

    def record_enqueue(
        self,
        total_ns,
        pickle_ns,
        transport_ns,
        payload_bytes,
        is_overflow,
        wait_ns=0,
        obj_type: Optional[str] = None,
        sent_remote: bool = False,
    ):
        self.enq_count += 1
        self.enq_total_ns += total_ns
        if total_ns > self.enq_max_ns:
            self.enq_max_ns = total_ns
        if total_ns < self.enq_min_ns:
            self.enq_min_ns = total_ns
        self.pickle_total_ns += pickle_ns
        if pickle_ns > self.pickle_max_ns:
            self.pickle_max_ns = pickle_ns
        if is_overflow:
            self.zmq_send_total_ns += transport_ns
            if transport_ns > self.zmq_send_max_ns:
                self.zmq_send_max_ns = transport_ns
            self.enq_overflow_count += 1
            self.zmq_send_count += 1
        else:
            self.shm_write_total_ns += transport_ns
            if transport_ns > self.shm_write_max_ns:
                self.shm_write_max_ns = transport_ns
            self.shm_write_count += 1
            self.enq_inline_count += 1
        if sent_remote:
            self.enq_remote_send_count += 1
        self.enq_bytes_total += payload_bytes
        if payload_bytes > self.enq_bytes_max:
            self.enq_bytes_max = payload_bytes
        if payload_bytes < self.enq_bytes_min:
            self.enq_bytes_min = payload_bytes
        if wait_ns > 0:
            self.write_wait_total_ns += wait_ns
            if wait_ns > self.write_wait_max_ns:
                self.write_wait_max_ns = wait_ns
            self.write_wait_count += 1
        if obj_type:
            self.payload_type_counts[obj_type] = (
                self.payload_type_counts.get(obj_type, 0) + 1
            )
            self.payload_type_max_bytes[obj_type] = max(
                payload_bytes,
                self.payload_type_max_bytes.get(obj_type, 0),
            )
            if is_overflow:
                self.payload_type_overflow_counts[obj_type] = (
                    self.payload_type_overflow_counts.get(obj_type, 0) + 1
                )

    def record_dequeue(
        self,
        total_ns,
        transport_ns,
        unpickle_ns,
        payload_bytes,
        is_zmq,
        wait_ns=0,
    ):
        self.deq_count += 1
        self.deq_total_ns += total_ns
        if total_ns > self.deq_max_ns:
            self.deq_max_ns = total_ns
        if total_ns < self.deq_min_ns:
            self.deq_min_ns = total_ns
        if is_zmq:
            self.zmq_recv_total_ns += transport_ns
            if transport_ns > self.zmq_recv_max_ns:
                self.zmq_recv_max_ns = transport_ns
            self.zmq_recv_count += 1
            self.deq_zmq_count += 1
        else:
            self.shm_read_total_ns += transport_ns
            if transport_ns > self.shm_read_max_ns:
                self.shm_read_max_ns = transport_ns
            self.shm_read_count += 1
            self.deq_inline_count += 1
        self.unpickle_total_ns += unpickle_ns
        if unpickle_ns > self.unpickle_max_ns:
            self.unpickle_max_ns = unpickle_ns
        if payload_bytes is not None:
            self.deq_bytes_total += payload_bytes
            if payload_bytes > self.deq_bytes_max:
                self.deq_bytes_max = payload_bytes
            if payload_bytes < self.deq_bytes_min:
                self.deq_bytes_min = payload_bytes
            self.deq_bytes_known_count += 1
        if wait_ns > 0:
            self.read_wait_total_ns += wait_ns
            if wait_ns > self.read_wait_max_ns:
                self.read_wait_max_ns = wait_ns
            self.read_wait_count += 1

    @staticmethod
    def _fmt(ns):
        if ns >= 1_000_000:
            return f"{ns / 1_000_000:.1f}ms"
        if ns >= 1_000:
            return f"{ns / 1_000:.1f}us"
        return f"{ns}ns"

    def _avg(self, total, count):
        return total // count if count > 0 else 0

    def _top_payload_types(self):
        rows = []
        for obj_type, count in self.payload_type_counts.items():
            rows.append(
                {
                    "type": obj_type,
                    "count": count,
                    "max_bytes": self.payload_type_max_bytes.get(obj_type, 0),
                    "overflow_count": self.payload_type_overflow_counts.get(obj_type, 0),
                }
            )
        rows.sort(
            key=lambda row: (
                row["overflow_count"],
                row["max_bytes"],
                row["count"],
                row["type"],
            ),
            reverse=True,
        )
        return rows[:_MYELON_INSTRUMENT_TOPN]

    def as_dict(self):
        a = self._avg
        return {
            "kind": "myelon_ipc_stats",
            "tag": self.tag,
            "name": self.name,
            "pid": self.pid,
            "hostname": self.hostname,
            "queue_config": self.queue_config,
            "enqueue": {
                "count": self.enq_count,
                "inline_count": self.enq_inline_count,
                "overflow_count": self.enq_overflow_count,
                "remote_send_count": self.enq_remote_send_count,
                "avg_ns": a(self.enq_total_ns, self.enq_count),
                "min_ns": 0 if self.enq_count == 0 else self.enq_min_ns,
                "max_ns": self.enq_max_ns,
                "pickle_avg_ns": a(self.pickle_total_ns, self.enq_count),
                "pickle_max_ns": self.pickle_max_ns,
                "shm_write_avg_ns": a(self.shm_write_total_ns, self.shm_write_count),
                "shm_write_max_ns": self.shm_write_max_ns,
                "zmq_send_avg_ns": a(self.zmq_send_total_ns, self.zmq_send_count),
                "zmq_send_max_ns": self.zmq_send_max_ns,
                "payload_avg_bytes": a(self.enq_bytes_total, self.enq_count),
                "payload_min_bytes": 0 if self.enq_count == 0 else self.enq_bytes_min,
                "payload_max_bytes": self.enq_bytes_max,
                "write_wait_avg_ns": a(self.write_wait_total_ns, self.write_wait_count),
                "write_wait_max_ns": self.write_wait_max_ns,
                "write_wait_count": self.write_wait_count,
            },
            "dequeue": {
                "count": self.deq_count,
                "inline_count": self.deq_inline_count,
                "zmq_count": self.deq_zmq_count,
                "avg_ns": a(self.deq_total_ns, self.deq_count),
                "min_ns": 0 if self.deq_count == 0 else self.deq_min_ns,
                "max_ns": self.deq_max_ns,
                "shm_read_avg_ns": a(self.shm_read_total_ns, self.shm_read_count),
                "shm_read_max_ns": self.shm_read_max_ns,
                "zmq_recv_avg_ns": a(self.zmq_recv_total_ns, self.zmq_recv_count),
                "zmq_recv_max_ns": self.zmq_recv_max_ns,
                "unpickle_avg_ns": a(self.unpickle_total_ns, self.deq_count),
                "unpickle_max_ns": self.unpickle_max_ns,
                "payload_known_count": self.deq_bytes_known_count,
                "payload_avg_bytes": a(self.deq_bytes_total, self.deq_bytes_known_count),
                "payload_min_bytes": (
                    0 if self.deq_bytes_known_count == 0 else self.deq_bytes_min
                ),
                "payload_max_bytes": self.deq_bytes_max,
                "read_wait_avg_ns": a(self.read_wait_total_ns, self.read_wait_count),
                "read_wait_max_ns": self.read_wait_max_ns,
                "read_wait_count": self.read_wait_count,
            },
            "payload_types": self._top_payload_types(),
        }

    def dump(self):
        if self.dumped or (self.enq_count == 0 and self.deq_count == 0):
            return
        self.dumped = True
        f = self._fmt
        a = self._avg
        lines = [f"[MyelonInstr] === {self.name} shm_broadcast Stats ==="]
        if self.enq_count > 0:
            avg_bytes = a(self.enq_bytes_total, self.enq_count)
            lines.append(
                f"[MyelonInstr]   enqueue: n={self.enq_count} "
                f"avg={f(a(self.enq_total_ns, self.enq_count))} "
                f"min={f(self.enq_min_ns)} max={f(self.enq_max_ns)} "
                f"avg_bytes={avg_bytes} max_bytes={self.enq_bytes_max}"
            )
            lines.append(
                f"[MyelonInstr]     pickle:    avg={f(a(self.pickle_total_ns, self.enq_count))} "
                f"max={f(self.pickle_max_ns)}"
            )
            lines.append(
                f"[MyelonInstr]     shm_write: avg={f(a(self.shm_write_total_ns, self.shm_write_count))} "
                f"max={f(self.shm_write_max_ns)} "
                f"(n={self.shm_write_count})"
            )
            if self.enq_overflow_count > 0:
                lines.append(
                    f"[MyelonInstr]     zmq_send:  avg={f(a(self.zmq_send_total_ns, self.enq_overflow_count))} "
                    f"max={f(self.zmq_send_max_ns)} "
                    f"(overflow n={self.enq_overflow_count})"
                )
            lines.append(
                f"[MyelonInstr]     path_mix:   inline={self.enq_inline_count} "
                f"overflow={self.enq_overflow_count} remote_send={self.enq_remote_send_count}"
            )
            if self.write_wait_count > 0:
                lines.append(
                    f"[MyelonInstr]     write_wait: avg={f(a(self.write_wait_total_ns, self.write_wait_count))} "
                    f"max={f(self.write_wait_max_ns)} n={self.write_wait_count}"
                )
            top_payload_types = self._top_payload_types()
            if top_payload_types:
                lines.append(
                    "[MyelonInstr]     top_payload_types: "
                    + ", ".join(
                        (
                            f"{row['type']} count={row['count']} "
                            f"max_bytes={row['max_bytes']} "
                            f"overflow={row['overflow_count']}"
                        )
                        for row in top_payload_types
                    )
                )
        if self.deq_count > 0:
            avg_bytes = a(self.deq_bytes_total, self.deq_bytes_known_count)
            lines.append(
                f"[MyelonInstr]   dequeue: n={self.deq_count} "
                f"avg={f(a(self.deq_total_ns, self.deq_count))} "
                f"min={f(self.deq_min_ns)} max={f(self.deq_max_ns)} "
                f"known_avg_bytes={avg_bytes} max_bytes={self.deq_bytes_max}"
            )
            lines.append(
                f"[MyelonInstr]     shm_read:  avg={f(a(self.shm_read_total_ns, self.shm_read_count))} "
                f"max={f(self.shm_read_max_ns)} (n={self.shm_read_count})"
            )
            lines.append(
                f"[MyelonInstr]     zmq_recv:  avg={f(a(self.zmq_recv_total_ns, self.zmq_recv_count))} "
                f"max={f(self.zmq_recv_max_ns)} (n={self.zmq_recv_count})"
            )
            lines.append(
                f"[MyelonInstr]     unpickle:  avg={f(a(self.unpickle_total_ns, self.deq_count))} "
                f"max={f(self.unpickle_max_ns)}"
            )
            lines.append(
                f"[MyelonInstr]     path_mix:   inline={self.deq_inline_count} "
                f"zmq={self.deq_zmq_count}"
            )
            if self.read_wait_count > 0:
                lines.append(
                    f"[MyelonInstr]     read_wait: avg={f(a(self.read_wait_total_ns, self.read_wait_count))} "
                    f"max={f(self.read_wait_max_ns)} n={self.read_wait_count}"
                )
        for line in lines:
            logger.info(line)
        _append_jsonl(_MYELON_INSTRUMENT_JSONL, self.as_dict())
# ── End Myelon IPC Instrumentation ──────────────────────────────────────


class ShmRingBuffer:

    def __init__(
        self,
        n_reader: int,
        max_chunk_bytes: int,
        max_chunks: int,
        name: Optional[str] = None,
    ):
        """
        A shared memory ring buffer implementation for broadcast communication.
        Essentially, it is a queue where only one will `enqueue` and multiple
        will `dequeue`. The max size of each item, together with the max number
        of items that can be stored in the buffer are known in advance.
        In this case, we don't need to synchronize the access to
         the buffer.

        Buffer memory layout:
                  data                                 metadata
                    |                                      |
                    | (current_idx)                        | (current_idx)
                    v                                      v
        +-------------------------------+----------------------------------------+
        | chunk0 | chunk1 | ... | chunk | metadata0 | metadata1 | ... | metadata |
        +-------------------------------+----------------------------------------+
        | max_chunks x max_chunk_bytes  | max_chunks x (1 + n_reader) bytes      |

        metadata memory layout: each byte is a flag, the first byte is the written
        flag, and the rest are reader flags. The flags are set to 0 by default.
        +--------------+--------------+--------------+-----+--------------+
        | written_flag | reader0_flag | reader1_flag | ... | readerN_flag |
        +--------------+--------------+--------------+-----+--------------+

        The state of metadata is as follows:

        (case 1) 0???...???: the block is not written yet, cannot read, can write
        (case 2) 1000...000: the block is just written, can read, cannot write
        (case 3) 1???...???: the block is written and read by some readers, can read if not read, cannot write
        (case 4) 1111...111: the block is written and read by all readers, cannot read, can write

        State transition for readers:

        When a reader finds a block that it can read (case 2 or 3), it can yield the block for caller to read.
        Only after the caller finishes reading the block, the reader can mark the block as read.
        Readers only mark the block as read (from 0 to 1), the writer marks the block as ready to read (from 1 to 0).

        State transition for writer:

        When the writer writes to a block (case 1 or 4), it first resets the written flag to 0, converting either case
        to case 1. Then it can yield the block for caller to write. After the caller finishes writing the block, the writer
        can reset the reader flags to 0, and mark the block as written (from 0 to 1).
        NOTE: the order is important here, first reset the reader flags (so that we are still in case 1), then mark the block as written. The state transition is atomic. If we do it in the reverse order, it will go through case 3 and then back to case 2, and readers might read the intermediate case 3, which is not correct.

        During creation, `name` is None and the buffer is created. We can pass the
        created object to other processes by pickling it. The other processes will
        get the name of the shared memory and open it, so that they can access the
        same shared memory buffer.
        """  # noqa
        self.n_reader = n_reader
        self.metadata_size = 1 + n_reader
        self.max_chunk_bytes = max_chunk_bytes
        self.max_chunks = max_chunks
        self.total_bytes_of_buffer = (
            self.max_chunk_bytes + self.metadata_size
        ) * self.max_chunks
        self.data_offset = 0
        self.metadata_offset = self.max_chunk_bytes * self.max_chunks

        if name is None:
            # we are creating a buffer
            self.is_creator = True
            self.shared_memory = shared_memory.SharedMemory(
                create=True, size=self.total_bytes_of_buffer
            )
            # initialize the metadata section to 0
            with memoryview(
                self.shared_memory.buf[self.metadata_offset :]
            ) as metadata_buffer:
                torch.frombuffer(metadata_buffer, dtype=torch.uint8).fill_(0)
        else:
            # we are opening an existing buffer
            self.is_creator = False
            # fix to https://stackoverflow.com/q/62748654/9191338
            # Python incorrectly tracks shared memory even if it is not
            # created by the process. The following patch is a workaround.
            with patch(
                "multiprocessing.resource_tracker.register",
                lambda *args, **kwargs: None,
            ):
                try:
                    self.shared_memory = shared_memory.SharedMemory(name=name)
                    assert self.shared_memory.size == self.total_bytes_of_buffer
                except FileNotFoundError:
                    # we might deserialize the object in a different node
                    # in this case, this object is not used,
                    # and we should suppress the error
                    pass

    def __reduce__(self):
        return (
            self.__class__,
            (
                self.n_reader,
                self.max_chunk_bytes,
                self.max_chunks,
                self.shared_memory.name,
            ),
        )

    def __del__(self):
        if hasattr(self, "shared_memory"):
            self.shared_memory.close()
            if self.is_creator:
                self.shared_memory.unlink()

    @contextmanager
    def get_data(self, current_idx: int):
        start = self.data_offset + current_idx * self.max_chunk_bytes
        end = start + self.max_chunk_bytes
        with memoryview(self.shared_memory.buf[start:end]) as buf:
            yield buf

    @contextmanager
    def get_metadata(self, current_idx: int):
        start = self.metadata_offset + current_idx * self.metadata_size
        end = start + self.metadata_size
        with memoryview(self.shared_memory.buf[start:end]) as buf:
            yield buf


@dataclass
class Handle:
    connect_ip: str
    local_reader_ranks: List[int] = field(default_factory=list)

    buffer: Optional[ShmRingBuffer] = None
    local_subscribe_port: Optional[int] = None
    remote_subscribe_port: Optional[int] = None


class MessageQueue:

    def __init__(
        self,
        n_reader,  # number of all readers
        n_local_reader,  # number of local readers through shared memory
        local_reader_ranks: Optional[List[int]] = None,
        max_chunk_bytes: int = 1024 * 1024 * 10,
        max_chunks: int = 10,
        connect_ip: Optional[str] = None,
    ):
        if local_reader_ranks is None:
            local_reader_ranks = list(range(n_local_reader))
        else:
            assert len(local_reader_ranks) == n_local_reader
        self.n_local_reader = n_local_reader
        n_remote_reader = n_reader - n_local_reader
        self.n_remote_reader = n_remote_reader

        if connect_ip is None:
            connect_ip = (
                get_local_ip_auto("0.0.0.0") if n_remote_reader > 0 else "127.0.0.1"
            )

        context = Context()

        if n_local_reader > 0:
            # for local readers, we will:
            # 1. create a shared memory ring buffer to communicate small data
            # 2. create a publish-subscribe socket to communicate large data
            self.buffer = ShmRingBuffer(n_local_reader, max_chunk_bytes, max_chunks)

            # XPUB is very similar to PUB,
            # except that it can receive subscription messages
            # to confirm the number of subscribers
            self.local_socket = context.socket(XPUB)
            # set the verbose option so that we can receive every subscription
            # message. otherwise, we will only receive the first subscription
            # see http://api.zeromq.org/3-3:zmq-setsockopt for more details
            self.local_socket.setsockopt(XPUB_VERBOSE, True)
            local_subscribe_port = get_open_port()
            socket_addr = f"tcp://127.0.0.1:{local_subscribe_port}"
            logger.debug("Binding to %s", socket_addr)
            self.local_socket.bind(socket_addr)
            self.current_idx = 0

        else:
            self.buffer = None  # type: ignore
            local_subscribe_port = None
            self.local_socket = None
            self.current_idx = -1

        if n_remote_reader > 0:
            # for remote readers, we will:
            # create a publish-subscribe socket to communicate large data
            self.remote_socket = context.socket(XPUB)
            self.remote_socket.setsockopt(XPUB_VERBOSE, True)
            remote_subscribe_port = get_open_port()
            na = NetworkAddress(connect_ip, remote_subscribe_port)
            if na.is_ipv6:
                self.remote_socket.setsockopt(IPV6, 1)
            address = na.to_tcp()
            logger.debug(f"class MessageQueue: Binding remote socket to {address=}")
            self.remote_socket.bind(address)

        else:
            remote_subscribe_port = None
            self.remote_socket = None

        self._is_writer = True
        self._is_local_reader = False
        self.local_reader_rank = -1
        # rank does not matter for remote readers
        self._is_remote_reader = False
        self._last_write_wait_ns = 0
        self._last_read_wait_ns = 0
        self._stats = (
            _MyelonIpcStats(
                "writer",
                {
                    "role": "writer",
                    "n_reader": n_reader,
                    "n_local_reader": n_local_reader,
                    "n_remote_reader": n_remote_reader,
                    "max_chunk_bytes": max_chunk_bytes,
                    "max_chunks": max_chunks,
                },
            )
            if _MYELON_INSTRUMENT
            else None
        )

        self.handle = Handle(
            connect_ip=connect_ip,
            local_reader_ranks=local_reader_ranks,
            buffer=self.buffer,
            local_subscribe_port=local_subscribe_port,
            remote_subscribe_port=remote_subscribe_port,
        )

        logger.debug("Message queue communication handle: %s", self.handle)

    def export_handle(self) -> Handle:
        return self.handle

    @staticmethod
    def create_from_handle(handle: Handle, rank) -> "MessageQueue":
        self = MessageQueue.__new__(MessageQueue)
        self.handle = handle
        self._is_writer = False
        self._last_write_wait_ns = 0
        self._last_read_wait_ns = 0

        context = Context()

        if rank in handle.local_reader_ranks:
            assert handle.buffer is not None
            self.buffer = handle.buffer
            self.current_idx = 0
            self.local_reader_rank = handle.local_reader_ranks.index(rank)
            self._is_local_reader = True
            self._is_remote_reader = False

            self.local_socket = context.socket(SUB)
            self.local_socket.setsockopt_string(SUBSCRIBE, "")
            socket_addr = f"tcp://127.0.0.1:{handle.local_subscribe_port}"
            logger.debug("Connecting to %s", socket_addr)
            self.local_socket.connect(socket_addr)

            self.remote_socket = None
        else:
            self.buffer = None  # type: ignore
            self.current_idx = -1
            self.local_reader_rank = -1
            self._is_local_reader = False
            self._is_remote_reader = True

            self.local_socket = None

            self.remote_socket = context.socket(SUB)
            self.remote_socket.setsockopt_string(SUBSCRIBE, "")
            na = NetworkAddress(handle.connect_ip, handle.remote_subscribe_port)
            if na.is_ipv6:
                self.remote_socket.setsockopt(IPV6, 1)
            socket_addr = na.to_tcp()
            logger.debug("Connecting to %s", socket_addr)
            self.remote_socket.connect(socket_addr)

        self._stats = (
            _MyelonIpcStats(
                f"reader-{rank}",
                {
                    "role": (
                        "local_reader" if self._is_local_reader else "remote_reader"
                    ),
                    "rank": rank,
                    "local_reader_rank": self.local_reader_rank,
                    "n_local_reader": len(handle.local_reader_ranks),
                    "max_chunk_bytes": (
                        handle.buffer.max_chunk_bytes if handle.buffer is not None else None
                    ),
                    "max_chunks": (
                        handle.buffer.max_chunks if handle.buffer is not None else None
                    ),
                },
            )
            if _MYELON_INSTRUMENT
            else None
        )
        return self

    def wait_until_ready(self):
        """This is a collective operation. All processes (including the
        readers and the writer) should call this function.
        """
        if self._is_writer:
            # wait for all readers to connect

            # local readers
            for i in range(self.n_local_reader):
                # wait for subscription messages from all local readers
                self.local_socket.recv()
            if self.n_local_reader > 0:
                # send a message to all local readers
                # to make sure the publish channel is working
                self.local_socket.send(b"READY")

            # remote readers
            for i in range(self.n_remote_reader):
                # wait for subscription messages from all remote readers
                self.remote_socket.recv()
            if self.n_remote_reader > 0:
                # send a message to all remote readers
                # to make sure the publish channel is working
                self.remote_socket.send(b"READY")
        elif self._is_local_reader:
            # wait for the writer to send a message
            recv = self.local_socket.recv()
            assert recv == b"READY"
        elif self._is_remote_reader:
            # wait for the writer to send a message
            recv = self.remote_socket.recv()
            assert recv == b"READY"

    @contextmanager
    def acquire_write(self):
        assert self._is_writer, "Only writers can acquire write"
        start_time = time.monotonic()
        wait_start_ns = _perf_ns() if self._stats is not None else 0
        n_warning = 1
        while True:
            with self.buffer.get_metadata(self.current_idx) as metadata_buffer:
                read_count = sum(metadata_buffer[1:])
                written_flag = metadata_buffer[0]
                if written_flag and read_count != self.buffer.n_reader:
                    # this block is written and not read by all readers
                    # for writers, `self.current_idx` is the next block to write
                    # if this block is not ready to write,
                    # we need to wait until it is read by all readers

                    # Release the processor to other threads
                    os.sched_yield()

                    # if we wait for a long time, we should warn the user
                    if (
                        time.monotonic() - start_time
                        > SGLANG_RINGBUFFER_WARNING_INTERVAL * n_warning
                    ):
                        logger.warning(
                            "No available block found in %s second. ",
                            SGLANG_RINGBUFFER_WARNING_INTERVAL,
                        )
                        n_warning += 1

                    continue
                # found a block that is either
                # (1) not written
                # (2) read by all readers

                # mark the block as not written
                metadata_buffer[0] = 0
                if wait_start_ns:
                    self._last_write_wait_ns = _perf_ns() - wait_start_ns
                # let caller write to the buffer
                with self.buffer.get_data(self.current_idx) as buf:
                    yield buf

                # caller has written to the buffer
                # NOTE: order is important here
                # first set the read flags to 0
                # then set the written flag to 1
                # otherwise, the readers may think they already read the block
                for i in range(1, self.buffer.n_reader + 1):
                    # set read flag to 0, meaning it is not read yet
                    metadata_buffer[i] = 0
                # mark the block as written
                metadata_buffer[0] = 1
                self.current_idx = (self.current_idx + 1) % self.buffer.max_chunks
                break

    @contextmanager
    def acquire_read(self):
        assert self._is_local_reader, "Only readers can acquire read"
        start_time = time.monotonic()
        wait_start_ns = _perf_ns() if self._stats is not None else 0
        n_warning = 1
        while True:
            with self.buffer.get_metadata(self.current_idx) as metadata_buffer:
                read_flag = metadata_buffer[self.local_reader_rank + 1]
                written_flag = metadata_buffer[0]
                if not written_flag or read_flag:
                    # this block is either
                    # (1) not written
                    # (2) already read by this reader

                    # for readers, `self.current_idx` is the next block to read
                    # if this block is not ready,
                    # we need to wait until it is written

                    # Release the processor to other threads
                    os.sched_yield()

                    # if we wait for a long time, we should warn the user
                    if (
                        time.monotonic() - start_time
                        > SGLANG_RINGBUFFER_WARNING_INTERVAL * n_warning
                    ):
                        logger.warning(
                            "No available block found in %s second. ",
                            SGLANG_RINGBUFFER_WARNING_INTERVAL,
                        )
                        n_warning += 1

                    continue
                # found a block that is not read by this reader
                if wait_start_ns:
                    self._last_read_wait_ns = _perf_ns() - wait_start_ns
                # let caller read from the buffer
                with self.buffer.get_data(self.current_idx) as buf:
                    yield buf

                # caller has read from the buffer
                # set the read flag
                metadata_buffer[self.local_reader_rank + 1] = 1
                self.current_idx = (self.current_idx + 1) % self.buffer.max_chunks
                break

    def enqueue(self, obj):
        assert self._is_writer, "Only writers can enqueue"
        if self._stats is not None:
            t0 = _perf_ns()
            obj_type = _payload_signature(obj)
            serialized_obj = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
            t_pickle = _perf_ns()
            payload_bytes = len(serialized_obj)
            is_overflow = False
            if self.n_local_reader > 0:
                if payload_bytes >= self.buffer.max_chunk_bytes:
                    is_overflow = True
                    with self.acquire_write() as buf:
                        buf[0] = 1  # overflow
                    self.local_socket.send(serialized_obj)
                else:
                    with self.acquire_write() as buf:
                        buf[0] = 0  # not overflow
                        buf[1 : payload_bytes + 1] = serialized_obj
            t_transport = _perf_ns()
            if self.n_remote_reader > 0:
                self.remote_socket.send(serialized_obj)
            t_end = _perf_ns()
            self._stats.record_enqueue(
                t_end - t0, t_pickle - t0, t_transport - t_pickle,
                payload_bytes, is_overflow,
                wait_ns=self._last_write_wait_ns,
                obj_type=obj_type,
                sent_remote=self.n_remote_reader > 0,
            )
        else:
            serialized_obj = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
            if self.n_local_reader > 0:
                if len(serialized_obj) >= self.buffer.max_chunk_bytes:
                    with self.acquire_write() as buf:
                        buf[0] = 1  # overflow
                    self.local_socket.send(serialized_obj)
                else:
                    with self.acquire_write() as buf:
                        buf[0] = 0  # not overflow
                        buf[1 : len(serialized_obj) + 1] = serialized_obj
            if self.n_remote_reader > 0:
                self.remote_socket.send(serialized_obj)

    def dequeue(self):
        if self._stats is not None:
            t0 = _perf_ns()
            is_zmq = False
            payload_bytes = None
            if self._is_local_reader:
                with self.acquire_read() as buf:
                    overflow = buf[0] == 1
                    if not overflow:
                        t_transport = _perf_ns()
                        obj = pickle.loads(buf[1:])
                        t_unpickle = _perf_ns()
                if overflow:
                    recv = self.local_socket.recv()
                    t_transport = _perf_ns()
                    payload_bytes = len(recv)
                    obj = pickle.loads(recv)
                    t_unpickle = _perf_ns()
                    is_zmq = True
            elif self._is_remote_reader:
                recv = self.remote_socket.recv()
                t_transport = _perf_ns()
                payload_bytes = len(recv)
                obj = pickle.loads(recv)
                t_unpickle = _perf_ns()
                is_zmq = True
            else:
                raise RuntimeError("Only readers can dequeue")
            t_end = _perf_ns()
            self._stats.record_dequeue(
                t_end - t0, t_transport - t0, t_unpickle - t_transport,
                payload_bytes, is_zmq, wait_ns=self._last_read_wait_ns)
            return obj
        else:
            if self._is_local_reader:
                with self.acquire_read() as buf:
                    overflow = buf[0] == 1
                    if not overflow:
                        obj = pickle.loads(buf[1:])
                if overflow:
                    recv = self.local_socket.recv()
                    obj = pickle.loads(recv)
            elif self._is_remote_reader:
                recv = self.remote_socket.recv()
                obj = pickle.loads(recv)
            else:
                raise RuntimeError("Only readers can dequeue")
            return obj

    def broadcast_object(self, obj=None):
        if self._is_writer:
            self.enqueue(obj)
            return obj
        else:
            return self.dequeue()

    def dump_stats(self):
        if self._stats is not None:
            self._stats.dump()

    def __del__(self):
        if hasattr(self, "_stats") and self._stats is not None:
            self._stats.dump()

    @staticmethod
    def create_from_process_group(
        pg: ProcessGroup, max_chunk_bytes, max_chunks, writer_rank=0
    ) -> "MessageQueue":
        group_rank = dist.get_rank(pg)
        group_world_size = dist.get_world_size(pg)
        global_ranks = dist.get_process_group_ranks(pg)

        from sglang.srt.distributed.parallel_state import in_the_same_node_as

        status = in_the_same_node_as(pg, source_rank=writer_rank)
        same_node_ranks = [i for i, s in enumerate(status) if s]
        n_reader = group_world_size - 1
        n_local_reader = len(same_node_ranks) - 1
        local_reader_ranks = [i for i in same_node_ranks if i != writer_rank]
        buffer_io: MessageQueue
        if group_rank == writer_rank:
            buffer_io = MessageQueue(
                n_reader=n_reader,
                n_local_reader=n_local_reader,
                local_reader_ranks=local_reader_ranks,
                max_chunk_bytes=max_chunk_bytes,
                max_chunks=max_chunks,
            )
            handle = buffer_io.export_handle()
            dist.broadcast_object_list(
                [handle], src=global_ranks[writer_rank], group=pg
            )
        else:
            recv = [None]
            dist.broadcast_object_list(recv, src=global_ranks[writer_rank], group=pg)
            handle = recv[0]  # type: ignore
            buffer_io = MessageQueue.create_from_handle(handle, group_rank)
        buffer_io.wait_until_ready()
        return buffer_io
