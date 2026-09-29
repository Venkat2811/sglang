"""Paired NCCL transport and MLA-only execution for the SQD prototype.

Both pools advance one fixed batch in lockstep. This is not a request RPC
service: admission, dynamic batching and reconnects are deliberately absent.
"""

import torch
from torch import nn

from sglang.srt.distributed.device_communicators.pynccl import PyNcclCommunicator
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.utils.common import BumpAllocator


class PairedTransport:
    def __init__(self, group, device):
        self.group = group
        self.peer = 1 - group.rank
        self.comm = PyNcclCommunicator(group, device=device)
        if not self.comm.available:
            raise RuntimeError("SQD requires an available NCCL communicator")
        self.sent_bytes = 0

    def send(self, tensor):
        if not tensor.is_contiguous():
            raise ValueError("SQD sends must be contiguous")
        with self.comm.change_state(enable=True):
            self.comm.send(tensor, self.peer)
        self.sent_bytes += tensor.numel() * tensor.element_size()

    def recv(self, tensor):
        with self.comm.change_state(enable=True):
            self.comm.recv(tensor, self.peer)
        return tensor

    def close(self):
        torch.cuda.synchronize(self.comm.device)
        self.comm.nccl.ncclCommDestroy(self.comm.comm)


class RemoteMLAAttention(nn.Module):
    """Replaces an MLA module on the KDA/FFN pool; holds no local weights."""

    def __init__(self, transport):
        super().__init__()
        self.transport = transport
        self.calls = 0

    def forward(self, hidden_states, positions, forward_batch, **kwargs):
        if not forward_batch.forward_mode.is_decode():
            raise ValueError("Remote MLA currently supports ordinary decode only")
        self.transport.send(hidden_states.contiguous())
        output = self.transport.recv(torch.empty_like(hidden_states))
        self.calls += 1
        return output


class MLAService(nn.Module):
    """Runs only MLA submodules against the prefill pool's retained KV."""

    def __init__(self, model, layer_ids, transport, hidden_size, dtype):
        super().__init__()
        self.attention = nn.ModuleList(
            [model.model.layers[i].self_attn for i in layer_ids]
        )
        self.transport = transport
        self.hidden_size = hidden_size
        self.dtype = dtype
        self.calls = 0

    def forward(self, input_ids, positions, forward_batch, **kwargs):
        if not forward_batch.forward_mode.is_decode():
            raise ValueError("MLA service accepts ordinary decode only")
        hidden = torch.empty(
            (input_ids.numel(), self.hidden_size),
            device=input_ids.device,
            dtype=self.dtype,
        )
        scratch = BumpAllocator(
            buffer_size=2 * len(self.attention),
            dtype=torch.float32,
            device=input_ids.device,
        )
        for attention in self.attention:
            self.transport.recv(hidden)
            output = attention(
                hidden_states=hidden,
                positions=positions,
                forward_batch=forward_batch,
                zero_allocator=scratch,
            )
            self.transport.send(output.contiguous())
            self.calls += 1
        return LogitsProcessorOutput(next_token_logits=None)


def transfer_recurrent_state(runner, batch, transport, sender):
    """Copy request-owned KDA slots, with independent slot IDs on each pool."""
    pool = runner.req_to_token_pool
    indices = torch.stack([req.kv.mamba_pool_idx for req in batch.reqs])
    indices = pool.translate_mamba_indices(indices)
    cache = pool.mamba_pool.mamba_cache
    tensors = [*cache.conv, cache.temporal]
    schema = [(tuple(t[:, indices].shape), str(t.dtype)) for t in tensors]
    peers = transport.group.all_gather_obj(schema)
    if peers[0] != peers[1]:
        raise ValueError(f"Incompatible recurrent state: {peers}")
    for tensor in tensors:
        if sender:
            transport.send(tensor[:, indices].contiguous())
        else:
            received = transport.recv(torch.empty_like(tensor[:, indices]))
            tensor[:, indices] = received
    # Complete the handoff before either pool advances the request.
    torch.cuda.synchronize()
