import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from sglang.benchmark.sqd_transport import (
    MLAService,
    RemoteMLAAttention,
    transfer_recurrent_state,
)


class MemoryTransport:
    def __init__(self, incoming=()):
        self.incoming = iter(incoming)
        self.sent = []
        self.group = SimpleNamespace(all_gather_obj=lambda schema: [schema, schema])

    def send(self, tensor):
        self.sent.append(tensor.clone())

    def recv(self, tensor):
        return tensor.copy_(next(self.incoming))


class TestSplitExecution(unittest.TestCase):
    def batch(self, decode=True):
        return SimpleNamespace(forward_mode=SimpleNamespace(is_decode=lambda: decode))

    def test_remote_boundary_returns_peer_output_without_local_attention(self):
        hidden = torch.arange(12).reshape(3, 4).float()
        transport = MemoryTransport([hidden + 7])
        attention = RemoteMLAAttention(transport)
        result = attention(hidden, torch.arange(3), self.batch())
        torch.testing.assert_close(result, hidden + 7)
        torch.testing.assert_close(transport.sent[0], hidden)
        self.assertEqual(list(attention.parameters()), [])
        self.assertEqual(attention.calls, 1)
        with self.assertRaises(ValueError):
            attention(hidden, torch.arange(3), self.batch(decode=False))
        self.assertEqual(len(transport.sent), 1)

    def test_service_calls_only_selected_mla_layers(self):
        calls = []

        class Attention(nn.Module):
            def __init__(self, layer):
                super().__init__()
                self.layer = layer

            def forward(self, hidden_states, positions, forward_batch, **kwargs):
                calls.append(self.layer)
                return hidden_states + self.layer

        model = SimpleNamespace(
            model=SimpleNamespace(
                layers=[SimpleNamespace(self_attn=Attention(i)) for i in range(8)]
            )
        )
        incoming = [torch.full((2, 4), 3.0), torch.full((2, 4), 11.0)]
        transport = MemoryTransport(incoming)
        service = MLAService(model, (3, 7), transport, 4, torch.float32)
        result = service(torch.tensor([10, 20]), torch.tensor([17, 31]), self.batch())
        self.assertEqual(calls, [3, 7])
        self.assertIsNone(result.next_token_logits)
        torch.testing.assert_close(transport.sent[0], incoming[0] + 3)
        torch.testing.assert_close(transport.sent[1], incoming[1] + 7)

    @patch("torch.cuda.synchronize")
    def test_handoff_maps_request_owned_slots_and_preserves_other_requests(self, _):
        def runner(fill):
            cache = SimpleNamespace(
                conv=[torch.full((2, 6, 3), fill)],
                temporal=torch.full((2, 6, 4, 4), fill),
            )
            return SimpleNamespace(
                req_to_token_pool=SimpleNamespace(
                    mamba_pool=SimpleNamespace(mamba_cache=cache),
                    translate_mamba_indices=lambda indices: indices,
                )
            )

        def batch(slots):
            return SimpleNamespace(
                reqs=[
                    SimpleNamespace(kv=SimpleNamespace(mamba_pool_idx=torch.tensor(i)))
                    for i in slots
                ]
            )

        source, target = runner(0.0), runner(-1.0)
        src_cache = source.req_to_token_pool.mamba_pool.mamba_cache
        for tensor in [*src_cache.conv, src_cache.temporal]:
            tensor[:, 4] = 42
            tensor[:, 1] = 17
        send = MemoryTransport()
        transfer_recurrent_state(source, batch([4, 1]), send, sender=True)
        self.assertEqual(len(send.sent), len(src_cache.conv) + 1)
        transfer_recurrent_state(
            target, batch([2, 5]), MemoryTransport(send.sent), sender=False
        )
        dst_cache = target.req_to_token_pool.mamba_pool.mamba_cache
        for tensor in [*dst_cache.conv, dst_cache.temporal]:
            self.assertTrue((tensor[:, 2] == 42).all())
            self.assertTrue((tensor[:, 5] == 17).all())
            self.assertTrue((tensor[:, [0, 1, 3, 4]] == -1).all())
        incompatible = MemoryTransport()
        incompatible.group.all_gather_obj = lambda schema: [schema, []]
        with self.assertRaises(ValueError):
            transfer_recurrent_state(source, batch([4, 1]), incompatible, sender=True)
        self.assertEqual(incompatible.sent, [])


if __name__ == "__main__":
    unittest.main()
