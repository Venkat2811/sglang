# Kimi-K3: `nvidia_kda` prefill produces NaN after ~1,800 tokens

Reproduction for an SGLang issue. `minimal.pt` is one KDA head (model layer 5, TP rank 7 of 8), 2,048 tokens, with the
inputs exactly as SGLang v0.5.21 passed them to `NvidiaKDAKernel.extend` while serving Kimi-K3 on 8 x B300.

    python3 repro_nvidia_kda_nan.py minimal.pt

Needs torch and the sglang package from `lmsysorg/sglang:v0.5.21-cu130`, and one SM100/SM103 GPU. Expected output is
in `expected_output.txt`: the vendored NVIDIA pipeline gives NaN from token 1,792, while Triton `chunk_kda` and an fp32
token-by-token reference agree to 1.4e-4.
