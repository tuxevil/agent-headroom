# IA Solar: pruebas de 5 SLM con Agent Headroom

Experimental **fork-specific** extension for [tuxevil/agent-headroom](https://github.com/tuxevil/agent-headroom).
Do not present estimated results as measurements of the RTX 4000.

## Models

| Model ID | BF16 growing attention KV/token | Extra memory not in growing KV |
|---|---:|---|
| gemma4-e2b | 6 KiB | bounded sliding-window KV (~6 MiB/seq), runtime buffers |
| qwen3.5-2b | 12 KiB | Gated DeltaNet state |
| lfm3-1.7b | 12 KiB | local-window KV (~2.5 MiB/seq), convolution state |
| minicpm5-2b | 42 KiB | runtime buffers |
| nemotron3-nano-4b | 16 KiB | Mamba-2 recurrent state |

Figures are **structural FP16/BF16 estimates** from Hugging Face configurations. The simulator's \`state_bytes_per_seq\` records only bounded local-attention KV where estimated; **hybrid recurrent state is missing**, so actual memory can be higher. Engine implementations can differ in cache sharing, memory retention and per-slot allocation. Compare model **quality** and **tool calls** separately through \`tuxevil-benchmark\`. The GGUF files must be selected by exact revision/quantization before comparative runs.

## Use on the RTX 4000 8 GiB

Install upstream Agent Headroom prerequisites (\`python >=3.10\`, \`llama-server\`, \`agentperf-local\`); do not run while the GPU serves homelab workloads.

The bundled catalog is left untouched. Generate a custom local catalog:

~~~bash
python3 -m bench.solar_catalog --out out/catalog-solar.json \
  --artifact gemma4-e2b=/models/gemma4-e2b-q4_k_m.gguf \
  --artifact qwen3.5-2b=/models/qwen35-2b-q4_k_m.gguf \
  --artifact lfm3-1.7b=/models/lfm3-1.7b-q4_k_m.gguf \
  --artifact minicpm5-2b=/models/minicpm5-2b-q4_k_m.gguf \
  --artifact nemotron3-nano-4b=/models/nemotron3-4b-q4_k_m.gguf \
  --sha256
~~~

The filenames here are placeholders; replace them with local GGUF paths. You may supply fewer models initially; missing weights use **clearly flagged unverified estimates**. Supplying a file measures size and computes a local hash when \`--sha256\` is used, but **does not prove the model architecture or inference engine compatibility**.

~~~bash
python3 -m bench probe
python3 -m bench plan host_pressure \
  --catalog out/catalog-solar.json --model gemma4-e2b \
  --quant q4_k_m --context 8192 --n-seq 8 --kv-quant q8 \
  --out out/plan-solar-gemma.json --explain
python3 -m bench run out/plan-solar-gemma.json \
  --catalog out/catalog-solar.json \
  --model-file /models/gemma4-e2b-q4_k_m.gguf \
  --levels 1,2,4,8 --replay aa-mini-v1
~~~

Repeat with **8K, 16K, 32K context per agent**, when planner/runtime fit permits. For \`llama.cpp\`, \`--ctx-size\` in CROWD is aggregate across slots (so each receives \`context\`). Record the exact quantization and build. A planning failure is an estimate; do not override until verifying safe memory headroom.

**Meaningful service capacity** is not simply the number of failure-free agents: establish p95 TTFT, tokens/s per active agent, max queue delay, tool-call correctness, and tail latency limits. Record CPU/GPU energy, 100 W vs 125 W power cap, available solar input, battery SOC and Wh per 1M output tokens separately; this fork does not measure those automatically.

Before a public offering, also test end-to-end via LiteLLM with **real virtual-key budgets and rate limits**; Agent Headroom connects directly to its spawned backend and does not verify the gateway controls.

## Sources

- Google Gemma4: https://huggingface.co/google/gemma-4-E2B-it/blob/main/config.json
- Qwen3.5: https://huggingface.co/Qwen/Qwen3.5-2B/blob/main/config.json
- LiquidAI LFM3: https://huggingface.co/LiquidAI/LFM3-1.7B/blob/main/config.json
- OpenBMB MiniCPM5: https://huggingface.co/openbmb/MiniCPM5-2B/blob/main/config.json
- NVIDIA Nemotron: https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16/blob/main/config.json

Unit tests: \`python3 -m unittest tests.test_solar_catalog -v\`.
