"""Generate an Agent Headroom catalog for IA Solar small-model experiments.

No downloads, server launch or GPU access. Unmeasured file sizes are explicitly
estimates. Pass --artifact id=/path/to/model.gguf to replace each with real bytes.
Use --sha256 to pin local model artifacts; this does not validate architecture.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from .catalog import load_catalog, validate_catalog

MiB = 1024 ** 2
GiB = 1024 ** 3

# KV in BF16/FP16, count only layers storing growing full-context KV.
# Hybrid SSM state and CUDA buffers need empirical calibration.
# source_config URLs point to the upstream model's configuration.
PROFILES = {
    "gemma4-e2b": {
        "name": "Google Gemma 4 E2B-it",
        "params_total": 5_100_000_000,
        "context_max": 131_072,
        "attention_layers": 3,
        "kv_heads": 1,
        "head_dim": 512,
        "bounded_kv_bytes": 12 * 1 * 256 * 2 * 2 * 512,
        "weight_estimate_bytes": 2_900_000_000,
        "hf": "https://huggingface.co/google/gemma-4-E2B-it",
        "source_config": "https://huggingface.co/google/gemma-4-E2B-it/blob/main/config.json",
        "caveat": "20 of 35 layers share KV. 12 local attention layers window=512; verify actual engine KV sharing and caching.",
    },
    "qwen3.5-2b": {
        "name": "Qwen 3.5 2B",
        "params_total": 2_000_000_000,
        "context_max": 262_144,
        "attention_layers": 6,
        "kv_heads": 2,
        "head_dim": 256,
        "bounded_kv_bytes": 0,
        "weight_estimate_bytes": 1_280_000_000,
        "hf": "https://huggingface.co/unsloth/Qwen3.5-2B-GGUF",
        "source_config": "https://huggingface.co/Qwen/Qwen3.5-2B/blob/main/config.json",
        "caveat": "18 Gated DeltaNet layers have recurrent state NOT included in the KV estimate.",
    },
    "lfm3-1.7b": {
        "name": "Liquid AI LFM3 1.7B",
        "params_total": 1_700_000_000,
        "context_max": 262_144,
        "attention_layers": 6,
        "kv_heads": 4,
        "head_dim": 128,
        "bounded_kv_bytes": 5 * 4 * 128 * 2 * 2 * 256,
        "weight_estimate_bytes": 1_050_000_000,
        "hf": "https://huggingface.co/LiquidAI/LFM3-1.7B-GGUF",
        "source_config": "https://huggingface.co/LiquidAI/LFM3-1.7B/blob/main/config.json",
        "caveat": "Five 256-token local layers are modeled as bounded KV; convolution state excluded.",
    },
    "minicpm5-2b": {
        "name": "OpenBMB MiniCPM5 2B",
        "params_total": 2_516_000_000,
        "context_max": 131_072,
        "attention_layers": 42,
        "kv_heads": 2,
        "head_dim": 128,
        "bounded_kv_bytes": 0,
        "weight_estimate_bytes": 1_650_000_000,
        "hf": "https://huggingface.co/openbmb/MiniCPM5-2B-GGUF",
        "source_config": "https://huggingface.co/openbmb/MiniCPM5-2B/blob/main/config.json",
        "caveat": "All 42 attention layers have growing KV; use reduced context for high concurrency.",
    },
    "nemotron3-nano-4b": {
        "name": "NVIDIA Nemotron 3 Nano 4B",
        "params_total": 4_000_000_000,
        "context_max": 262_144,
        "attention_layers": 4,
        "kv_heads": 8,
        "head_dim": 128,
        "bounded_kv_bytes": 0,
        "weight_estimate_bytes": 2_840_000_000,
        "hf": "https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF",
        "source_config": "https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16/blob/main/config.json",
        "caveat": "4 attention layers; Mamba-2 recurrent state not included. Check license and GGUF support.",
    },
}


def growing_kv_bytes_per_token(profile: dict) -> int:
    return 2 * profile["attention_layers"] * profile["kv_heads"] * profile["head_dim"] * 2


def parse_artifacts(values: list[str]) -> dict[str, Path]:
    paths = {}
    for assignment in values:
        model_id, sep, raw_path = assignment.partition("=")
        if not sep or model_id not in PROFILES or not raw_path:
            raise ValueError(f"invalid --artifact {assignment!r}; use MODEL_ID=/full/path.gguf")
        path = Path(raw_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(path)
        paths[model_id] = path
    return paths


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * MiB), b""):
            digest.update(block)
    return digest.hexdigest()


def build_catalog(base_catalog: dict, artifacts: dict[str, Path] | None = None, with_hash: bool = False) -> dict:
    artifacts = artifacts or {}
    cat = json.loads(json.dumps(base_catalog))
    cat["catalog_version"] = f"{cat['catalog_version']}-solar-0.1"
    cat["_solar_note"] = (
        "Model sizes are estimates until --artifact is used. KV is a structural "
        "BF16/FP16 growing-cache estimate, not GPU allocation. Hybrid recurrent "
        "states and activations require runtime measurement. Do not quote plans as benchmarks."
    )
    existing = {m["id"] for m in cat["models"]}
    for model_id, p in PROFILES.items():
        if model_id in existing:
            raise ValueError(f"model id already exists in base catalog: {model_id}")
        artifact = artifacts.get(model_id)
        file_bytes = artifact.stat().st_size if artifact else p["weight_estimate_bytes"]
        sha = hash_file(artifact) if artifact and with_hash else None
        entry = {
            "id": model_id,
            "modality": "text",
            "params_total": p["params_total"],
            "params_active": p["params_total"],
            "arch": {
                "n_layers": p["attention_layers"],
                "n_kv_heads": p["kv_heads"],
                "head_dim": p["head_dim"],
                "_note": "Growing attention KV only; not the full hybrid model layer count.",
            },
            "context": {"min": 4096, "max": p["context_max"]},
            "verified": False,
            "files": [{
                "quant": "q4_k_m",
                "bytes": file_bytes,
                "sha256": sha,
                "url": p["hf"],
                "engines": ["llama.cpp"],
                "_note": "Local measured file size" if artifact else "UNVERIFIED estimated file size — provide --artifact",
            }],
            "kv_bytes_per_token": growing_kv_bytes_per_token(p),
            "state_bytes_per_seq": p["bounded_kv_bytes"],
            "_solar": {
                "name": p["name"],
                "source_config": p["source_config"],
                "caveat": p["caveat"],
                "growing_kv_fp16_per_token": growing_kv_bytes_per_token(p),
                "bounded_attention_bytes_estimate": p["bounded_kv_bytes"],
                "file_size_measured": bool(artifact),
            },
        }
        cat["models"].append(entry)
    validate_catalog(cat)
    return cat


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", help="Existing Agent Headroom catalog (default: bundled)")
    parser.add_argument("--out", default="out/catalog-solar.json")
    parser.add_argument("--artifact", action="append", default=[], metavar="MODEL_ID=PATH",
                        help="Actual local Q4_K_M GGUF file; repeat for each downloaded candidate")
    parser.add_argument("--sha256", action="store_true", help="Hash local artifacts to pin model bytes")
    args = parser.parse_args(argv)
    artifacts = parse_artifacts(args.artifact)
    cat = build_catalog(load_catalog(args.base), artifacts, args.sha256)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cat, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out}; {len(artifacts)}/5 model file sizes measured.")
    for mid, p in PROFILES.items():
        kind = "measured" if mid in artifacts else "ESTIMATE"
        print(f"  {mid:<21} growing KV fp16={growing_kv_bytes_per_token(p) // 1024} KiB/token; weights={kind}")
    if len(artifacts) < 5:
        print("WARNING: missing weights are provisional estimates; rerun with --artifact for each model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
