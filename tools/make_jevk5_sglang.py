"""
Repackage a JevK5 decision model so SGLang can serve it.

JevK5 (alibiserikbay/JevK5-9B, alibiserikbay/JevK5) is Qwen3.5 with a LoRA
merged into the language model, saved text-only (`qwen3_5_text`,
Qwen3_5ForCausalLM). SGLang (0.5.9 up to at least 0.5.21) only serves Qwen3.5
in its multimodal form, Qwen3_5ForConditionalGeneration -- the form
Qwen/Qwen3.5-9B and Qwen3.6-27B ship in. So this builds that form:

  config, tokenizer, processor files   from the base model (Qwen/Qwen3.5-9B)
  language model + lm_head             JevK5's weights, untouched (symlinked)
  vision tower                         the base model's (never used: prompts are text)

JevK5's tensors already carry the multimodal names (model.language_model.*),
so nothing is renamed; the script checks that its tensors are exactly the base
model's language-model tensors, with the same shapes, and stops otherwise.
Only the base shards that hold the vision tower are downloaded.

    uv run python tools/make_jevk5_sglang.py                                   # JevK5-9B
    uv run python tools/make_jevk5_sglang.py --jevk5 alibiserikbay/JevK5 --base Qwen/Qwen3.5-4B \\
        --out models/JevK5-sglang
    bash tools/run_sglang_server.sh --model-path models/JevK5-9B-sglang \\
        --served-model-name alibiserikbay/JevK5-9B --port 30001
"""

from __future__ import annotations

import argparse
import json
import shutil
import struct
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

LANGUAGE_FILE = "model-language.safetensors"
VISION_FILE = "model-vision.safetensors"


def read_header(path: Path) -> tuple[dict, int]:
    """(tensor name -> {dtype, shape, data_offsets}, byte where the data starts)."""
    with path.open("rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(n))
    header.pop("__metadata__", None)
    return header, 8 + n


def is_language(name: str) -> bool:
    return name.startswith("model.language_model.") or name == "lm_head.weight"


def extract(sources: list[Path], names: set[str], out: Path) -> dict[str, dict]:
    """Copy the tensors `names` from safetensors `sources` into `out`, byte for
    byte (no torch needed). Returns their headers."""
    plan, offset = [], 0
    header = {"__metadata__": {"format": "pt"}}
    for src in sources:
        src_header, start = read_header(src)
        for name in sorted(n for n in src_header if n in names):
            info = src_header[name]
            begin, end = info["data_offsets"]
            header[name] = {"dtype": info["dtype"], "shape": info["shape"], "data_offsets": [offset, offset + end - begin]}
            plan.append((src, start + begin, end - begin))
            offset += end - begin
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)
    with out.open("wb") as fh:
        fh.write(struct.pack("<Q", len(raw)) + raw)
        for src, at, size in plan:
            with src.open("rb") as s:
                s.seek(at)
                while size:
                    chunk = s.read(min(size, 1 << 26))
                    fh.write(chunk)
                    size -= len(chunk)
    header.pop("__metadata__")
    return header


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Repackage JevK5 as Qwen3_5ForConditionalGeneration for SGLang")
    p.add_argument("--jevk5", default="alibiserikbay/JevK5-9B")
    p.add_argument("--base", default="Qwen/Qwen3.5-9B", help="the Qwen3.5 model JevK5 was trained from")
    p.add_argument("--out", default="models/JevK5-9B-sglang")
    args = p.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"downloading {args.jevk5} ...")
    language = Path(hf_hub_download(args.jevk5, "model.safetensors")).resolve()
    jev_header, _ = read_header(language)
    print(f"downloading the small files of {args.base} ...")
    base_dir = Path(snapshot_download(args.base, allow_patterns=["*.json", "*.jinja", "*.txt"]))
    weight_map = json.loads((base_dir / "model.safetensors.index.json").read_text())["weight_map"]

    # JevK5's tensors must be exactly the base model's language model.
    base_language = {n for n in weight_map if is_language(n)}
    if set(jev_header) != base_language:
        extra, missing = sorted(set(jev_header) - base_language), sorted(base_language - set(jev_header))
        raise SystemExit(f"{args.jevk5} does not match {args.base}: extra {extra[:5]}, missing {missing[:5]}")

    vision = {n for n in weight_map if n.startswith("model.visual.")}
    shards = sorted({weight_map[n] for n in vision})
    print(f"downloading the vision tower: {', '.join(shards)} ...")
    shard_paths = [Path(hf_hub_download(args.base, s)) for s in shards]
    # Shapes of the language tensors those shards hold must match JevK5's.
    # dtypes may differ: JevK5 saves linear_attn.A_log and linear_attn.norm in
    # BF16 where the base has F32; SGLang casts on load, as JevK5's runtime does.
    for shard in shard_paths:
        header, _ = read_header(shard)
        for name, info in header.items():
            if is_language(name) and info["shape"] != jev_header[name]["shape"]:
                raise SystemExit(f"shape mismatch for {name}: {info['shape']} vs {jev_header[name]['shape']}")
    vision_header = extract(shard_paths, vision, out / VISION_FILE)

    for f in base_dir.iterdir():
        if f.name != "model.safetensors.index.json":
            shutil.copy(f, out / f.name)
    link = out / LANGUAGE_FILE
    link.unlink(missing_ok=True)
    link.symlink_to(language)
    index = {n: LANGUAGE_FILE for n in jev_header} | {n: VISION_FILE for n in vision_header}
    (out / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": index}, indent=2))
    # The decision readout's calibration temperature, for reference.
    shutil.copy(hf_hub_download(args.jevk5, "jevk5_config.json"), out / "jevk5_config.json")
    print(f"\nDone: {out}  ({len(jev_header)} language tensors from {args.jevk5}, {len(vision_header)} vision "
          f"tensors from {args.base})\nServe it with:\n  bash tools/run_sglang_server.sh --model-path {out} "
          f"--served-model-name {args.jevk5} --port 30001")


if __name__ == "__main__":
    main()
