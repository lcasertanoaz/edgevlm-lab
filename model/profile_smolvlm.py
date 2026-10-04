#!/usr/bin/env python3
"""Profile SmolVLM-256M-Instruct with PyTorch forward hooks.

Records, for every nn.Linear layer during one prefill pass: input/output
features, tokens processed, multiply-accumulates, and parameter count. Writes
a per-layer CSV and a summary JSON in the format perf/perf_model.py reads.

Note: attention-score matmuls (QK^T, AV) are not nn.Linear layers and are not
counted yet; add hooks on the attention modules to include them.

Setup:  pip install torch transformers pillow
Run:    python model/profile_smolvlm.py --image path/to/img.jpg --question "What is in the image?"
STATUS: starter code, not yet run.
"""
import argparse, csv, json
import torch
from PIL import Image
from transformers import AutoProcessor
try:
    from transformers import AutoModelForImageTextToText as AutoVLM
except ImportError:                                   # older transformers
    from transformers import AutoModelForVision2Seq as AutoVLM

MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"


def component(name):
    if "vision" in name:
        return "vision"
    if "connector" in name:
        return "connector"
    return "lm"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--question", default="What is in the image?")
    ap.add_argument("--out", default="profile")
    args = ap.parse_args()

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoVLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32).eval()

    rows = []

    def make_hook(name):
        def hook(mod, inputs, output):
            x = inputs[0]
            tokens = x.numel() // x.shape[-1]
            rows.append({"layer": name, "component": component(name),
                         "in_features": mod.in_features, "out_features": mod.out_features,
                         "tokens": tokens, "macs": tokens * mod.in_features * mod.out_features,
                         "params": mod.weight.numel()})
        return hook

    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear):
            mod.register_forward_hook(make_hook(name))

    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": args.question}]}]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[Image.open(args.image).convert("RGB")], return_tensors="pt")
    with torch.no_grad():
        model(**inputs)                                # one prefill pass

    with open(args.out + "_layers.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    def total(comp, key):
        return sum(r[key] for r in rows if r["component"] == comp)

    lm_rows = [r for r in rows if r["component"] == "lm"]
    summary = {
        "note": f"profiled {MODEL_ID}, Linear layers only",
        "vision": {"params": total("vision", "params"), "macs": total("vision", "macs") + total("connector", "macs")},
        "lm": {"params": total("lm", "params")},
        "prefill_tokens": max(r["tokens"] for r in lm_rows) if lm_rows else 0,
    }
    json.dump(summary, open(args.out + "_summary.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
