#!/usr/bin/env python3
"""Generate token-preserving captions for a COCO split."""

import argparse
import json
import random
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


SCHEMA_VERSION = "foresteer.coco_generation.v1"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="llava")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--coco-root", required=True)
    parser.add_argument("--split", default="train2014")
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt", default="Describe this image in detail.")
    parser.add_argument("--max-new-tokens", dest="max_length", type=int, default=128)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--max-samples", type=int, default=0,
        help="0 processes every remaining image.",
    )
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    return parser.parse_args()


def expanded_path(value):
    return Path(value).expanduser().resolve()


def load_images(coco_root, split):
    image_dir = coco_root / split
    if not image_dir.is_dir():
        raise FileNotFoundError(f"COCO image directory not found: {image_dir}")

    filename_pattern = re.compile(
        rf"^COCO_{re.escape(split)}_(\d{{12}})\.jpg$",
        flags=re.IGNORECASE,
    )
    images_by_id = {}
    ignored_files = 0
    for image_path in image_dir.iterdir():
        if not image_path.is_file():
            continue
        match = filename_pattern.fullmatch(image_path.name)
        if match is None:
            ignored_files += 1
            continue
        image_id = int(match.group(1))
        if image_id in images_by_id:
            raise ValueError(
                f"Multiple image files resolve to COCO image_id={image_id}: "
                f"{images_by_id[image_id]['file_name']} and {image_path.name}"
            )
        images_by_id[image_id] = {
            "id": image_id,
            "file_name": image_path.name,
        }

    images = [images_by_id[image_id] for image_id in sorted(images_by_id)]
    if not images:
        raise FileNotFoundError(
            f"No COCO images matching {filename_pattern.pattern!r} found in {image_dir}"
        )
    if ignored_files:
        print(f"Ignored {ignored_files} non-COCO files in {image_dir}")
    print(f"Discovered {len(images)} actual image files in {image_dir}")
    return images, 2014


def main():
    args = parse_args()
    if args.temperature != 0:
        raise ValueError("This dataset requires greedy generation; set --temperature 0.")
    if args.num_beams != 1:
        raise ValueError("This dataset requires greedy generation; set --num-beams 1.")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Require num_shards >= 1 and 0 <= shard_index < num_shards.")

    model_path = expanded_path(args.model_path)
    coco_root = expanded_path(args.coco_root)
    output_path = expanded_path(args.output)
    if not model_path.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Importing torch/model code after cheap argument and path validation gives
    # fast, actionable failures on login nodes without a CUDA runtime.
    import torch
    from model import build_model

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Load the cluster CUDA/GPU environment, "
            "then run this script again."
        )
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    images, annotation_year = load_images(coco_root, args.split)
    images = images[args.start_index:]
    if args.max_samples > 0:
        images = images[:args.max_samples]
    # Apply the global subset first, then distribute it across workers. Thus
    # MAX_SAMPLES always means the total dataset size, not samples per GPU.
    images = images[args.shard_index::args.num_shards]

    print(
        f"split={args.split} shard={args.shard_index}/{args.num_shards} "
        f"selected={len(images)}"
    )
    if not images:
        output_path.write_text("", encoding="utf-8")
        print(f"No images assigned to this shard: {output_path}")
        return

    args.model_path = str(model_path)
    model = build_model(args)
    tokenizer_name = getattr(model.tokenizer, "name_or_path", str(model_path))

    from tqdm import tqdm

    # Every run starts a fresh shard; records from previous jobs are not reused.
    with output_path.open("w", encoding="utf-8", buffering=1) as output_file:
        for image_info in tqdm(
            images,
            desc=f"GPU shard {args.shard_index + 1}/{args.num_shards}",
            unit="image",
            position=args.shard_index if args.num_shards > 1 else 0,
        ):
            image_id = int(image_info["id"])
            image_path = coco_root / args.split / image_info["file_name"]
            if not image_path.is_file():
                raise FileNotFoundError(f"COCO image not found: {image_path}")
            generation = model.generate(str(image_path), args.prompt)
            record = {
                "schema_version": SCHEMA_VERSION,
                "image_id": image_id,
                "image_file": str(image_path.relative_to(coco_root)),
                "split": args.split,
                "annotation_year": annotation_year,
                "prompt": args.prompt,
                "caption": generation.text,
                "model": {
                    "adapter": args.model_name,
                    "path": str(model_path),
                },
                "decoding": {
                    "strategy": "greedy",
                    "max_new_tokens": args.max_length,
                    "num_beams": args.num_beams,
                    "temperature": args.temperature,
                },
                "tokenization": {
                    "tokenizer": tokenizer_name,
                    "token_ids": generation.token_ids,
                    "tokens": generation.tokens,
                    "token_texts": generation.token_texts,
                    "char_offsets": generation.token_char_offsets,
                    "special_tokens_mask": generation.special_tokens_mask,
                },
            }
            if generation.terminated is not None:
                record['terminated'] = generation.terminated
            if generation.synthetic_prefix_tokens is not None:
                record['tokenization']['synthetic_prefix_tokens'] = generation.synthetic_prefix_tokens
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved generations to {output_path}")


if __name__ == "__main__":
    main()
