import argparse
import json
import os
import random
import re
import sys
import uuid

from tqdm import tqdm
from transformers import set_seed


REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from model import build_model
from foresteer.core import (
    add_inference_args, chat_batch, enable_steering, validate_inference_args,
)


COCO_IMAGE_PATTERN = re.compile(r".*_(\d+)\.[^.]+$")


def get_chunk(items, num_chunks, chunk_idx):
    if num_chunks < 1 or not 0 <= chunk_idx < num_chunks:
        raise ValueError("Invalid chunk index/count")
    start = len(items) * chunk_idx // num_chunks
    end = len(items) * (chunk_idx + 1) // num_chunks
    return items[start:end]


def parse_image_id(image_name):
    match = COCO_IMAGE_PATTERN.match(image_name)
    if match is None:
        raise ValueError(f"Cannot parse COCO image_id from filename: {image_name}")
    return int(match.group(1))


def list_image_files(image_folder):
    image_files = [
        file_name
        for file_name in os.listdir(image_folder)
        if os.path.isfile(os.path.join(image_folder, file_name))
        and file_name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
    ]
    image_files.sort()
    return image_files


def sample_image_files(image_files, max_samples, seed):
    rng = random.Random(seed)
    shuffled = list(image_files)
    rng.shuffle(shuffled)

    if max_samples is None or max_samples <= 0:
        return shuffled
    return shuffled[:max_samples]


def build_prompt(model_name):
    return "Describe this image in detail."


def run_inference(args):
    validate_inference_args(args)
    model = build_model(args)
    if model is None:
        raise ValueError(f"Unsupported model_name: {args.model_name}")
    steering = enable_steering(model, args)

    image_files = list_image_files(args.image_folder)
    if not image_files:
        raise ValueError("No evaluation images found")
    image_files = sample_image_files(image_files, args.max_samples, args.seed)
    image_files = get_chunk(image_files, args.num_chunks, args.chunk_idx)

    answers_file = os.path.expanduser(args.answers_file)
    answers_dir = os.path.dirname(answers_file)
    if answers_dir:
        os.makedirs(answers_dir, exist_ok=True)

    prompt = build_prompt(args.model_name)
    model_id = os.path.basename(os.path.expanduser(args.model_path.rstrip("/")))

    with open(answers_file, "w") as fout:
        for start in tqdm(range(0, len(image_files), args.batch_size)):
            batch = image_files[start:start+args.batch_size]
            captions = chat_batch(model, steering,
                                  [os.path.join(args.image_folder, f) for f in batch],
                                  [prompt]*len(batch))
            for image_file, caption in zip(batch, captions):
                record = {
                    "image_id": parse_image_id(image_file),
                    "image": image_file,
                    "caption": caption,
                    "prompt": prompt,
                    "caption_id": str(uuid.uuid4()),
                    "model_id": model_id,
                    "metadata": steering.to_metadata(),
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                fout.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="LLaVA-7B")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--image-folder", type=str, required=True)
    parser.add_argument("--answers-file", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    add_inference_args(parser)
    args = parser.parse_args()

    set_seed(args.seed)
    run_inference(args)
