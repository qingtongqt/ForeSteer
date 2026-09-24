import argparse
import json
import os
import sys
from tqdm import tqdm
from transformers import set_seed


REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from model import build_model
from foresteer.core import (
    add_inference_args, chat_batch, enable_steering, validate_inference_args,
)


def get_chunk(items, num_chunks, chunk_idx):
    if num_chunks < 1 or not 0 <= chunk_idx < num_chunks:
        raise ValueError("Invalid chunk index/count")
    start = len(items) * chunk_idx // num_chunks
    end = len(items) * (chunk_idx + 1) // num_chunks
    return items[start:end]


def build_prompt(question):
    question = question.strip()
    return f"{question} Answer with yes or no."


def resolve_image_path(task_dir, image_name):
    image_path = os.path.join(task_dir, "images", image_name)
    if os.path.exists(image_path):
        return image_path

    image_path = os.path.join(task_dir, image_name)
    if os.path.exists(image_path):
        return image_path

    return None


def load_reference_lines(reference_file):
    with open(os.path.expanduser(reference_file), "r") as fin:
        return fin.readlines()


def list_reference_files(reference_dir):
    reference_files = [
        file_name
        for file_name in os.listdir(reference_dir)
        if os.path.isfile(os.path.join(reference_dir, file_name)) and file_name.endswith(".txt")
    ]
    reference_files.sort()
    return reference_files


def sanitize_response(response):
    if "ASSISTANT:" in response:
        response = response.rsplit("ASSISTANT:", 1)[-1]
    for stop_token in ("</s>", "###"):
        if stop_token in response:
            response = response.split(stop_token, 1)[0]
    return response.replace("\n", " ").replace("\t", " ").strip()


def run_inference(args):
    validate_inference_args(args)
    model = build_model(args)
    if model is None:
        raise ValueError(f"Unsupported model_name: {args.model_name}")
    steering = enable_steering(model, args)

    reference_dir = os.path.expanduser(args.reference_dir)
    base_dir = os.path.expanduser(args.base_dir)
    answers_dir = os.path.expanduser(args.answers_dir)
    os.makedirs(answers_dir, exist_ok=True)

    with open(os.path.join(answers_dir, "metadata.json"), "w") as fout:
        json.dump(steering.to_metadata(), fout, indent=2)
    reference_files = list_reference_files(reference_dir)
    if not reference_files:
        raise ValueError("No MME reference files")
    reference_files = get_chunk(reference_files, args.num_chunks, args.chunk_idx)

    for file_name in reference_files:
        task_name = os.path.splitext(file_name)[0]
        task_dir = os.path.join(base_dir, task_name)
        if not os.path.exists(task_dir):
            raise FileNotFoundError(task_dir)

        input_file_path = os.path.join(reference_dir, file_name)
        output_file_path = os.path.join(answers_dir, f"{task_name}.txt")
        lines = load_reference_lines(input_file_path)

        samples = []
        for line_idx, line in enumerate(lines):
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                raise ValueError(f"Malformed reference: {file_name}:{line_idx}")
            image_name, question, ground_truth_answer = parts
            image_path = resolve_image_path(task_dir, image_name)
            if image_path is None:
                raise FileNotFoundError(f"{task_dir}/{image_name}")
            samples.append((image_name, question, ground_truth_answer, image_path))
        with open(output_file_path, "w") as fout:
            for start in tqdm(range(0, len(samples), args.batch_size), desc=task_name):
                batch = samples[start:start+args.batch_size]
                responses = chat_batch(model, steering, [s[3] for s in batch],
                                       [build_prompt(s[1]) for s in batch])
                for (image_name, question, ground_truth_answer, _), response in zip(batch, responses):
                    response = sanitize_response(response)
                    fout.write(f"{image_name}\t{question}\t{ground_truth_answer}\t{response}\n")
                    fout.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-dir", "--reference_dir", type=str, default="data/mme/eval_tool/Your_Results")
    parser.add_argument("--base-dir", "--base_dir", type=str, default="data/mme/MME_Benchmark")
    parser.add_argument("--answers-dir", "--answers_dir", type=str, required=True)
    parser.add_argument("--model-name", "--model_name", type=str, default="LLaVA-7B")
    parser.add_argument("--model-path", "--model_path", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_length", "--max-length", type=int, default=3)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    add_inference_args(parser)
    args = parser.parse_args()

    set_seed(args.seed)
    run_inference(args)
