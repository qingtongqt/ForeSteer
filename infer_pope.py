import argparse
import json
import os
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


def get_chunk(items, num_chunks, chunk_idx):
    if num_chunks < 1 or not 0 <= chunk_idx < num_chunks:
        raise ValueError("Invalid chunk index/count")
    start = len(items) * chunk_idx // num_chunks
    end = len(items) * (chunk_idx + 1) // num_chunks
    return items[start:end]


def build_prompt(question):
    question = question.strip()
    return f"{question} Answer with yes or no."


def normalize_yes_no(response):
    from eval_pope import parse_answer
    return parse_answer(response).capitalize()


def run_inference(args):
    validate_inference_args(args)
    model = build_model(args)
    if model is None:
        raise ValueError(f"Unsupported model_name: {args.model_name}")
    steering = enable_steering(model, args)

    with open(os.path.expanduser(args.question_file), "r") as f:
        questions = [json.loads(line) for line in f]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    answers_file = os.path.expanduser(args.answers_file)
    answers_dir = os.path.dirname(answers_file)
    if answers_dir:
        os.makedirs(answers_dir, exist_ok=True)

    with open(answers_file, "w") as fout:
        for start in tqdm(range(0, len(questions), args.batch_size)):
            batch = questions[start:start+args.batch_size]
            outputs = chat_batch(model, steering,
                                 [os.path.join(args.image_folder, sample["image"]) for sample in batch],
                                 [build_prompt(sample["text"]) for sample in batch])
            for sample, output in zip(batch, outputs):
                raw_output = output
                output = normalize_yes_no(output)

                record = {
                    "question_id": sample["question_id"],
                    "image": sample["image"],
                    "prompt": sample["text"],
                    "text": output,
                    "raw_text": raw_output,
                    "answer_id": str(uuid.uuid4()),
                    "model_id": os.path.basename(os.path.expanduser(args.model_path.rstrip("/"))),
                    "metadata": steering.to_metadata(),
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                fout.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="LLaVA-7B")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--image-folder", type=str, required=True)
    parser.add_argument(
        "--question-file",
        type=str,
        default="data/pope/coco/coco_pope_random.json",
    )
    parser.add_argument("--answers-file", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=3)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    add_inference_args(parser)
    args = parser.parse_args()

    set_seed(args.seed)
    run_inference(args)
