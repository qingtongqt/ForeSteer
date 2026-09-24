#!/usr/bin/env python3
"""Map CHAIR hallucinated object onsets to generated model tokens."""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chair_token_annotator import TokenCHAIR


SCHEMA_VERSION = "foresteer.coco_chair_tokens.v1"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Generation JSONL.")
    parser.add_argument("--output", required=True, help="Annotated JSONL.")
    parser.add_argument("--summary", required=True, help="Dataset metric summary JSON.")
    parser.add_argument("--coco-root", required=True)
    parser.add_argument("--cache", required=True)
    return parser.parse_args()


def expanded_path(value):
    return Path(value).expanduser().resolve()


def load_or_build_evaluator(cache_path, annotation_path):
    expected_path = str(annotation_path)
    evaluator = None
    if cache_path.is_file():
        with cache_path.open("rb") as handle:
            candidate = pickle.load(handle)
        cached_path = str(Path(candidate.coco_path).expanduser().resolve())
        if isinstance(candidate, TokenCHAIR) and cached_path == expected_path:
            evaluator = candidate
            print(f"Loaded CHAIR cache: {cache_path}")
        else:
            print("Ignoring incompatible CHAIR cache; rebuilding it once.")

    if evaluator is None:
        print("Building CHAIR ground-truth index (first run only)...")
        evaluator = TokenCHAIR(expected_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_cache = cache_path.with_suffix(cache_path.suffix + ".tmp")
        with temporary_cache.open("wb") as handle:
            pickle.dump(evaluator, handle)
        os.replace(temporary_cache, cache_path)
        print(f"Saved CHAIR cache: {cache_path}")
    return evaluator


def validate_tokenization(record):
    tokenization = record["tokenization"]
    lengths = {
        len(tokenization[key])
        for key in (
            "token_ids", "tokens", "token_texts", "char_offsets",
            "special_tokens_mask",
        )
    }
    if len(lengths) != 1:
        raise ValueError(f"Token fields have different lengths for image {record['image_id']}")
    caption = record["caption"]
    for span in tokenization["char_offsets"]:
        if span is not None and not (0 <= span[0] <= span[1] <= len(caption)):
            raise ValueError(f"Invalid token offset {span} for image {record['image_id']}")
    return tokenization


def first_overlapping_token(object_span, token_offsets, special_mask):
    object_start, object_end = object_span
    for token_index, (token_span, is_special) in enumerate(zip(token_offsets, special_mask)):
        if token_span is None or is_special:
            continue
        token_start, token_end = token_span
        if token_end > object_start and token_start < object_end:
            return token_index
    return None


def annotate_record(record, evaluator):
    tokenization = validate_tokenization(record)
    chair = evaluator.annotate_caption(record["image_id"], record["caption"])
    labels = [0] * len(tokenization["token_ids"])
    mapped_mentions = []
    for mention in chair["hallucinated_object_mentions"]:
        mention = dict(mention)
        token_index = first_overlapping_token(
            mention["char_span"],
            tokenization["char_offsets"],
            tokenization["special_tokens_mask"],
        )
        mention["token_index"] = token_index
        if token_index is None:
            mention["token_id"] = None
            mention["token"] = None
        else:
            labels[token_index] = 1
            mention["token_id"] = tokenization["token_ids"][token_index]
            mention["token"] = tokenization["tokens"][token_index]
        mapped_mentions.append(mention)

    chair["hallucinated_object_mentions"] = mapped_mentions
    annotated = dict(record)
    annotated["schema_version"] = SCHEMA_VERSION
    annotated["hallucination_start_labels"] = labels
    annotated["chair"] = chair
    return annotated


def main():
    args = parse_args()
    input_path = expanded_path(args.input)
    output_path = expanded_path(args.output)
    summary_path = expanded_path(args.summary)
    coco_root = expanded_path(args.coco_root)
    cache_path = expanded_path(args.cache)
    annotation_path = coco_root / "annotations"

    if not input_path.is_file():
        raise FileNotFoundError(f"Generation JSONL not found: {input_path}")
    if not annotation_path.is_dir():
        raise FileNotFoundError(f"COCO annotations not found: {annotation_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    evaluator = load_or_build_evaluator(cache_path, annotation_path)

    counts = {
        "captions": 0,
        "hallucinated_captions": 0,
        "generated_object_mentions": 0,
        "hallucinated_object_mentions": 0,
        "unmapped_hallucinated_mentions": 0,
    }
    temporary_output = output_path.with_suffix(output_path.suffix + ".tmp")

    from tqdm import tqdm

    with input_path.open("r", encoding="utf-8") as input_file, temporary_output.open(
        "w", encoding="utf-8"
    ) as output_file:
        for line_number, line in enumerate(tqdm(input_file, desc="Annotating", unit="caption"), start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                annotated = annotate_record(record, evaluator)
            except Exception as error:
                raise RuntimeError(f"Failed at {input_path}:{line_number}") from error

            chair = annotated["chair"]
            hallucinated_mentions = chair["hallucinated_object_mentions"]
            counts["captions"] += 1
            counts["hallucinated_captions"] += chair["metrics"]["CHAIRs"]
            counts["generated_object_mentions"] += len(chair["object_mentions"])
            counts["hallucinated_object_mentions"] += len(hallucinated_mentions)
            counts["unmapped_hallucinated_mentions"] += sum(
                mention["token_index"] is None for mention in hallucinated_mentions
            )
            output_file.write(json.dumps(annotated, ensure_ascii=False) + "\n")

    captions = counts["captions"]
    object_mentions = counts["generated_object_mentions"]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "input": str(input_path),
        "output": str(output_path),
        "annotation_year": 2014,
        "counts": counts,
        "metrics": {
            "CHAIRs": counts["hallucinated_captions"] / captions if captions else 0.0,
            "CHAIRi": (
                counts["hallucinated_object_mentions"] / object_mentions
                if object_mentions else 0.0
            ),
        },
    }
    temporary_summary = summary_path.with_suffix(summary_path.suffix + ".tmp")
    with temporary_summary.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    os.replace(temporary_output, output_path)
    os.replace(temporary_summary, summary_path)
    print(json.dumps(summary["metrics"], indent=2))
    print(f"Saved token annotations to {output_path}")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
