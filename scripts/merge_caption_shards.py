#!/usr/bin/env python3
"""Merge sorted caption JSONL shards atomically and deduplicate image ids."""

import argparse
import heapq
import json
import os
from pathlib import Path


EXPECTED_SCHEMA = "foresteer.coco_generation.v1"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def records(path):
    previous_image_id = None
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
            if record.get("schema_version") != EXPECTED_SCHEMA:
                raise ValueError(
                    f"Unexpected schema at {path}:{line_number}: "
                    f"{record.get('schema_version')!r}"
                )
            image_id = int(record["image_id"])
            if previous_image_id is not None and image_id <= previous_image_id:
                raise ValueError(
                    f"Records in {path} are not strictly ordered by image_id "
                    f"at line {line_number}."
                )
            previous_image_id = image_id
            yield image_id, record, path


def main():
    args = parse_args()
    output_path = Path(args.output).expanduser().resolve()
    input_paths = []
    seen_paths = set()
    for value in args.input:
        path = Path(value).expanduser().resolve()
        if path.is_file() and path not in seen_paths:
            input_paths.append(path)
            seen_paths.add(path)
    if not input_paths:
        raise FileNotFoundError("None of the caption shard inputs exist.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output_path.with_suffix(output_path.suffix + ".tmp")
    iterators = [records(path) for path in input_paths]
    merged = heapq.merge(*iterators, key=lambda item: item[0])

    count = 0
    previous_id = None
    previous_record = None
    with temporary_output.open("w", encoding="utf-8") as output_file:
        for image_id, record, source_path in merged:
            if image_id == previous_id:
                if record != previous_record:
                    raise ValueError(
                        f"Conflicting records for image_id={image_id}; "
                        f"one source is {source_path}."
                    )
                continue
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            previous_id = image_id
            previous_record = record
            count += 1

    os.replace(temporary_output, output_path)
    print(f"Merged {len(input_paths)} files / {count} captions into {output_path}")


if __name__ == "__main__":
    main()
