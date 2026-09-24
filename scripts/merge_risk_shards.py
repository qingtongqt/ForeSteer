"""Validate complete extraction shards and publish a training manifest."""
import argparse
import json
from pathlib import Path


def merge(output, num_shards):
    output = Path(output)
    if num_shards < 2:
        raise ValueError('Shard merge requires at least two shards')
    if (output/'manifest.json').exists():
        raise FileExistsError('Completed extraction already exists')
    shards = [json.loads((output/f'manifest.shard-{i}-of-{num_shards}.json').read_text())
              for i in range(num_shards)]
    keys = ('input', 'input_sha256', 'model_path', 'model_name', 'layers', 'seed', 'total_records', 'num_shards')
    rows = []
    for i, shard in enumerate(shards):
        if shard['shard_index'] != i or shard['num_shards'] != num_shards:
            raise ValueError('Invalid shard identity')
        if any(shard[k] != shards[0][k] for k in keys):
            raise ValueError('Incompatible extraction shards')
        for row in shard['records']:
            if row['index'] % num_shards != i:
                raise ValueError('Record assigned to wrong shard')
            if row['file'] != f"states/{row['index']:06d}.pt" or not (output/row['file']).is_file():
                raise ValueError('Missing or invalid feature file')
        rows.extend(shard['records'])
    rows.sort(key=lambda row: row['index'])
    if [r['index'] for r in rows] != list(range(shards[0]['total_records'])):
        raise ValueError('Incomplete or duplicate extraction records')
    if len({r['image_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate image IDs')
    result = {k: shards[0][k] for k in keys}
    result['records'] = rows
    tmp = output/'manifest.json.tmp'
    tmp.write_text(json.dumps(result, indent=2)+'\n')
    tmp.replace(output/'manifest.json')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--num-shards', required=True, type=int)
    args = parser.parse_args()
    result = merge(args.output, args.num_shards)
    print(f"Merged {len(result['records'])} feature records")
