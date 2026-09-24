"""Recover true EOS termination without loading a language model."""
import hashlib
import json
from pathlib import Path


def model_eos_ids(model_path):
    root = Path(model_path).expanduser()
    for filename in ('generation_config.json', 'config.json'):
        path = root / filename
        if path.exists():
            value = json.loads(path.read_text()).get('eos_token_id')
            if value is not None:
                return {value} if isinstance(value, int) else set(value)
    raise ValueError(f'No EOS IDs in {root}; re-extract with termination metadata')


def is_terminated(record, eos_ids):
    # Explicit generation metadata takes precedence, including length stops.
    if 'terminated' in record:
        if not isinstance(record['terminated'], bool):
            raise ValueError('terminated must be a boolean')
        return record['terminated']
    ids = record['tokenization']['token_ids']
    if not ids:
        raise ValueError('Empty token trajectory')
    return ids[-1] in eos_ids


def cached_termination_flags(manifest):
    """Older feature caches lack EOS metadata: recover it from verified tokens."""
    rows = manifest['records']
    if all(isinstance(row.get('terminated'), bool) for row in rows):
        return {row['image_id']: row['terminated'] for row in rows}
    source = Path(manifest['input']).expanduser().read_bytes()
    if hashlib.sha256(source).hexdigest() != manifest['input_sha256']:
        raise ValueError('Source trajectories changed; cannot recover cache termination')
    records = {r['image_id']: r for r in (json.loads(line) for line in source.splitlines() if line.strip())}
    eos_ids = model_eos_ids(manifest['model_path'])
    return {row['image_id']: is_terminated(records[row['image_id']], eos_ids) for row in rows}
