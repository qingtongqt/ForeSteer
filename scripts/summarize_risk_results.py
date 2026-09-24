"""Collect completed experiment artifacts without treating partial outputs as scores."""
import argparse
from collections import Counter
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--risk-dir', type=Path, default=Path('outputs/risk/llava'))
    parser.add_argument('--eval-dir', type=Path, default=Path('outputs/eval/llava'))
    args = parser.parse_args()
    manifest_path = args.risk_dir/'manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    splits = dict(Counter(r['split'] for r in manifest.get('records', [])))
    lines = ['# ForeSteer risk-gradient experiment', '',
             f"Risk image splits: {splits}; split seed={manifest.get('seed', 'unknown')}.",
             'Benchmark alpha is prespecified, not tuned on test scores.', '',
             '| Predictor | Horizon | Test MSE | SmoothL1 |', '|---|---|---:|---:|']
    for path in sorted(args.risk_dir.glob('risk_*.json')):
        result = json.loads(path.read_text())
        lines.append(f"| {path.stem} | {result.get('risk_horizon', 'legacy full trajectory')} | {result['test_mse']:.6f} | {result['test_smooth_l1']:.6f} |")
    lines += ['', '## Completed benchmark metrics', '',
              'CHAIR/POPE rates are percentages; CHAIR Len is mean word count; MME uses its native point scale.',
              'Default settings: CHAIR 500 val2014 images / 64 tokens; POPE/MME greedy / 3 tokens.',
              'Consult prediction metadata for per-run overrides.',
              'POPE: case-insensitive substring matching (no first, then yes); unknown responses count as incorrect.', '',
              '| Run | Benchmark | Metrics |', '|---|---|---|']
    count = 0
    for path in sorted(args.eval_dir.glob('*/*metrics.json')):
        result = json.loads(path.read_text())
        run = path.parent.name
        if path.name == 'chair_metrics.json':
            metrics = result['overall_metrics']
            score = ', '.join(f'{key}={100*metrics[key]:.2f}' for key in ('CHAIRs','CHAIRi','Recall','Precision','Len') if key in metrics)
        elif path.name.startswith('pope_'):
            score = ', '.join(f'{key}={100*result[key]:.2f}' for key in ('accuracy','f1','yes_proportion','unknown_proportion'))
        elif path.name == 'mme_metrics.json':
            score = ', '.join(f"{key}={result[key]['total_score']:.2f}" for key in ('Perception','Cognition'))
        else:
            continue
        lines.append(f'| {run} | {path.stem.removesuffix("_metrics")} | {score} |')
        count += 1
    if not count:
        lines.append('| — | — | No completed benchmark metrics yet |')
    lines += ['', 'Metric JSON files retain detailed counts and per-task scores. In-progress predictions are not scored here.',
              'One training seed is evaluated; these results do not measure between-seed uncertainty.', '']
    args.risk_dir.mkdir(parents=True, exist_ok=True)
    (args.risk_dir/'report.md').write_text('\n'.join(lines))

if __name__ == '__main__':
    main()
