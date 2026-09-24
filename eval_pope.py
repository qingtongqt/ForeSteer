"""POPE scoring with case-insensitive substring matching, No taking priority."""
import argparse
import json
from pathlib import Path


def parse_answer(text):
    text = str(text).lower()
    if 'no' in text:
        return 'no'
    if 'yes' in text:
        return 'yes'
    return 'other'


def evaluate(gt, predictions):
    if not gt or len(gt) != len(predictions):
        raise ValueError('POPE requires a nonempty, complete prediction set')
    if len({row['question_id'] for row in gt}) != len(gt):
        raise ValueError('Duplicate ground-truth question IDs')
    counts = dict(true_pos=0, true_neg=0, false_pos=0, false_neg=0, unknown=0, yes_answers=0, total_questions=len(gt))
    for truth, prediction in zip(gt, predictions):
        if truth['question_id'] != prediction['question_id']:
            raise ValueError('POPE question ID/order mismatch')
        label = truth['label'].strip().lower()
        if label not in ('yes','no'): raise ValueError('Invalid POPE label')
        # Older files have already-normalized text (including "Other");
        # re-score the preserved model response when available.
        answer = parse_answer(prediction.get('raw_text', prediction.get('text', '')))
        counts['unknown'] += answer == 'other'
        counts['yes_answers'] += answer == 'yes'
        if label == 'yes':
            counts['true_pos' if answer == 'yes' else 'false_neg'] += 1
        else:
            counts['true_neg' if answer == 'no' else 'false_pos'] += 1
    def div(a,b): return a/b if b else 0.
    precision = div(counts['true_pos'], counts['true_pos']+counts['false_pos'])
    recall = div(counts['true_pos'], counts['true_pos']+counts['false_neg'])
    return dict(precision=precision, recall=recall, f1=div(2*precision*recall,precision+recall),
                accuracy=(counts['true_pos']+counts['true_neg'])/len(gt),
                yes_proportion=counts['yes_answers']/len(gt), unknown_proportion=counts['unknown']/len(gt),
                counts=counts, answer_parsing='case-insensitive substring: no first, then yes; other is incorrect')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gt_files', required=True)
    parser.add_argument('--gen_files', required=True)
    parser.add_argument('--save-json')
    args=parser.parse_args()
    def read(path): return [json.loads(x) for x in Path(path).expanduser().read_text().splitlines() if x.strip()]
    result=evaluate(read(args.gt_files),read(args.gen_files))
    print(json.dumps(result,indent=2))
    if args.save_json:
        path=Path(args.save_json).expanduser(); path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__': main()
