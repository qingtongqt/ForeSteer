import argparse
import json
import os


EVAL_TYPE_DICT = {
    "Perception": [
        "existence",
        "count",
        "position",
        "color",
        "posters",
        "celebrity",
        "scene",
        "landmark",
        "artwork",
        "OCR",
    ],
    "Cognition": [
        "commonsense_reasoning",
        "numerical_calculation",
        "text_translation",
        "code_reasoning",
    ],
}

LABEL_MAP = {
    "yes": 1,
    "no": 0,
    "other": -1,
}


def divide_chunks(items, chunk_size=2):
    for i in range(0, len(items), chunk_size):
        yield items[i : i + chunk_size]


def parse_pred_ans(pred_ans):
    pred_ans = pred_ans.strip().lower()
    if pred_ans in ["yes", "no"]:
        return pred_ans

    prefix_pred_ans = pred_ans[:4]
    if "yes" in prefix_pred_ans:
        return "yes"
    if "no" in prefix_pred_ans:
        return "no"
    return "other"


def safe_divide(numerator, denominator):
    if denominator == 0:
        return 0.0
    return numerator / denominator


def compute_metric(gts, preds):
    assert len(gts) == len(preds)

    mapped_gts = [LABEL_MAP[x] for x in gts]
    mapped_preds = [LABEL_MAP[x] for x in preds]

    correct = sum(1 for gt, pred in zip(mapped_gts, mapped_preds) if gt == pred)
    acc = safe_divide(correct, len(mapped_gts))

    tp = fn = tn = fp = other_num = 0
    for gt, pred in zip(mapped_gts, mapped_preds):
        if pred == -1:
            other_num += 1
            continue
        if gt == 1 and pred == 1:
            tp += 1
        elif gt == 1 and pred == 0:
            fn += 1
        elif gt == 0 and pred == 0:
            tn += 1
        elif gt == 0 and pred == 1:
            fp += 1

    return {
        "TP": tp,
        "FN": fn,
        "TN": tn,
        "FP": fp,
        "precision": safe_divide(tp, tp + fp),
        "recall": safe_divide(tp, tp + fn),
        "other_num": other_num,
        "acc": acc,
    }


def read_task_lines(task_txt):
    with open(task_txt, "r") as fin:
        return fin.readlines()


def process_task(task_txt):
    lines = read_task_lines(task_txt)
    if not lines:
        raise ValueError(f"Empty MME result file: {task_txt}")
    chunk_lines = list(divide_chunks(lines))

    img_num = 0
    acc_plus_correct_num = 0
    gts = []
    preds = []

    for img_items in chunk_lines:
        if len(img_items) != 2:
            raise ValueError(f"Incomplete question pair in {task_txt}")

        if img_items[0].split("\t")[0] != img_items[1].split("\t")[0]:
            raise ValueError(f"Question pair has different images in {task_txt}")
        img_num += 1
        img_correct_num = 0

        for img_item in img_items:
            parts = img_item.rstrip("\n").split("\t", 3)
            if len(parts) != 4:
                raise ValueError(f"Expected 4 tab-separated columns in {task_txt}: {img_item!r}")

            _, _, gt_ans, pred_ans = parts
            gt_ans = gt_ans.strip().lower()
            if gt_ans not in ["yes", "no"]:
                raise ValueError(f"Ground truth answer must be yes/no in {task_txt}, got {gt_ans!r}")

            pred_ans = parse_pred_ans(pred_ans)
            gts.append(gt_ans)
            preds.append(pred_ans)

            if gt_ans == pred_ans:
                img_correct_num += 1

        if img_correct_num == 2:
            acc_plus_correct_num += 1

    metric_dict = compute_metric(gts, preds)
    metric_dict["acc_plus"] = safe_divide(acc_plus_correct_num, img_num)
    return metric_dict


def process_results(results_dir):
    summary = {}
    for eval_type, task_name_list in EVAL_TYPE_DICT.items():
        print("===========", eval_type, "===========")

        scores = 0.0
        task_score_dict = {}
        summary[eval_type] = {"total_score": 0.0, "tasks": {}}

        for task_name in task_name_list:
            task_txt = os.path.join(results_dir, task_name + ".txt")
            if not os.path.exists(task_txt):
                print(f"\t{task_name} skipped: missing {task_txt}")
                continue

            metric_dict = process_task(task_txt)
            task_score = (metric_dict["acc"] + metric_dict["acc_plus"]) * 100
            task_score_dict[task_name] = task_score
            summary[eval_type]["tasks"][task_name] = {
                "score": task_score,
                **metric_dict,
            }
            scores += task_score

        summary[eval_type]["total_score"] = scores
        print("total score:", scores, "\n")
        for task_name, score in task_score_dict.items():
            print("\t", task_name, " score:", score)
        print("\n")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", "--results_dir", type=str, required=True)
    parser.add_argument("--save-json", type=str, default=None)
    args = parser.parse_args()

    results_dir = os.path.expanduser(args.results_dir)
    summary = process_results(results_dir)

    if args.save_json:
        save_json = os.path.expanduser(args.save_json)
        save_dir = os.path.dirname(save_json)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        with open(save_json, "w") as fout:
            json.dump(summary, fout, indent=2)
