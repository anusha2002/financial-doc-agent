"""
Reads eval_results.csv (after you've filled in the 'correct' column with y/n)
and prints overall accuracy plus a breakdown by question category.

Usage:
    python score_eval.py
"""
import csv
from collections import defaultdict

RESULTS_FILE = "eval_results.csv"


def main():
    with open(RESULTS_FILE, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    ungraded = [r for r in rows if r["correct"].strip().lower() not in ("y", "n")]
    if ungraded:
        print(f"Warning: {len(ungraded)} row(s) not yet graded (missing y/n in 'correct' column):")
        for r in ungraded:
            print(f"  [{r['id']}] {r['question']}")
        print()

    graded = [r for r in rows if r["correct"].strip().lower() in ("y", "n")]
    if not graded:
        print("No graded rows yet. Fill in the 'correct' column (y/n) in eval_results.csv first.")
        return

    correct_count = sum(1 for r in graded if r["correct"].strip().lower() == "y")
    overall_accuracy = correct_count / len(graded) * 100

    print(f"Overall accuracy: {correct_count}/{len(graded)} = {overall_accuracy:.1f}%\n")

    by_category = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in graded:
        cat = r["category"]
        by_category[cat]["total"] += 1
        if r["correct"].strip().lower() == "y":
            by_category[cat]["correct"] += 1

    print("By category:")
    for cat, stats in by_category.items():
        pct = stats["correct"] / stats["total"] * 100
        print(f"  {cat:15s} {stats['correct']}/{stats['total']} = {pct:.1f}%")


if __name__ == "__main__":
    main()
