"""
Runs every question in eval_questions.csv through the agent and saves the answers
to eval_results.csv, so you can grade them and compare results to check accuracy

Usage:
    python run_eval.py
"""
import csv
from agent import ask_agent

INPUT_FILE = "eval_questions.csv"
OUTPUT_FILE = "eval_results.csv"


def main():
    with open(INPUT_FILE, newline="", encoding="utf-8") as f:
        questions = list(csv.DictReader(f))

    results = []
    for row in questions:
        print(f"[{row['id']}] {row['question']}")
        answer = ask_agent(row["question"])
        print(f"  -> {answer[:200]}{'...' if len(answer) > 200 else ''}\n")
        results.append(
            {
                "id": row["id"],
                "question": row["question"],
                "category": row["category"],
                "answer": answer,
                "correct": "",  # fill in y/n yourself after reviewing
                "notes": "",    # optional: why it was wrong, what broke
            }
        )

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["id", "question", "category", "answer", "correct", "notes"]
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved {len(results)} results to {OUTPUT_FILE}.")
    print("Open it, review each 'answer' column, and mark 'correct' as y or n before running score_eval.py.")


if __name__ == "__main__":
    main()
