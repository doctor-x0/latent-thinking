import json
from datasets import load_dataset

OUTPUT_PATH = "gsm8k_test_cot.jsonl"

# 加载gsm8k原始数据集（test split）
dataset = load_dataset("gsm8k", "main", split="test")

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for item in dataset:
        question = item["question"]
        answer = item["answer"]
        if "####" in answer:
            cot, final = answer.split("####", 1)
            steps = [s.strip() for s in cot.split("\n") if s.strip()]
            final = final.strip()
        else:
            steps = [answer.strip()] if answer.strip() else []
            final = ""
        out = {
            "question": question,
            "steps": steps,
            "final_answer": final
        }
        f.write(json.dumps(out, ensure_ascii=False) + "\n")

print(f"✅ 已保存到 {OUTPUT_PATH}") 