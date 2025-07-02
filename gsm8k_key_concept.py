import json
import re
from openai import OpenAI
from datasets import load_dataset

client = OpenAI(
    api_key="sk-2NSYfGn6ZikoJOBGKa8pCn2n6Y2FVg1jIMBPPmLcnHuYDu4t",  # 你的API Key
    base_url="https://api.moonshot.cn/v1",
)

K = 3
OUTPUT_PATH = "kimi_gsm8k_concepts.jsonl"

# 用datasets库加载gsm8k原始数据集（test split），只取前100条
data = []
dataset = load_dataset("gsm8k", "main", split="test")
for item in dataset.select(range(50)):
    q = item["question"]
    a = item["answer"]
    # 提取推理步骤
    if "####" in a:
        steps = [s.strip() for s in a.split("####")[0].split("\n") if s.strip()]
    else:
        steps = [a.strip()] if a.strip() else []
    data.append({"question": q, "steps": steps})
    

print(f"\n🎯 正在处理前50个样本...")
# 构造大prompt
prompt = f"""
你是一个数学推理概念抽取助手。请针对每个推理步骤，精确抽取 {K} 个最关键的token，这些token能够代表该步骤的推理过程。请严格返回一个嵌套JSON数组，每个子数组对应每一步，如 [[\"概念1\", \"概念2\", \"概念3\"], ...]。

下面是{len(data)}个样本的推理步骤：
"""
for idx, item in enumerate(data):
    steps_text = "\n".join([f"{i+1}. {s}" for i, s in enumerate(item["steps"])])
    prompt += f"\n【样本{idx+1}】\n推理步骤：\n{steps_text}\n"
prompt += "\n请依次输出每个样本的嵌套JSON数组，顺序与上面样本一致。"

completion = client.chat.completions.create(
    model = "moonshot-v1-8k",
    messages = [
        {"role": "system", "content": "你是 Kimi，由 Moonshot AI 提供的助手，你擅长中文与英文推理，返回严格JSON数组。"},
        {"role": "user", "content": prompt}
    ],
    temperature = 0.2,
)
text = completion.choices[0].message.content
print(f"➡ 输出: {text[:200]} ...")
# 清洗掉 markdown 代码块标记
text = re.sub(r"^```json\\s*|^```|```$", "", text.strip(), flags=re.MULTILINE).strip()
# 尝试按行解析每个样本的JSON
results = []
try:
    batch_concepts = json.loads(text)
    for i, item in enumerate(data):
        results.append({
            "question": item["question"],
            "concept_steps": batch_concepts[i] if i < len(batch_concepts) else []
        })
except Exception as e:
    print(f"⚠ JSON解析失败: {e}")
    for item in data:
        results.append({
            "question": item["question"],
            "concept_steps": []
        })

# 保存文件
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for item in results:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"\n✅ 前100个样本处理完成，已保存到 {OUTPUT_PATH}")
