# preprocess_data.py (V2 - 使用 config.py 来管理路径)

import torch
import os
import json
import argparse
from tqdm import tqdm
from datasets import load_dataset
from modelscope import AutoModelForCausalLM, AutoTokenizer
import config # <-- 导入我们的配置

def preprocess_and_save(split: str, target_layer: int, output_dir: str):
    """
    执行所有耗时的预处理工作，并将结果保存到指定的输出目录中。
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 根据split和配置构建完整的文件路径
    if split == 'train':
        final_answers_path = os.path.join(output_dir, config.TRAIN_ANSWERS_FILENAME)
        cot_hiddens_path = os.path.join(output_dir, config.TRAIN_HIDDENS_FILENAME_TEMPLATE.format(target_layer))
    else: # 'test'
        final_answers_path = os.path.join(output_dir, config.TEST_ANSWERS_FILENAME)
        cot_hiddens_path = os.path.join(output_dir, config.TEST_HIDDENS_FILENAME_TEMPLATE.format(target_layer))

    if os.path.exists(final_answers_path) and os.path.exists(cot_hiddens_path):
        print(f"Preprocessed data for split '{split}' already exists in '{output_dir}'. Skipping.")
        return

    # --- 开始执行耗时任务 (内部逻辑与之前完全相同) ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nPreprocessing data for split '{split}' on device '{device}'...")

    # 1. 加载原始数据集
    raw_dataset = load_dataset("gsm8k", "main", split=split, cache_dir=config.RAW_DATASETS_CACHE_DIR, download_mode="reuse_cache_if_exists")
    
    # 2. 加载纯净模型
    tokenizer = AutoTokenizer.from_pretrained("qwen/Qwen2.5-Math-1.5B", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("qwen/Qwen2.5-Math-1.5B", trust_remote_code=True, device_map='auto')
    model.eval()

    # 3. 循环处理 (逻辑不变)
    final_answer_dataset, cot_hiddens_data = [], {}
    with torch.no_grad():
        for idx, example in enumerate(tqdm(raw_dataset, desc=f"Processing {split} split")):
            question, full_answer_text = example['question'], example['answer']
            cot_process, final_answer_part = full_answer_text, ""
            if '####' in full_answer_text:
                parts = full_answer_text.split('####')
                cot_process, final_answer_part = parts[0].strip(), "####" + parts[1].strip()
            final_answer_dataset.append({"question": question, "answer": final_answer_part})
            cot_steps = [step for step in cot_process.split('\n') if step.strip()]
            example_hiddens, current_context = [], question
            for step in cot_steps:
                inputs = tokenizer(current_context + "\n" + step, return_tensors="pt").to(device)
                outputs = model(**inputs, output_hidden_states=True)
                hidden = outputs.hidden_states[target_layer][:, -1, :].detach().cpu()
                example_hiddens.append(hidden)
                current_context += "\n" + step
            cot_hiddens_data[idx] = example_hiddens

    # 4. 保存结果到指定的输出目录
    print(f"\nPreprocessing complete. Saving to '{output_dir}'...")
    with open(final_answers_path, 'w') as f:
        json.dump(final_answer_dataset, f, indent=2)
    print(f"  - Saved final answer data to: {final_answers_path}")
    torch.save(cot_hiddens_data, cot_hiddens_path)
    print(f"  - Saved CoT hidden states to: {cot_hiddens_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess GSM8K dataset.")
    parser.add_argument('--split', type=str, required=True, choices=['train', 'test'], help="Dataset split to process.")
    # 新增：允许从命令行指定输出目录，默认使用config文件中的设置
    parser.add_argument('--output_dir', type=str, default=config.PROCESSED_DATA_DIR, help="Directory to save preprocessed files.")
    parser.add_argument('--target_layer', type=int, default=config.DEFAULT_TARGET_LAYER, help="The transformer layer for hidden state extraction.")
    args = parser.parse_args()
    
    preprocess_and_save(args.split, args.target_layer, args.output_dir)