# inspect_golden_hiddens.py

import torch
import os
import json
from model_wrapper import TrainableCoTGenerator
import config

def inspect_hiddens(num_samples_to_inspect: int = 3, top_k: int = 10):
    """
    加载并分析由 preprocess_data.py 生成的“黄金”hiddens。
    """
    print("="*60)
    print("Inspecting Golden Hidden States from Preprocessed Data")
    print("="*60)

    # --- 1. 定义并加载文件 ---
    TARGET_LAYER = config.DEFAULT_TARGET_LAYER
    # 我们将分析测试集的数据，以模拟真实评估场景
    test_data_path = os.path.join(config.PROCESSED_DATA_DIR, config.TEST_ANSWERS_FILENAME)
    hiddens_path = os.path.join(config.PROCESSED_DATA_DIR, config.TEST_HIDDENS_FILENAME_TEMPLATE.format(TARGET_LAYER))

    print(f"Loading test questions from: {test_data_path}")
    if not os.path.exists(test_data_path):
        print("Error: Test data not found. Please run 'python preprocess_data.py --split test' first.")
        return
    with open(test_data_path, 'r') as f:
        qa_dataset = json.load(f)

    print(f"Loading precomputed hiddens from: {hiddens_path}")
    if not os.path.exists(hiddens_path):
        print("Error: Precomputed hiddens not found. Please run 'python preprocess_data.py --split test' first.")
        return
    golden_hiddens_all = torch.load(hiddens_path)
    
    # --- 2. 初始化模型包装器，以获取 tokenizer 和 lm_head ---
    # 我们只需要一个模型实例来进行投影，不需要训练它
    print("\nInitializing a model instance for projection...")
    model_wrapper = TrainableCoTGenerator()
    model = model_wrapper.model
    tokenizer = model_wrapper.tokenizer
    model.eval() # 确保在评估模式
    device = next(model.parameters()).device # 获取模型所在的设备

    # --- 3. 循环分析指定数量的样本 ---
    for i in range(min(num_samples_to_inspect, len(qa_dataset))):
        print("\n" + "-"*25 + f" Sample #{i+1} " + "-"*25)
        
        question = qa_dataset[i]['question']
        print(f"\n[Question]: {question}")
        
        # 获取该样本对应的“黄金”hiddens列表
        hidden_steps = golden_hiddens_all.get(i)
        
        if not hidden_steps:
            print("  -> No hidden steps found for this sample.")
            continue
            
        print(f"\n[Analysis of {len(hidden_steps)} Golden CoT Steps]:")
        
        with torch.no_grad():
            for step_idx, hidden_state in enumerate(hidden_steps):
                print(f"\n--- Step {step_idx + 1} ---")
                
                # a) 打印形状
                print(f"  Shape of Hidden State: {hidden_state.shape}")
                
                # b) 投影到词表空间
                # 将 hidden state 移动到模型所在的设备
                hidden_state = hidden_state.to(device, dtype=model.dtype)
                
                # 使用模型的 lm_head 进行投影
                logits = model.lm_head(hidden_state)
                
                # 获取 Top-K 的词和它们的分数
                top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)
                
                # 解码为可读的词语
                # top_k_indices 的形状是 (1, top_k)，所以需要 squeeze()
                top_k_tokens = tokenizer.convert_ids_to_tokens(top_k_indices.squeeze().tolist())
                
                print(f"  Top {top_k} Projected Tokens:")
                for token, logit_val in zip(top_k_tokens, top_k_logits.squeeze().tolist()):
                    # 使用 Ġ (U+0120) 来表示一个词的开头，这是很多分词器（如BPE）的惯例
                    clean_token = token.replace('Ġ', ' ').strip()
                    print(f"    - '{clean_token}' (logit: {logit_val:.4f})")

if __name__ == '__main__':
    # 您可以在这里修改想分析的样本数量
    inspect_hiddens(num_samples_to_inspect=3)