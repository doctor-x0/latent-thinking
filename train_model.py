# train.py (V9 - 多任务学习 + 样本数量控制)

import torch
import torch.optim as optim
import json
import os
import argparse  # <-- 新增
import random    # <-- 新增
from model_wrapper import TrainableCoTGenerator
import config 
import torch.nn as nn

def main():
    # ========================================================================
    #  新增：使用 argparse 定义和解析命令行参数
    # ========================================================================
    parser = argparse.ArgumentParser(description="Train the custom filtering module with multi-task loss.")
    parser.add_argument(
        '--samples', 
        type=int, 
        default=200, 
        help="Number of data samples to use for training. Set to -1 to use all available data."
    )
    args = parser.parse_args()
    # ========================================================================

    # --- 从配置加载参数 ---
    NUM_EPOCHS = 10
    LEARNING_RATE = 5e-5
    NUM_COT_LOOPS = 5 
    TARGET_LAYER = config.DEFAULT_TARGET_LAYER
    
    # --- 加载所有数据 ---
    train_data_path = os.path.join(config.PROCESSED_DATA_DIR, config.TRAIN_ANSWERS_FILENAME)
    hiddens_path = os.path.join(config.PROCESSED_DATA_DIR, config.TRAIN_HIDDENS_FILENAME_TEMPLATE.format(TARGET_LAYER))

    print(f"Loading final answer data from: {train_data_path}")
    if not os.path.exists(train_data_path):
        # ... (错误处理不变)
        return
    with open(train_data_path, 'r') as f:
        full_dataset = json.load(f)

    print(f"Loading precomputed CoT hiddens from: {hiddens_path}")
    if not os.path.exists(hiddens_path):
        # ... (错误处理不变)
        return
    full_target_hiddens = torch.load(hiddens_path)

    # ========================================================================
    #  变化：根据 --samples 参数对 QA 和 Hiddens 数据进行同步切片
    # ========================================================================
    num_available = len(full_dataset)
    indices = list(range(num_available))
    random.shuffle(indices) # 打乱索引

    if args.samples == -1:
        # 使用所有数据
        selected_indices = indices
        print(f"Using all {num_available} available samples for training.")
    else:
        # 使用指定数量的数据
        num_to_use = min(args.samples, num_available)
        selected_indices = indices[:num_to_use]
        print(f"Using a random subset of {len(selected_indices)} samples for training.")
    
    # 根据筛选出的索引创建新的数据集和hiddens字典
    dataset = [full_dataset[i] for i in selected_indices]
    target_hiddens_all = {new_idx: full_target_hiddens[old_idx] for new_idx, old_idx in enumerate(selected_indices)}
    # ========================================================================
    
    # --- 初始化和训练 ---
    print("Initializing model...")
    model_wrapper = TrainableCoTGenerator()
    # model_wrapper.hiddens_loss_fn = nn.MSELoss() # 这一行在 model_wrapper 的 __init__ 中已经做了
    trainable_module = model_wrapper.get_trainable_module()
    optimizer = optim.AdamW(trainable_module.parameters(), lr=LEARNING_RATE)
    
    print("\n--- Starting Multi-Task Training ---")
    for epoch in range(NUM_EPOCHS):
        total_loss, total_ans_loss, total_hid_loss = 0, 0, 0
        # 现在的 dataset 和 target_hiddens_all 是同步且经过切片的
        for i, data in enumerate(dataset):
            question = data["question"]
            final_answer = data["answer"]
            
            target_hiddens = target_hiddens_all.get(i)
            if not target_hiddens or not final_answer:
                continue

            # (调用模型的逻辑完全不变)
            combined_loss, answer_loss, hiddens_loss = model_wrapper.forward_for_loss(
                question=question,
                target_answer_text=final_answer,
                target_hiddens_steps=target_hiddens,
                num_loops=NUM_COT_LOOPS,
                target_layer=TARGET_LAYER
            )
            
            optimizer.zero_grad()
            combined_loss.backward()
            optimizer.step()
            
            # (累加和打印损失的逻辑完全不变)
            total_loss += combined_loss.item()
            total_ans_loss += answer_loss
            total_hid_loss += hiddens_loss
            print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Step {i+1}/{len(dataset)} | Total Loss: {combined_loss.item():.4f} (Ans: {answer_loss:.4f}, Hid: {hiddens_loss:.4f})", end='\r')
        
        avg_loss = total_loss / len(dataset)
        avg_ans_loss = total_ans_loss / len(dataset)
        avg_hid_loss = total_hid_loss / len(dataset)
        print()
        print(f"--- Epoch {epoch+1} Finished | Avg Total Loss: {avg_loss:.4f} (Avg Ans: {avg_ans_loss:.4f}, Avg Hid: {avg_hid_loss:.4f}) ---")

    print("\n--- Training Finished ---")
    os.makedirs(config.CHECKPOINTS_DIR, exist_ok=True)
    model_wrapper.save_trainable_module(config.TRAINED_WEIGHTS_PATH)

if __name__ == '__main__':
    main()