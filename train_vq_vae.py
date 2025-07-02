# train_vq_vae.py (V5 - 最终版，包含完善的指标记录和绘图功能)

import torch
import torch.optim as optim
import torch.nn.functional as F
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import config
from vq_vae_module import get_vq_vae_model 
from model_wrapper import TrainableCoTGenerator

# 定义全局设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def plot_training_curves(recon_losses, vq_losses, perplexities, num_epochs):
    """一个辅助函数，用于绘制并保存训练曲线图。"""
    plt.figure(figsize=(18, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(range(1, num_epochs + 1), recon_losses, 'r-', label='Reconstruction Loss')
    plt.title('Avg. Reconstruction Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(range(1, num_epochs + 1), vq_losses, 'g-', label='VQ Loss')
    plt.title('Avg. VQ Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(range(1, num_epochs + 1), perplexities, 'm-', label='Perplexity')
    plt.title('Avg. Codebook Perplexity per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('vq_vae_training_curves.png', dpi=150)
    print("Training curves saved as 'vq_vae_training_curves.png'")

def main():
    # --- 超参数设置 ---
    TARGET_LAYER = config.DEFAULT_TARGET_LAYER
    NUM_EPOCHS = 40
    LEARNING_RATE = 3e-4
    BATCH_SIZE = 256
    COMMITMENT_COST = config.VQ_VAE_COMMITMENT_COST

    # --- 加载数据并进行标准化 ---
    hiddens_path = os.path.join(config.PROCESSED_DATA_DIR, config.TRAIN_HIDDENS_FILENAME_TEMPLATE.format(TARGET_LAYER))
    all_hiddens_dict = torch.load(hiddens_path, map_location='cpu')
    training_data_tensors = [t.squeeze(0) for h_list in all_hiddens_dict.values() for t in h_list if t.numel() > 0]
    full_data_tensor = torch.stack(training_data_tensors)
    data_mean = torch.mean(full_data_tensor, dim=0).to(device)
    data_std = torch.std(full_data_tensor, dim=0).to(device)
    data_std[data_std < 1e-6] = 1.0
    print(f"Loaded and normalized {len(training_data_tensors)} vectors.")

    # --- 获取模型维度 ---
    input_dim = 1536 

    # --- 初始化模型与优化器 ---
    model = get_vq_vae_model(
        architecture=config.VQ_VAE_ARCHITECTURE,
        input_dim=input_dim,
        num_embeddings=config.VQ_VAE_NUM_EMBEDDINGS,
        embedding_dim=config.VQ_VAE_EMBEDDING_DIM,
        commitment_cost=COMMITMENT_COST
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    
    warmup_epochs = 5
    lr_lambda = lambda epoch: (epoch + 1) / warmup_epochs if epoch < warmup_epochs else \
                0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (NUM_EPOCHS - warmup_epochs)))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # ========================================================================
    #  变化 1：初始化用于记录每个epoch历史的列表
    # ========================================================================
    recon_losses, vq_losses, perplexities = [], [], []
    
    print("\n--- Starting VQ-VAE Training with Full Logging ---")
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loader = torch.utils.data.DataLoader(full_data_tensor, batch_size=BATCH_SIZE, shuffle=True)
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        
        # 用于累积当前epoch内所有batch的指标
        epoch_recon, epoch_vq, epoch_perp = [], [], []
        code_usage_counts = torch.zeros(config.VQ_VAE_NUM_EMBEDDINGS, device=device)

        for data_batch in pbar:
            data_batch = data_batch.to(device)
            normalized_batch = (data_batch - data_mean) / data_std
            
            optimizer.zero_grad()
            
            # --- 前向传播，获取所有返回值 ---
            z = model._encoder(normalized_batch)
            vq_loss, quantized, perplexity, indices = model._vq_layer(z)
            data_recon = model._decoder(quantized)
            
            # --- 统计码本使用情况 (用于重置) ---
            indices_one_hot = F.one_hot(indices.squeeze(1), num_classes=config.VQ_VAE_NUM_EMBEDDINGS).float()
            code_usage_counts += indices_one_hot.sum(0)
            
            # --- 计算损失并反向传播 (不变) ---
            recon_loss = F.mse_loss(data_recon, normalized_batch)
            total_loss = recon_loss + vq_loss
            total_loss.backward()
            optimizer.step()
            
            # ========================================================================
            #  变化 2：将当前batch的指标存入epoch列表
            # ========================================================================
            epoch_recon.append(recon_loss.item())
            epoch_vq.append(vq_loss.item())
            epoch_perp.append(perplexity.item())
            
            pbar.set_postfix({"Recon": f"{recon_loss.item():.4f}", "VQ": f"{vq_loss.item():.4f}", "Perp": f"{perplexity.item():.2f}"})
        
        scheduler.step()
        
        # ========================================================================
        #  变化 3：整合您提供的代码，计算并记录整个epoch的平均指标
        # ========================================================================
        avg_recon_loss = np.mean(epoch_recon)
        avg_vq_loss = np.mean(epoch_vq)
        avg_perplexity = np.mean(epoch_perp)
        
        # 添加到历史记录中，用于最终绘图
        recon_losses.append(avg_recon_loss)
        vq_losses.append(avg_vq_loss)
        perplexities.append(avg_perplexity)
        
        # 打印这个epoch的总结
        print(f"--- Epoch {epoch+1} Summary: Avg Recon Loss={avg_recon_loss:.4f}, Avg VQ Loss={avg_vq_loss:.4f}, "
              f"Avg Perplexity={avg_perplexity:.2f}, LR={optimizer.param_groups[0]['lr']:.6f} ---")
        # ========================================================================

        # --- 码本重置逻辑 (不变) ---
        with torch.no_grad():
            dead_codes = torch.where(code_usage_counts == 0)[0]
            if len(dead_codes) > 0:
                print(f"Epoch {epoch+1}: Found {len(dead_codes)} dead codes. Resetting...")
                popular_codes = torch.where(code_usage_counts > 0)[0]
                if len(popular_codes) > 0:
                    random_popular_indices = torch.randint(0, len(popular_codes), (len(dead_codes),))
                    replacement_vectors = model._vq_layer._embedding.weight.data[popular_codes[random_popular_indices]]
                    model._vq_layer._embedding.weight.data[dead_codes] = replacement_vectors

    # --- 训练结束 ---
    print("\n--- Training Finished ---")
    os.makedirs(config.CHECKPOINTS_DIR, exist_ok=True)
    torch.save(model.state_dict(), config.VQ_VAE_CHECKPOINT_PATH)
    print(f"Final VQ-VAE model saved to {config.VQ_VAE_CHECKPOINT_PATH}")

    # 使用记录好的历史数据进行绘图
    plot_training_curves(recon_losses, vq_losses, perplexities, NUM_EPOCHS)

if __name__ == '__main__':
    main()