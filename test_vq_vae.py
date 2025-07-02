# test_vq_vae.py - 测试VQ-VAE效果

import torch
import torch.nn.functional as F
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import config
from vq_vae_module import VQ_VAE
from model_wrapper import TrainableCoTGenerator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_vq_vae_model():
    """加载训练好的VQ-VAE模型"""
    print("Loading VQ-VAE model...")
    
    # 动态获取hidden_size
    temp_wrapper = TrainableCoTGenerator()
    input_dim = temp_wrapper.hidden_size
    del temp_wrapper
    torch.cuda.empty_cache()
    
    # 创建模型
    model = VQ_VAE(
        input_dim=input_dim,
        num_embeddings=config.VQ_VAE_NUM_EMBEDDINGS,
        embedding_dim=config.VQ_VAE_EMBEDDING_DIM,
        commitment_cost=config.VQ_VAE_COMMITMENT_COST
    ).to(device)
    
    # 加载训练好的权重
    if os.path.exists(config.VQ_VAE_CHECKPOINT_PATH):
        model.load_state_dict(torch.load(config.VQ_VAE_CHECKPOINT_PATH, map_location=device))
        print(f"Model loaded from {config.VQ_VAE_CHECKPOINT_PATH}")
    else:
        print(f"Warning: No checkpoint found at {config.VQ_VAE_CHECKPOINT_PATH}")
        print("Testing with untrained model...")
    
    model.eval()
    return model, input_dim

def load_test_data():
    """加载测试数据"""
    print("Loading test data...")
    
    # 加载测试集的hidden states
    test_hiddens_path = os.path.join(config.PROCESSED_DATA_DIR, config.TEST_HIDDENS_FILENAME_TEMPLATE.format(config.DEFAULT_TARGET_LAYER))
    
    if not os.path.exists(test_hiddens_path):
        print(f"Test data not found at {test_hiddens_path}")
        print("Using training data for testing...")
        test_hiddens_path = os.path.join(config.PROCESSED_DATA_DIR, config.TRAIN_HIDDENS_FILENAME_TEMPLATE.format(config.DEFAULT_TARGET_LAYER))
    
    all_hiddens_dict = torch.load(test_hiddens_path)
    test_data = []
    for idx in sorted(all_hiddens_dict.keys()):
        test_data.extend(all_hiddens_dict[idx])
    test_data = [t.squeeze(1) for t in test_data if t.numel() > 0]
    
    print(f"Loaded {len(test_data)} test samples")
    return test_data

def test_reconstruction_quality(model, test_data, input_dim, num_samples=1000):
    """测试重构质量"""
    print(f"\n=== Testing Reconstruction Quality (using {min(num_samples, len(test_data))} samples) ===")
    
    model.eval()
    reconstruction_losses = []
    vq_losses = []
    perplexities = []
    
    # 随机选择样本进行测试
    indices = np.random.choice(len(test_data), min(num_samples, len(test_data)), replace=False)
    
    with torch.no_grad():
        for idx in tqdm(indices, desc="Testing reconstruction"):
            data = test_data[idx].to(device)  # 确保是1D tensor
            
            # 检查并修正数据形状
            if data.dim() > 1:
                data = data.view(-1)  # 展平为1D
            if data.size(0) != input_dim:
                print(f"Warning: Sample {idx} has wrong dimension {data.size(0)}, expected {input_dim}")
                continue
                
            data = data.unsqueeze(0)  # (1, input_dim)
            
            # 前向传播
            vq_loss, data_recon, perplexity = model(data)
            
            # 计算重构损失
            recon_loss = F.mse_loss(data_recon, data)
            
            reconstruction_losses.append(recon_loss.item())
            vq_losses.append(vq_loss.item())
            perplexities.append(perplexity.item())
    
    # 计算统计信息
    avg_recon_loss = np.mean(reconstruction_losses)
    avg_vq_loss = np.mean(vq_losses)
    avg_perplexity = np.mean(perplexities)
    
    print(f"Average Reconstruction Loss: {avg_recon_loss:.6f}")
    print(f"Average VQ Loss: {avg_vq_loss:.6f}")
    print(f"Average Perplexity: {avg_perplexity:.2f}")
    
    return {
        'recon_losses': reconstruction_losses,
        'vq_losses': vq_losses,
        'perplexities': perplexities,
        'avg_recon_loss': avg_recon_loss,
        'avg_vq_loss': avg_vq_loss,
        'avg_perplexity': avg_perplexity
    }

def visualize_reconstruction(model, test_data, input_dim, num_examples=5):
    """可视化重构结果"""
    print(f"\n=== Visualizing Reconstruction (showing {num_examples} examples) ===")
    
    model.eval()
    
    # 随机选择几个样本
    indices = np.random.choice(len(test_data), num_examples, replace=False)
    
    fig, axes = plt.subplots(num_examples, 2, figsize=(12, 3*num_examples))
    if num_examples == 1:
        axes = axes.reshape(1, -1)
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            original = test_data[idx].to(device)
            
            # 检查并修正数据形状
            if original.dim() > 1:
                original = original.view(-1)
            if original.size(0) != input_dim:
                print(f"Warning: Sample {idx} has wrong dimension, skipping...")
                continue
                
            original_np = original.cpu().numpy()
            
            # 重构
            vq_loss, reconstructed, perplexity = model(original.unsqueeze(0))
            reconstructed_np = reconstructed.squeeze(0).cpu().numpy()
            
            # 绘制原始和重构的向量
            axes[i, 0].plot(original_np, label='Original', alpha=0.7)
            axes[i, 0].set_title(f'Original Hidden State {idx}')
            axes[i, 0].set_xlabel('Dimension')
            axes[i, 0].set_ylabel('Value')
            axes[i, 0].legend()
            
            axes[i, 1].plot(reconstructed_np, label='Reconstructed', alpha=0.7)
            axes[i, 1].set_title(f'Reconstructed (Perplexity: {perplexity.item():.2f})')
            axes[i, 1].set_xlabel('Dimension')
            axes[i, 1].set_ylabel('Value')
            axes[i, 1].legend()
            
            # 计算这个样本的重构损失
            recon_loss = F.mse_loss(reconstructed.squeeze(0), original)
            print(f"Sample {idx}: Recon Loss = {recon_loss.item():.6f}, Perplexity = {perplexity.item():.2f}")
    
    plt.tight_layout()
    plt.savefig('vq_vae_reconstruction_visualization.png', dpi=150, bbox_inches='tight')
    print("Visualization saved as 'vq_vae_reconstruction_visualization.png'")
    plt.show()

def analyze_codebook_usage(model, test_data, input_dim, num_samples=1000):
    """分析码本使用情况"""
    print(f"\n=== Analyzing Codebook Usage (using {min(num_samples, len(test_data))} samples) ===")
    
    model.eval()
    codebook_usage = torch.zeros(config.VQ_VAE_NUM_EMBEDDINGS).to(device)
    
    # 随机选择样本
    indices = np.random.choice(len(test_data), min(num_samples, len(test_data)), replace=False)
    
    with torch.no_grad():
        for idx in tqdm(indices, desc="Analyzing codebook usage"):
            data = test_data[idx].to(device)
            
            # 检查并修正数据形状
            if data.dim() > 1:
                data = data.view(-1)
            if data.size(0) != input_dim:
                continue
                
            data = data.unsqueeze(0)
            
            # 获取编码器输出
            z = model._encoder(data)
            
            # 计算距离并找到最近的码向量
            sum_inputs_sq = torch.sum(z.pow(2.0), dim=1, keepdim=True)
            sum_embed_sq = torch.sum(model._vq_layer._embedding.weight.pow(2.0), dim=1)
            dot_product = torch.matmul(z, model._vq_layer._embedding.weight.t())
            distances = sum_inputs_sq - 2 * dot_product + sum_embed_sq.unsqueeze(0)
            encoding_indices = torch.argmin(distances, dim=1)
            
            # 统计使用次数
            for idx_code in encoding_indices:
                codebook_usage[idx_code] += 1
    
    # 计算使用统计
    total_usage = codebook_usage.sum().item()
    active_codes = (codebook_usage > 0).sum().item()
    usage_rate = active_codes / config.VQ_VAE_NUM_EMBEDDINGS * 100
    
    print(f"Total codebook entries: {config.VQ_VAE_NUM_EMBEDDINGS}")
    print(f"Active codes: {active_codes}")
    print(f"Codebook usage rate: {usage_rate:.2f}%")
    print(f"Most used code: {codebook_usage.argmax().item()} (used {codebook_usage.max().item():.0f} times)")
    print(f"Least used code: {codebook_usage.argmin().item()} (used {codebook_usage.min().item():.0f} times)")
    
    # 绘制码本使用分布
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.bar(range(config.VQ_VAE_NUM_EMBEDDINGS), codebook_usage.cpu().numpy())
    plt.title('Codebook Usage Distribution')
    plt.xlabel('Code Index')
    plt.ylabel('Usage Count')
    
    plt.subplot(1, 2, 2)
    sorted_usage = torch.sort(codebook_usage, descending=True)[0].cpu().numpy()
    plt.plot(sorted_usage)
    plt.title('Codebook Usage (Sorted)')
    plt.xlabel('Code Index (Sorted)')
    plt.ylabel('Usage Count')
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig('vq_vae_codebook_usage.png', dpi=150, bbox_inches='tight')
    print("Codebook usage analysis saved as 'vq_vae_codebook_usage.png'")
    plt.show()
    
    return {
        'total_usage': total_usage,
        'active_codes': active_codes,
        'usage_rate': usage_rate,
        'codebook_usage': codebook_usage.cpu().numpy()
    }

def main():
    """主测试函数"""
    print("=== VQ-VAE Testing Script ===")
    
    # 加载模型
    model, input_dim = load_vq_vae_model()
    
    # 加载测试数据
    test_data = load_test_data()
    
    if len(test_data) == 0:
        print("No test data available!")
        return
    
    # 测试重构质量
    results = test_reconstruction_quality(model, test_data, input_dim)
    
    # 可视化重构结果
    visualize_reconstruction(model, test_data, input_dim)
    
    # 分析码本使用情况
    codebook_stats = analyze_codebook_usage(model, test_data, input_dim)
    
    # 总结
    print("\n=== Test Summary ===")
    print(f"Model: VQ-VAE with {config.VQ_VAE_NUM_EMBEDDINGS} codebook entries")
    print(f"Input dimension: {input_dim}")
    print(f"Embedding dimension: {config.VQ_VAE_EMBEDDING_DIM}")
    print(f"Average reconstruction loss: {results['avg_recon_loss']:.6f}")
    print(f"Average perplexity: {results['avg_perplexity']:.2f}")
    print(f"Codebook usage rate: {codebook_stats['usage_rate']:.2f}%")
    
    # 评估模型质量
    if results['avg_recon_loss'] < 0.1:
        print("✅ Reconstruction quality: GOOD")
    elif results['avg_recon_loss'] < 0.5:
        print("⚠️  Reconstruction quality: FAIR")
    else:
        print("❌ Reconstruction quality: POOR")
    
    if codebook_stats['usage_rate'] > 80:
        print("✅ Codebook utilization: GOOD")
    elif codebook_stats['usage_rate'] > 50:
        print("⚠️  Codebook utilization: FAIR")
    else:
        print("❌ Codebook utilization: POOR")

if __name__ == '__main__':
    main() 