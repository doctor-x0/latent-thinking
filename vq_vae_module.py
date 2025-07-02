# vq_vae_module.py (V2 - 修正了距离计算的维度问题)

import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    """
    向量量化层 (Vector Quantization Layer)。
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        input_shape = inputs.shape
        
        # 添加调试信息
        print(f"Debug - inputs shape: {inputs.shape}")
        print(f"Debug - embedding weight shape: {self._embedding.weight.shape}")
        
        # ========================================================================
        #  变化：使用更稳健的方式计算距离，避免广播错误
        # ========================================================================
        # 计算 L2 距离的平方: (a-b)^2 = a^2 - 2ab + b^2
        # a^2: (B, D) -> sum(dim=1) -> (B, 1)
        sum_inputs_sq = torch.sum(inputs.pow(2.0), dim=1, keepdim=True)
        # b^2: (K, D) -> sum(dim=1) -> (K,) -> (1, K)
        sum_embed_sq = torch.sum(self._embedding.weight.pow(2.0), dim=1)
        
        # 2ab: (B, D) @ (D, K) -> (B, K)
        dot_product = torch.matmul(inputs, self._embedding.weight.t())
        
        print(f"Debug - sum_inputs_sq shape: {sum_inputs_sq.shape}")
        print(f"Debug - sum_embed_sq shape: {sum_embed_sq.shape}")
        print(f"Debug - dot_product shape: {dot_product.shape}")
        print(f"Debug - sum_embed_sq.unsqueeze(0) shape: {sum_embed_sq.unsqueeze(0).shape}")
        
        # (B, 1) - 2 * (B, K) + (1, K) -> (B, K)  (PyTorch会正确广播)
        distances = sum_inputs_sq - 2 * dot_product + sum_embed_sq.unsqueeze(0)
        # ========================================================================

        # 找到最近的码向量索引
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        
        # ... (后续代码完全不变) ...
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(inputs, quantized.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()
        
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        return loss, quantized, perplexity, encoding_indices

class VQ_VAE(nn.Module):
    """
    完整的VQ-VAE模型 (此部分无需修改)
    """
    def __init__(self, input_dim, num_embeddings, embedding_dim, commitment_cost):
        super(VQ_VAE, self).__init__()
        
        self._encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, embedding_dim)
        )
        
        self._vq_layer = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        
        self._decoder = nn.Sequential(
            nn.Linear(embedding_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, input_dim)
        )

    def forward(self, x):
        z = self._encoder(x)
        vq_loss, quantized, perplexity, _ = self._vq_layer(z)
        x_recon = self._decoder(quantized)
        
        return vq_loss, x_recon, perplexity


class ResidualBlock(nn.Module):
    """一个简单的残差块"""
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )

    def forward(self, x):
        return x + self.net(x)

class ResidualVQ_VAE(nn.Module):
    """
    方案二：使用残差块构建的更强大的VQ-VAE。
    """
    def __init__(self, input_dim=1536, hidden_dim=768, num_embeddings=1024, embedding_dim=256, commitment_cost=0.25):
        super().__init__()
        
        # Encoder: 1536 -> 768 -> ResBlock -> ResBlock -> 256
        self._encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), # 1536 -> 768
            nn.GELU(),
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
            nn.Linear(hidden_dim, embedding_dim) # 768 -> 256
        )
        
        self._vq_layer = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        
        # Decoder: 256 -> 768 -> ResBlock -> ResBlock -> 1536
        self._decoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim), # 256 -> 768
            nn.GELU(),
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
            nn.Linear(hidden_dim, input_dim) # 768 -> 1536
        )

    def forward(self, x):
        z = self._encoder(x)
        vq_loss, quantized, perplexity, _ = self._vq_layer(z)
        x_recon = self._decoder(quantized)
        
        return vq_loss, x_recon, perplexity


def get_vq_vae_model(architecture: str, **kwargs):
    """
    工厂函数，根据名称返回对应的VQ-VAE模型实例。
    """
    if architecture == 'VQ_VAE':
        return VQ_VAE(**kwargs)
    elif architecture == 'ResidualVQ_VAE':
        return ResidualVQ_VAE(**kwargs)
    else:
        raise ValueError(f"Unknown VQ-VAE architecture: {architecture}")