# filtering_modules.py

import torch
import torch.nn as nn

# ========================================================================
#  架构一：扩展型 MLP (ExpansionMLP) - 基于您的版本
# ========================================================================
class ExpansionMLP(nn.Module):
    """
    先将维度扩大，再映射回来。
    这可能允许网络在更高维空间中学习特征的复杂交互。
    """
    def __init__(self, hidden_size: int, expansion_factor: int = 2, dropout_rate: float = 0.1):
        super().__init__()
        expanded_dim = hidden_size * expansion_factor
        self.net = nn.Sequential(
            nn.Linear(hidden_size, expanded_dim),
            nn.GELU(), # GELU 是比 ReLU 更现代、平滑的激活函数
            nn.Dropout(dropout_rate),
            nn.Linear(expanded_dim, hidden_size)
        )
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        # 注意：这里我们应用一个残差连接，这几乎总是能带来好处
        return self.layer_norm(x + self.net(x))

# ========================================================================
#  架构二：瓶颈型 MLP (BottleneckMLP)
# ========================================================================
class BottleneckMLP(nn.Module):
    """
    先将维度压缩（瓶颈），再恢复。
    这会迫使模型学习输入的更高效、更精华的压缩表示。
    """
    def __init__(self, hidden_size: int, bottleneck_factor: int = 4):
        super().__init__()
        bottleneck_dim = hidden_size // bottleneck_factor
        self.net = nn.Sequential(
            nn.Linear(hidden_size, bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, hidden_size)
        )

    def forward(self, x):
        return self.net(x)

# ========================================================================
#  架构三：残差 MLP (ResidualMLP) - 强烈推荐
# ========================================================================
class ResidualMLP(nn.Module):
    """
    将瓶颈型MLP与残差连接相结合。
    这是最经典、最稳健的设计之一，让模块学习对输入的“修正量”。
    """
    def __init__(self, hidden_size: int, bottleneck_factor: int = 4):
        super().__init__()
        self.bottleneck_mlp = BottleneckMLP(hidden_size, bottleneck_factor)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, x):
        # 经典的 Pre-LN 结构: x -> Norm -> MLP -> + x
        return x + self.bottleneck_mlp(self.layer_norm(x))

# ========================================================================
#  架构四：门控 MLP (GatedMLP)
# ========================================================================
class GatedMLP(nn.Module):
    """
    使用门控机制动态地融合原始输入和变换后的结果。
    这赋予了模块决定“更新”多少信息的灵活性。
    """
    def __init__(self, hidden_size: int, bottleneck_factor: int = 4):
        super().__init__()
        self.transformed_path = BottleneckMLP(hidden_size, bottleneck_factor)
        self.gate_path = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid()
        )
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        # F(x)
        transformed_x = self.transformed_path(x)
        # g(x)
        gate = self.gate_path(x)
        # output = (1-g)*x + g*F(x)
        output = (1 - gate) * x + gate * transformed_x
        return self.layer_norm(output)

# ========================================================================
#  新增：高级模块定义
# ========================================================================

class StatefulGRUModule(nn.Module):
    """方案五：带有记忆的循环模块"""
    def __init__(self, hidden_size, **kwargs):
        super().__init__()
        # GRUCell 输入是LLM的h_t，隐藏状态是模块自己的记忆m_t
        self.gru_cell = nn.GRUCell(hidden_size, hidden_size)

    def forward(self, x, state):
        # x: 当前LLM的hidden_state (h_t)
        # state: 模块上一时刻的记忆 (m_{t-1})
        if state is None:
            # 初始化记忆状态为0
            state = torch.zeros_like(x)
        new_state = self.gru_cell(x, state)
        return new_state, new_state # 输出既是本次结果，也是下一轮的state

class CrossAttentionModule(nn.Module):
    """方案六：带有注意力的模块"""
    def __init__(self, hidden_size, n_heads=4, **kwargs):
        super().__init__()
        # 使用PyTorch内置的多头注意力机制
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=n_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size)
        )
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x, context):
        # x: 当前LLM的hidden_state (h_t), shape: (B, 1, D)
        # context: 原始问题的嵌入, shape: (B, SeqLen, D)
        
        # MultiheadAttention需要 B, Seq, Dim 的格式
        # h_t作为query，需要从 (B, D) unsqueeze到 (B, 1, D)
        attn_output, _ = self.attention(query=x.unsqueeze(1), key=context, value=context)
        
        # 将注意力的输出和原始输入拼接
        combined = torch.cat([x, attn_output.squeeze(1)], dim=1)
        
        # 通过MLP进行最终变换
        output = self.mlp(combined)
        return self.layer_norm(x + output), None # 残差连接, 无需返回state

class StatefulAttentionModule(nn.Module):
    """方案七：带记忆和注意力的模块"""
    def __init__(self, hidden_size, n_heads=4, **kwargs):
        super().__init__()
        self.gru_cell = nn.GRUCell(hidden_size, hidden_size)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=n_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x, state, context):
        # x: 当前LLM的hidden_state (h_t)
        # state: 模块上一时刻的记忆 (m_{t-1})
        # context: 原始问题的嵌入
        if state is None:
            state = torch.zeros_like(x)

        # 1. 更新内部记忆
        memory = self.gru_cell(x, state) # m_t

        # 2. 用新记忆作为Query进行注意力计算
        attn_output, _ = self.attention(query=memory.unsqueeze(1), key=context, value=context)

        # 3. 融合记忆和注意力上下文
        output = self.layer_norm(memory + attn_output.squeeze(1))
        
        return output, memory # 返回最终输出，和新的记忆状态

# ========================================================================
#  更新后的模块工厂函数
# ========================================================================
def get_filtering_module(module_name: str, **kwargs):
    hidden_size = kwargs.get("hidden_size")
    device = kwargs.get("device")
    dtype = kwargs.get("dtype")

    module_map = {
        'ExpansionMLP': ExpansionMLP, 'BottleneckMLP': BottleneckMLP,
        'ResidualMLP': ResidualMLP, 'GatedMLP': GatedMLP,
        'StatefulGRUModule': StatefulGRUModule,
        'CrossAttentionModule': CrossAttentionModule,
        'StatefulAttentionModule': StatefulAttentionModule,
    }
    if module_name not in module_map:
        raise ValueError(f"Unknown module name: {module_name}.")
    
    module = module_map[module_name](hidden_size=hidden_size)
    return module.to(device=device, dtype=dtype)