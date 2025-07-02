# my_module.py

import torch.nn as nn

def get_my_module(hidden_size, device, dtype):
    """
    定义并返回您的可训练模块。
    您可以在这里自由地修改模块的架构，例如增加层数、改变激活函数、使用不同类型的层等。
    """
    print("Building 'my_module' from my_module.py")
    
    # 示例：一个简单的序贯模型
    # 您可以将其替换为任何更复杂的 nn.Module 子类
    module = nn.Sequential(
        nn.Linear(hidden_size, hidden_size * 2), # 尝试将维度扩大
        nn.ReLU(),
        nn.Dropout(0.1), # 加入 Dropout 防止过拟合
        nn.Linear(hidden_size * 2, hidden_size), # 再将维度映射回来
        nn.LayerNorm(hidden_size)
    ).to(device=device, dtype=dtype)
    
    return module