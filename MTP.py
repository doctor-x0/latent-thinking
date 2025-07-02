import torch
import torch.nn as nn
import math

# -----------------------------------------------------------------------------
# 模块一：位置编码 (Positional Encoding)
# 这是一个标准的Transformer组件，用于给序列输入增加位置信息。
# -----------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        # 将pe注册为buffer，它不是模型参数，但应随模型移动(e.g., to(device))
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

# -----------------------------------------------------------------------------
# 模块二：核心MTP头 (MTPHeadTransformer)
# -----------------------------------------------------------------------------
class MTPHeadTransformer(nn.Module):
    def __init__(self, 
                 main_hidden_dim: int, 
                 vocab_size: int,
                 k: int,
                 mtp_hidden_dim: int = 768, # MTP内部维度，可以小于主模型维度
                 nhead: int = 8,             # 注意力头数
                 num_decoder_layers: int = 2,# 解码器层数
                 dropout: float = 0.1):
        super().__init__()
        
        self.k = k
        self.mtp_hidden_dim = mtp_hidden_dim
        
        # 1. 记忆投影层：将主模型的h_latent投影到MTP内部维度
        # 这是h_latent进入Transformer交叉注意力的入口
        self.memory_proj = nn.Linear(main_hidden_dim, mtp_hidden_dim)

        # 2. 词嵌入层：将在forward中从主模型传入，此处不定义
        
        # 3. 位置编码层
        self.pos_encoder = PositionalEncoding(mtp_hidden_dim, dropout)
        
        # 4. Transformer解码器层
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=mtp_hidden_dim,
            nhead=nhead,
            dim_feedforward=mtp_hidden_dim * 4, # 通用设置为d_model的4倍
            dropout=dropout,
            batch_first=True # 重要！使用batch_first以简化代码
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        # 5. 输出投影层：将解码结果映射回整个词表
        self.output_head = nn.Linear(mtp_hidden_dim, vocab_size)

    def forward(self, 
                h_latent: torch.Tensor, 
                main_embedding_table: nn.Embedding, 
                targets: torch.Tensor):
        """
        Args:
            h_latent (torch.Tensor): 主模型隐状态, shape [batch_size, main_hidden_dim]
            main_embedding_table (nn.Embedding): 主模型的词嵌入层，用于token lookup
            targets (torch.Tensor): 目标关键概念ID序列, shape [batch_size, k]
        
        Returns:
            torch.Tensor: 预测的logits, shape [batch_size, k, vocab_size]
        """
        batch_size = h_latent.size(0)

        # --- 步骤1: 准备记忆 (Memory) ---
        # h_latent作为交叉注意力的记忆源
        # shape: [batch_size, main_hidden_dim] -> [batch_size, mtp_hidden_dim] -> [batch_size, 1, mtp_hidden_dim]
        memory = self.memory_proj(h_latent).unsqueeze(1)

        # --- 步骤2: 准备目标序列 (Target Sequence) ---
        # 使用Teacher Forcing，目标序列是ground truth
        # 我们用targets来生成解码器的输入
        # shape: [batch_size, k] -> [batch_size, k, main_hidden_dim] (如果维度不匹配，需要投影)
        
        # 假设主模型的embedding维度与MTP维度一致
        if main_embedding_table.embedding_dim != self.mtp_hidden_dim:
            # 在实际应用中，如果维度不匹配，你需要一个投影层
            # 这里为了简化，我们假设它们是匹配的，或者主模型维度就是mtp_hidden_dim
            # e.g. self.embedding_proj = nn.Linear(main_embedding_table.embedding_dim, self.mtp_hidden_dim)
            # embedded_targets = self.embedding_proj(main_embedding_table(targets))
            raise ValueError("Embedding dimension mismatch. An embedding projection layer is needed.")
        
        embedded_targets = main_embedding_table(targets) * math.sqrt(self.mtp_hidden_dim)
        
        # 添加位置编码
        # 注意：PositionalEncoding期望 [seq_len, batch_size, dim]，所以需要转换维度
        embedded_targets = embedded_targets.permute(1, 0, 2) # [k, batch_size, dim]
        embedded_targets_pos = self.pos_encoder(embedded_targets)
        embedded_targets_pos = embedded_targets_pos.permute(1, 0, 2) # 转回 [batch_size, k, dim]

        # --- 步骤3: 创建掩码 (Masks) ---
        # 创建一个上三角矩阵来防止模型看到未来的token (Causal Mask)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(self.k).to(h_latent.device)
        
        # --- 步骤4: 通过Transformer解码器 ---
        # decoder_output shape: [batch_size, k, mtp_hidden_dim]
        decoder_output = self.transformer_decoder(
            tgt=embedded_targets_pos,
            memory=memory,
            tgt_mask=tgt_mask
        )
        
        # --- 步骤5: 生成最终输出 ---
        # logits shape: [batch_size, k, vocab_size]
        logits = self.output_head(decoder_output)
        
        return logits


# ==============================================================================
# 使用示例与测试
# ==============================================================================
if __name__ == '__main__':
    # --- 模型参数定义 ---
    BATCH_SIZE = 4
    MAIN_HIDDEN_DIM = 1536  # 根据您的要求
    VOCAB_SIZE = 32000
    NUM_CONCEPT_TOKENS = 15 # k值
    
    # --- 创建模型实例 ---
    # 这里我们让MTP内部维度等于主模型嵌入维度，以避免额外的投影
    mtp_head = MTPHeadTransformer(
        main_hidden_dim=MAIN_HIDDEN_DIM,
        vocab_size=VOCAB_SIZE,
        k=NUM_CONCEPT_TOKENS,
        mtp_hidden_dim=768, # 内部维度可以小一些以节省计算
        nhead=8,
        num_decoder_layers=2
    )
    print(f"MTP Head模型已创建:\n{mtp_head}")

    # --- 创建模拟输入数据 ---
    # 主模型在某个思考步骤的隐状态
    h_latent_dummy = torch.randn(BATCH_SIZE, MAIN_HIDDEN_DIM)
    
    # 目标关键概念序列 (整数ID)
    targets_dummy = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, NUM_CONCEPT_TOKENS))
    
    # 模拟主模型的词嵌入表
    # 为了让代码能跑通，我们假设主模型的embedding维度和MTP内部维度一致
    # 在实际应用中，您需要传入真实的embedding table
    # 如果维度不匹配，需要增加一个投影层
    main_embedding_table_dummy = nn.Embedding(VOCAB_SIZE, 768)
    
    # 增加一个从1536到768的投影层，使示例更真实
    embedding_proj_layer = nn.Linear(1536, 768)
    # 假设主模型嵌入维度是1536
    main_embedding_table_full_dim = nn.Embedding(VOCAB_SIZE, 1536)
    
    # 改造MTPHead以适应维度不匹配的情况，在__init__中添加：
    mtp_head.embedding_proj = nn.Linear(1536, 768)
    
    # 重写forward中的embedding部分，使其更健壮
    def forward_adapted(self, h_latent, main_embedding_table, targets):
        # ... (前面步骤不变) ...
        # 步骤2的修改版
        embedded_targets_full_dim = main_embedding_table(targets)
        embedded_targets = self.embedding_proj(embedded_targets_full_dim) * math.sqrt(self.mtp_hidden_dim)
        # ... (后面步骤不变) ...
        # (为了简洁，这里不重写整个类，只演示思路)
        print("\n注意: 实际使用中需要处理embedding维度不匹配的问题。")


    print(f"\n--- 运行前向传播测试 ---")
    print(f"输入 h_latent shape: {h_latent_dummy.shape}")
    print(f"输入 targets shape: {targets_dummy.shape}")

    # --- 执行前向传播 ---
    # 在这个测试中，我们直接将mtp_hidden_dim设为768，并使用一个假的768维的embedding table
    try:
        logits = mtp_head.forward(h_latent_dummy, main_embedding_table_dummy, targets_dummy)
        print(f"输出 logits shape: {logits.shape}")
        assert logits.shape == (BATCH_SIZE, NUM_CONCEPT_TOKENS, VOCAB_SIZE)
        print("测试通过！输出维度符合预期。")
    except Exception as e:
        print(f"测试失败: {e}")