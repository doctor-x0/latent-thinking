import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
import tqdm  # For a nice progress bar

# ==============================================================================
# 0. 配置参数 (Hyperparameters)
# 将所有参数集中管理，方便修改
# ==============================================================================
class TrainingConfig:
    # 模型维度
    VOCAB_SIZE = 32000
    MAIN_HIDDEN_DIM = 1536
    MTP_HIDDEN_DIM = 768
    
    # MTP Transformer参数
    MTP_NHEAD = 8
    MTP_NUM_DECODER_LAYERS = 2
    
    # 训练参数
    BATCH_SIZE = 8
    NUM_EPOCHS = 3
    LEARNING_RATE = 1e-4
    NUM_TRAINING_SAMPLES = 1000 # 模拟数据集大小
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 任务相关参数
    NUM_LATENT_STEPS = 4  # 假设模型进行4步隐性思考
    K_CONCEPT_TOKENS = 15 # MTP预测的关键概念数量
    PAD_TOKEN_ID = -100   # 用于忽略损失计算的padding token ID
    
    # 损失权重
    ALPHA_MTP_LOSS = 0.3  # MTP辅助损失的权重

# ==============================================================================
# 1. 模型定义
# ==============================================================================

# --- 1a. PositionalEncoding (与之前相同) ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 50):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(1, 0, 2) # [seq, batch, dim]
        x = x + self.pe[:x.size(0)]
        x = self.dropout(x)
        return x.permute(1, 0, 2) # [batch, seq, dim]

# --- 1b. MTPHeadTransformer (与之前相同, 稍作清理) ---
class MTPHeadTransformer(nn.Module):
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config
        self.memory_proj = nn.Linear(config.MAIN_HIDDEN_DIM, config.MTP_HIDDEN_DIM)
        self.embedding_proj = nn.Linear(config.MAIN_HIDDEN_DIM, config.MTP_HIDDEN_DIM) # 假设主模型嵌入维度=隐状态维度
        self.pos_encoder = PositionalEncoding(config.MTP_HIDDEN_DIM, max_len=config.K_CONCEPT_TOKENS)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.MTP_HIDDEN_DIM, nhead=config.MTP_NHEAD,
            dim_feedforward=config.MTP_HIDDEN_DIM * 4, batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.MTP_NUM_DECODER_LAYERS)
        self.output_head = nn.Linear(config.MTP_HIDDEN_DIM, config.VOCAB_SIZE)

    def forward(self, h_latent, main_embedding_table, targets):
        memory = self.memory_proj(h_latent).unsqueeze(1)
        embedded_targets = self.embedding_proj(main_embedding_table(targets)) * math.sqrt(self.config.MTP_HIDDEN_DIM)
        embedded_targets_pos = self.pos_encoder(embedded_targets)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(self.config.K_CONCEPT_TOKENS).to(self.config.DEVICE)
        decoder_output = self.transformer_decoder(tgt=embedded_targets_pos, memory=memory, tgt_mask=tgt_mask)
        logits = self.output_head(decoder_output)
        return logits

# --- 1c. 模拟的主模型 ---
# 这是一个占位符，代表您实际使用的、进行隐性推理的LLM。
# 它的任务是接收问题，并返回最终答案的logits和中间思考步骤的隐状态。
class DummyMainModel(nn.Module):
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config
        # 模拟的词嵌入表
        self.embedding = nn.Embedding(config.VOCAB_SIZE, config.MAIN_HIDDEN_DIM)
        # 模拟的模型主体 (例如一些线性层)
        self.body = nn.Sequential(nn.Linear(config.MAIN_HIDDEN_DIM, config.MAIN_HIDDEN_DIM), nn.ReLU())
        # 模拟的最终答案输出头
        self.final_answer_head = nn.Linear(config.MAIN_HIDDEN_DIM, config.VOCAB_SIZE)

    def get_embedding_table(self):
        return self.embedding

    def forward(self, question_ids):
        # 这是一个模拟过程，实际应是复杂的自回归和隐性思考循环
        batch_size = question_ids.size(0)
        
        # 1. 模拟隐性思考步骤，产生N个隐状态
        latent_hiddens = []
        # 简单地为每个步骤生成一个随机的隐状态
        for _ in range(self.config.NUM_LATENT_STEPS):
            h = torch.randn(batch_size, self.config.MAIN_HIDDEN_DIM).to(self.config.DEVICE)
            latent_hiddens.append(self.body(h)) # 通过一些层来使其可训练
        
        # 2. 模拟根据最后一个隐状态生成最终答案
        last_hidden = latent_hiddens[-1]
        final_answer_logits = self.final_answer_head(last_hidden) # shape: [batch_size, vocab_size]
        
        return final_answer_logits, latent_hiddens

# ==============================================================================
# 2. 数据加载器
# 创建一个模拟数据集，返回符合我们训练需求的批次数据。
# ==============================================================================
class DummyLatentReasoningDataset(Dataset):
    def __init__(self, config: TrainingConfig):
        self.config = config

    def __len__(self):
        return self.config.NUM_TRAINING_SAMPLES

    def __getitem__(self, idx):
        # 生成随机的ID序列作为数据
        question = torch.randint(0, self.config.VOCAB_SIZE, (10,)) # 假设问题长度为10
        final_answer_label = torch.randint(0, self.config.VOCAB_SIZE, (1,))[0]
        
        mtp_target_labels = []
        for _ in range(self.config.NUM_LATENT_STEPS):
            # 模拟真实情况，部分token是padding
            num_real_tokens = torch.randint(5, self.config.K_CONCEPT_TOKENS + 1, (1,)).item()
            real_tokens = torch.randint(0, self.config.VOCAB_SIZE, (num_real_tokens,))
            pad_tokens = torch.full((self.config.K_CONCEPT_TOKENS - num_real_tokens,), self.config.PAD_TOKEN_ID)
            mtp_target_labels.append(torch.cat([real_tokens, pad_tokens]))

        return {
            "question": question,
            "final_answer_label": final_answer_label,
            "mtp_target_labels": mtp_target_labels
        }

# ==============================================================================
# 3. 训练主函数
# ==============================================================================
def train():
    # --- 初始化 ---
    print("--- 开始训练流程 ---")
    config = TrainingConfig()
    print(f"使用设备: {config.DEVICE}")

    # --- 实例化模型 ---
    main_model = DummyMainModel(config).to(config.DEVICE)
    mtp_head = MTPHeadTransformer(config).to(config.DEVICE)
    print("模型已创建并移至设备。")

    # --- 准备数据 ---
    dataset = DummyLatentReasoningDataset(config)
    # 自定义collate_fn来处理mtp_target_labels列表
    def collate_fn(batch):
        questions = torch.stack([item['question'] for item in batch])
        final_answers = torch.stack([item['final_answer_label'] for item in batch])
        # 将mtp_targets的列表正确堆叠
        mtp_targets = [torch.stack([item['mtp_target_labels'][i] for item in batch]) for i in range(config.NUM_LATENT_STEPS)]
        return {'question': questions, 'final_answer_label': final_answers, 'mtp_target_labels': mtp_targets}

    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    
    # --- 定义优化器和损失函数 ---
    # 将两个模型的参数合并到一个优化器中
    optimizer = optim.Adam(
        list(main_model.parameters()) + list(mtp_head.parameters()), 
        lr=config.LEARNING_RATE
    )
    main_loss_fn = nn.CrossEntropyLoss()
    mtp_loss_fn = nn.CrossEntropyLoss(ignore_index=config.PAD_TOKEN_ID) # 忽略padding

    # --- 训练循环 ---
    for epoch in range(config.NUM_EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{config.NUM_EPOCHS} ---")
        main_model.train()
        mtp_head.train()
        
        total_loss_epoch = 0

        for batch in tqdm.tqdm(dataloader, desc=f"Epoch {epoch+1} Training"):
            # 将数据移动到设备
            question = batch['question'].to(config.DEVICE)
            final_answer_label = batch['final_answer_label'].to(config.DEVICE)
            mtp_target_labels = [t.to(config.DEVICE) for t in batch['mtp_target_labels']]

            # 1. 前向传播
            optimizer.zero_grad()
            final_answer_logits, latent_hiddens = main_model(question)
            
            # 2. 计算主任务损失
            loss_main = main_loss_fn(final_answer_logits, final_answer_label)
            
            # 3. 计算所有步骤的MTP辅助损失
            total_mtp_loss = 0.0
            main_embedding_table = main_model.get_embedding_table()
            
            for i in range(config.NUM_LATENT_STEPS):
                h_latent_i = latent_hiddens[i]
                mtp_targets_i = mtp_target_labels[i]
                
                mtp_logits_i = mtp_head(h_latent_i, main_embedding_table, mtp_targets_i)
                
                # 对齐形状并计算损失 [B, K, V] -> [B*K, V]
                mtp_loss = mtp_loss_fn(
                    mtp_logits_i.view(-1, config.VOCAB_SIZE),
                    mtp_targets_i.view(-1)
                )
                total_mtp_loss += mtp_loss
            
            # 4. 合并损失
            average_mtp_loss = total_mtp_loss / config.NUM_LATENT_STEPS
            total_loss = loss_main + config.ALPHA_MTP_LOSS * average_mtp_loss
            
            # 5. 反向传播和优化
            total_loss.backward()
            optimizer.step()
            
            total_loss_epoch += total_loss.item()
            
        avg_epoch_loss = total_loss_epoch / len(dataloader)
        print(f"Epoch {epoch + 1} 完成 | 平均总损失: {avg_epoch_loss:.4f}")

    print("\n--- 训练完成 ---")


if __name__ == '__main__':
    train()