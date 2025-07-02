# config.py (V3)

import os
import torch

# --- 基础目录配置 ---
BASE_DIR = os.getcwd()

# 目录：存放原始HF数据集缓存
RAW_DATASETS_CACHE_DIR = "/root/.cache/huggingface/datasets"

# 目录：存放我们脚本预处理好的数据
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'processed_data') # 文件夹名可以取得更通用

# 新增：目录：存放训练好的模型权重/检查点
CHECKPOINTS_DIR = os.path.join(BASE_DIR, 'checkpoints')

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 文件名与路径定义 ---

# 预处理数据的文件名
TRAIN_ANSWERS_FILENAME = 'gsm8k_train_final_answers.json'
TRAIN_HIDDENS_FILENAME_TEMPLATE = 'gsm8k_train_cot_hiddens_layer_{}.pt'
TEST_ANSWERS_FILENAME = 'gsm8k_test_final_answers.json'
TEST_HIDDENS_FILENAME_TEMPLATE = 'gsm8k_test_cot_hiddens_layer_{}.pt'

# 模型权重的文件名和完整路径
TRAINED_WEIGHTS_FILENAME = "my_module_gsm8k_trained.pth"
TRAINED_WEIGHTS_PATH = os.path.join(CHECKPOINTS_DIR, TRAINED_WEIGHTS_FILENAME) # <-- 路径已更新


# --- 默认超参数 ---
DEFAULT_TARGET_LAYER = 25

# ========================================================================
#  新增：多任务学习损失权重
# ========================================================================
# α: 最终答案损失的权重
LOSS_WEIGHT_ANSWER = 1.0
# β: 隐藏状态过程损失的权重
LOSS_WEIGHT_HIDDENS = 0.5 # 初始可以设小一点，再慢慢调整
# ========================================================================

# ========================================================================
#  新增：VQ-VAE 模型配置
# ========================================================================
# 训练好的 VQ-VAE 权重保存路径
VQ_VAE_CHECKPOINT_PATH = os.path.join(CHECKPOINTS_DIR, "vq_vae_gsm8k.pth")

# VQ-VAE 超参数
VQ_VAE_NUM_EMBEDDINGS = 512  # Codebook中"概念向量"的数量 (K)
VQ_VAE_EMBEDDING_DIM = 64    # 每个"概念向量"的维度 (D)，需要小于hidden_size
VQ_VAE_COMMITMENT_COST = 0.25 # VQ-VAE的commitment loss权重 (β)

# VQ-VAE 架构选择
VQ_VAE_ARCHITECTURE = 'ResidualVQ_VAE'  # 可选: 'VQ_VAE', 'ResidualVQ_VAE'
# ======================================================================== 

FILTERING_MODULE_NAME = 'StatefulAttentionModule' 
