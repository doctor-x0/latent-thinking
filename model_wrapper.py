# model_wrapper.py

import torch
import torch.nn as nn
import os
import gc
import config # <-- 导入配置以使用损失权重

from modelscope import AutoModelForCausalLM, AutoTokenizer
from filtering_module import get_my_module # <-- 新增：从我们独立的文件中导入
from filtering_modules import get_filtering_module, StatefulGRUModule, CrossAttentionModule, StatefulAttentionModule

# (环境设置代码与之前相同)
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    current_dir = os.getcwd()
os.environ['TRANSFORMERS_CACHE'] = os.path.join(current_dir, 'models')
os.environ['HF_HOME'] = os.path.join(current_dir, 'models')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class TrainableCoTGenerator(nn.Module):
    """
    模型包装器。
    整合了大型语言模型和从外部文件加载的可训练模块。
    """
    def __init__(self):
        super().__init__()
        # (加载 Qwen 模型部分与之前完全相同)
        MODEL_NAME = "qwen/Qwen2.5-Math-1.5B"
        CACHE_DIR = os.path.join(current_dir, 'models')
        print(f"Loading model: {MODEL_NAME} from cache dir: {CACHE_DIR}")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR, device_map='auto', trust_remote_code=True)
        
        # (冻结、设置pad_token等与之前完全相同)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.hidden_size = self.model.config.hidden_size
        self.dtype = next(self.model.parameters()).dtype
        self.hiddens_loss_fn = nn.MSELoss()
        
        # 通过工厂函数，根据config中的设置来构建模块
        print(f"Building filtering module of type: '{config.FILTERING_MODULE_NAME}'")
        self.my_module = get_filtering_module(
            module_name=config.FILTERING_MODULE_NAME,
            hidden_size=self.hidden_size,
            device=device,
            dtype=self.dtype
        )
        # ========================================================================

    # (其他所有方法: get_trainable_module, forward_for_loss, save/load_trainable_module 都保持不变)
    def get_trainable_module(self):
        return self.my_module

     # ========================================================================
    #  变化：修改 forward_for_loss 方法
    # ========================================================================
    # 您需要确保在 model_wrapper.py 的文件顶部已经导入了这些模块
# from filtering_modules import StatefulGRUModule, CrossAttentionModule, StatefulAttentionModule
# 并且导入了 config 和 torch

    def forward_for_loss(self, question: str, target_answer_text: str, target_hiddens_steps: list, num_loops: int, target_layer: int):
        """
        前向传播，计算并返回一个组合损失（答案损失 + 过程损失）。
        V2: 已整合对多种高级模块（有状态、有注意力）的支持。
        """
        # --- 阶段一: 生成思考链并计算"过程损失" ---
        # device = next(self.model.parameters()).device # 获取模型所在的设备
        question_ids = self.tokenizer(question, return_tensors="pt").input_ids.to(device)
        question_embeds = self.model.model.embed_tokens(question_ids)
        
        context_with_thoughts_embeds = question_embeds
        self.my_module.train()
        
        total_hiddens_loss = 0.0
        # 我们生成的思考步数，不能超过预计算的CoT步数
        effective_loops = min(num_loops, len(target_hiddens_steps))
        
        # 初始化模块的内部状态（只对有状态模块有效）
        module_state = None 
        
        # 判断模块类型
        is_stateful = isinstance(self.my_module, (StatefulGRUModule, StatefulAttentionModule))
        needs_context = isinstance(self.my_module, (CrossAttentionModule, StatefulAttentionModule))

        # --- 循环 1: 有“黄金hiddens”作为监督信号的步骤 ---
        for i in range(effective_loops):
            outputs = self.model(inputs_embeds=context_with_thoughts_embeds, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            last_token_hidden = hidden_states[target_layer][:, -1, :]
            
            # ========================================================================
            #  V2 变化：根据模块类型，使用不同的方式调用 forward 方法
            # ========================================================================
            if is_stateful and needs_context: # 对应 StatefulAttentionModule
                processed_hidden, module_state = self.my_module(last_token_hidden, module_state, context=question_embeds)
            elif is_stateful: # 对应 StatefulGRUModule
                processed_hidden, module_state = self.my_module(last_token_hidden, module_state)
            elif needs_context: # 对应 CrossAttentionModule
                processed_hidden, _ = self.my_module(last_token_hidden, context=question_embeds)
            else: # 对应所有无状态的MLP模块
                processed_hidden = self.my_module(last_token_hidden)
            # ========================================================================
            
            # h_target: 我们预计算的"黄金向量"
            target_hidden = target_hiddens_steps[i].to(device, dtype=self.dtype)

            # 计算当前步骤的过程损失
            step_hiddens_loss = self.hiddens_loss_fn(processed_hidden, target_hidden)
            total_hiddens_loss += step_hiddens_loss
            
            context_with_thoughts_embeds = torch.cat([context_with_thoughts_embeds, processed_hidden.unsqueeze(1)], dim=1)

        # --- 循环 2: 如果 num_loops 更大，继续“自由思考”的步骤 ---
        if num_loops > effective_loops:
            for _ in range(effective_loops, num_loops):
                outputs = self.model(inputs_embeds=context_with_thoughts_embeds, output_hidden_states=True)
                last_token_hidden = outputs.hidden_states[target_layer][:, -1, :]
                
                # ========================================================================
                #  V2 变化：在自由思考阶段，同样需要正确调用模块
                # ========================================================================
                if is_stateful and needs_context:
                    processed_hidden, module_state = self.my_module(last_token_hidden, module_state, context=question_embeds)
                elif is_stateful:
                    processed_hidden, module_state = self.my_module(last_token_hidden, module_state)
                elif needs_context:
                    processed_hidden, _ = self.my_module(last_token_hidden, context=question_embeds)
                else:
                    processed_hidden = self.my_module(last_token_hidden)
                # ========================================================================

                context_with_thoughts_embeds = torch.cat([context_with_thoughts_embeds, processed_hidden.unsqueeze(1)], dim=1)

        # --- 阶段二: 拟合目标答案并计算"答案损失" (逻辑不变) ---
        target_ids = self.tokenizer(target_answer_text, return_tensors="pt").input_ids.to(device)
        answer_embeds = self.model.model.embed_tokens(target_ids)
        full_sequence_embeds = torch.cat([context_with_thoughts_embeds, answer_embeds], dim=1)
        outputs = self.model(inputs_embeds=full_sequence_embeds)
        logits = outputs.logits
        context_len = context_with_thoughts_embeds.shape[1]
        answer_logits = logits[:, context_len-1:-1, :]
        
        answer_loss_fn = nn.CrossEntropyLoss()
        answer_loss = answer_loss_fn(answer_logits.reshape(-1, answer_logits.size(-1)), target_ids.reshape(-1))

        # --- 组合损失 (逻辑不变) ---
        avg_hiddens_loss = total_hiddens_loss / effective_loops if effective_loops > 0 else 0.0
        
        combined_loss = (config.LOSS_WEIGHT_ANSWER * answer_loss) + \
                        (config.LOSS_WEIGHT_HIDDENS * avg_hiddens_loss)
        
        # 返回三种损失值，方便打印和监控
        return combined_loss, answer_loss.item(), avg_hiddens_loss.item() if isinstance(avg_hiddens_loss, torch.Tensor) else avg_hiddens_loss
    
    def save_trainable_module(self, path: str):
        print(f"Saving trainable module weights to {path}")
        torch.save(self.my_module.state_dict(), path)
        print("Save complete.")

    def load_trainable_module(self, path: str):
        if not os.path.exists(path):
            print(f"Warning: Weight file not found at {path}. Skipping loading.")
            return
        print(f"Loading trainable module weights from {path}")
        self.my_module.load_state_dict(torch.load(path, map_location=device))
        print("Load complete.")