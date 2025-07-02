# inspect.py
import os
import torch
from model_wrapper import TrainableCoTGenerator # 从我们的 model.py 导入类
import config # <-- 导入配置

def inspect_and_generate(model_wrapper: TrainableCoTGenerator, question: str, num_loops: int, target_layer: int, top_k: int = 10, max_new_tokens: int = 50):
    """
    使用加载了权重的模型进行推理和分析。
    这个函数现在独立于模型类，更清晰。
    """
    print("\n==============================================")
    print("  Running Inference with Thought Inspection ")
    print("==============================================")
    
    # 获取模型和模块，并设置为评估模式
    model = model_wrapper.model
    my_module = model_wrapper.get_trainable_module()
    tokenizer = model_wrapper.tokenizer
    dtype = model_wrapper.dtype
    my_module.eval()
    
    with torch.no_grad():
        # --- 阶段一: 生成思考链并进行分析 ---
        question_ids = tokenizer(question, return_tensors="pt").input_ids.to(model.device)
        question_embeds = model.model.embed_tokens(question_ids)
        
        context_with_thoughts_embeds = question_embeds
        print(f"Question: \"{question}\"")
        print("--- Generating and Inspecting Thoughts ---")

        for i in range(num_loops):
            outputs = model(inputs_embeds=context_with_thoughts_embeds, output_hidden_states=True)
            last_token_hidden = outputs.hidden_states[target_layer][:, -1, :]
            processed_hidden = my_module(last_token_hidden.to(dtype))
            
            # --- 分析步骤 ---
            thought_logits = model.lm_head(processed_hidden)
            top_k_logits, top_k_indices = torch.topk(thought_logits, top_k, dim=-1)
            top_k_tokens = tokenizer.convert_ids_to_tokens(top_k_indices.squeeze().tolist())
            
            print(f"\n[Thought Step {i + 1}/{num_loops}]")
            for token, logit_val in zip(top_k_tokens, top_k_logits.squeeze().tolist()):
                print(f"  - {token:<15} (logit: {logit_val:.4f})")
            
            context_with_thoughts_embeds = torch.cat([context_with_thoughts_embeds, processed_hidden.unsqueeze(1)], dim=1)

        # --- 阶段二: 自回归生成答案 ---
        print("\n--- Generating Final Answer ---")
        context_len = context_with_thoughts_embeds.shape[1]
        dummy_attention_mask = torch.ones((1, context_len), dtype=torch.long, device=model.device)
        
        generated_ids = model.generate(
            inputs_embeds=context_with_thoughts_embeds,
            attention_mask=dummy_attention_mask,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=False
        )
        
        generated_answer = tokenizer.decode(generated_ids[0][context_len:], skip_special_tokens=True)
        print("\n==============================================")
        print("              Final Result")
        print("==============================================")
        print(f"Question: {question}")
        print(f"Generated Answer: {generated_answer}")

def main():
    # --- 从配置加载参数 ---
    NUM_COT_LOOPS = 5
    TARGET_LAYER = config.DEFAULT_TARGET_LAYER
    
    # 1. 初始化模型
    print("Initializing model...")
    model_wrapper = TrainableCoTGenerator()
    
    # 2. 加载训练好的模块权重
    print(f"Loading trained weights from: {config.TRAINED_WEIGHTS_PATH}")
    if not os.path.exists(config.TRAINED_WEIGHTS_PATH):
        print(f"Error: Trained weights not found at '{config.TRAINED_WEIGHTS_PATH}'")
        print("Please run 'python train.py' first to train and save the model.")
        return
        
    model_wrapper.load_trainable_module(config.TRAINED_WEIGHTS_PATH)
    
    # 3. 对新问题进行分析
    test_question = "Solve for x in 2x + 3 = 11."
    inspect_and_generate(
        model_wrapper,
        question=test_question,
        num_loops=NUM_COT_LOOPS,
        target_layer=TARGET_LAYER,
        top_k=30,
        max_new_tokens=100
    )

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        import traceback
        print(f"\nAn error occurred: {e}")
        traceback.print_exc()