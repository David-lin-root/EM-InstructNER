# import torch
# from transformers import GPT2LMHeadModel, GPT2Tokenizer, LlamaForCausalLM, LlamaTokenizer

# def compute_perplexity(human_summary, llama_summary, model_name='gpt2-medium'):
#     # 加载预训练的 GPT-2 模型和分词器
#     tokenizer = LlamaTokenizer.from_pretrained(model_name)
#     model = LlamaForCausalLM.from_pretrained(model_name)

#     # 将人类摘要和 LLM 生成的摘要编码成输入序列
#     inputs = tokenizer.encode(human_summary, add_special_tokens=True, return_tensors="pt")
#     labels = tokenizer.encode(llama_summary, add_special_tokens=True, return_tensors="pt")

#     # 计算生成摘要的困惑度
#     with torch.no_grad():
#         outputs = model(inputs, labels=labels)
#         logits = outputs.logits
#         loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
#         perplexity = torch.exp(loss)

#     return perplexity.item()

# # 示例用法
# human_summary = "This is a human-generated summary."
# llama_summary = "This is an automatically generated summary by a large language model."

# perplexity = compute_perplexity(human_summary, llama_summary, model_name='/mnt/nvme0n1/chendong/aiwork/code/SelfCritical/checkpoints/FinLLM7B')
# print("Perplexity of the llama summary:", perplexity)
