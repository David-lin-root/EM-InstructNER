CUDA_VISIBLE_DEVICES=1 python merge_peft_adapter.py \
    --adapter_model_name  /root/autodl-tmp/sft_ner/checkpoint-17 \
    --output_name  /root/autodl-tmp/llm_ner \
    --load8bit false \
    --tokenizer_fast false  
