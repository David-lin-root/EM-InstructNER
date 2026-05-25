import logging
import math
import os
import sys
import random
from dataclasses import dataclass, field
from itertools import chain
import deepspeed
from typing import Optional,List,Union
import datasets
import evaluate
import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model, get_peft_model_state_dict, prepare_model_for_int8_training, prepare_model_for_kbit_training, set_peft_model_state_dict
import transformers
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, get_last_checkpoint
from transformers import CONFIG_MAPPING, MODEL_FOR_CAUSAL_LM_MAPPING, AutoConfig, AutoModelForCausalLM, AutoTokenizer, TrainerCallback, TrainerState, \
TrainerControl, HfArgumentParser, Trainer, TrainingArguments, default_data_collator, BitsAndBytesConfig, is_torch_tpu_available, set_seed
from transformers.testing_utils import CaptureLogger
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
import time
import pdb

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")
logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

'''定义模型相关的参数类，使用dataclass装饰器自动生成初始化'''

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None)  # 基座模型路径
    
    model_type: Optional[str] = field(default=None)
    config_name: Optional[str] = field(default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"})
    tokenizer_name: Optional[str] = field(default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"})
    cache_dir: Optional[str] = field(default=None,metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},)
    model_revision: str = field(default="main")
    use_auth_token: bool = field(default=False)
    torch_dtype: Optional[str] = field(default=None)
    config_overrides: Optional[str] = field(default=None)
    lora_r: Optional[int] = field(default=16)  # LoRA中的秩参数
    lora_alpha: Optional[int] = field(default=32)  # LoRA中的alpha参数
    # 要替换为LoRA的目标模块名称列表
    target_modules: Optional[str] = field(default='q_proj,k_proj,v_proj,o_proj,down_proj,gate_proj,up_proj')
    use_fast_tokenizer: bool = field(default=False)  # 是否使用基于tokenizers库的快速分词器
    load_in_bits: Optional[int] = field(default=4)  # 加载模型时使用的位数

    # 在初始化后执行的额外检查
    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )  # 检查config_overrides不能与config_name或model_name_or_path同时使用
        if type(self.target_modules)==str:
            self.target_modules = self.target_modules.split(',')  # 如果target_modules是字符串，则将其按逗号分隔为列表
            
'''训练和评估的数据的参数'''

@dataclass
class DataTrainingArguments:
    train_on_inputs: bool = field(default=False, metadata={"help": "是否覆盖缓存的训练和评估数据集"})
    
    dataset_name: Optional[str] = field(default=None)
    dataset_config_name: Optional[str] = field(default=None)
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    
    train_files: Optional[List[str]]  = field(default=None)
    validation_files: Optional[List[str]]  = field(default=None)
    max_train_samples: Optional[int] = field(default=None,)  # 截断训练样本数量
    max_eval_samples: Optional[int] = field(default=None)
    block_size: Optional[int] = field(default=None)  # 训练时将数据集截断为此大小的块
    overwrite_cache: bool = field(default=False)  # # 是否覆盖缓存的训练和评估数据集
    validation_split_percentage: Optional[int] = field(default=5)  # # 如果没有验证集划分，使用训练集的一定比例作为验证集
    preprocessing_num_workers: Optional[int] = field(default=1)  # 用于预处理的进程数量
    keep_linebreaks: bool = field(default=True)  # # 使用TXT文件时是否保留换行符

    def __post_init__(self):
        if self.streaming:
            require_version("datasets>=2.0.0", "The streaming feature requires `datasets>=2.0.0`")  # 如果启用了流模式，检查datasets库的版本是否符合要求

        if self.dataset_name is None and self.train_files is None and self.validation_files is None:
            raise ValueError("Need either a dataset name or a training/validation file.")  # 如果dataset_name、train_files和validation_files都没有设置，抛出异常
        else:
            # 检查train_files和validation_files的扩展名是否是csv、json或txt
            if self.train_files is not None:
                extension = self.train_files[0].split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_files is not None:
                extension = self.validation_files[0].split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."

'''在训练过程中保存模型'''

class SavePeftModelCallback(TrainerCallback):
    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.is_world_process_zero:
            print('=====================================save call back=====================================')
            checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")
            kwargs["model"].save_pretrained(checkpoint_folder)  # 保存预训练模型到检查点文件夹
            pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
            if os.path.exists(pytorch_model_path):
                os.remove(pytorch_model_path)
            return control


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    send_example_telemetry("run_clm", model_args, data_args)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
            
    set_seed(training_args.seed)
    if True:
        data_files = {}
        dataset_args = {}
        data_files["train"] = data_args.train_files
        data_files["validation"] = data_args.validation_files
        extension = (
            data_args.train_files[0].split(".")[-1]
            if data_args.train_files is not None
            else data_args.validation_files.split(".")[-1]
        )
        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = data_args.keep_linebreaks
        raw_datasets = load_dataset(extension, data_files=data_files, cache_dir=os.path.join(training_args.output_dir,'dataset_cache'),
            use_auth_token=True if model_args.use_auth_token else None,
            **dataset_args,
        )

        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(extension, data_files=data_files, split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir, use_auth_token=True if model_args.use_auth_token else None,
                **dataset_args,
            )
            raw_datasets["train"] = load_dataset( extension, data_files=data_files, split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir, use_auth_token=True if model_args.use_auth_token else None,
                **dataset_args,
            )
    config_kwargs = {"cache_dir": model_args.cache_dir, "revision": model_args.model_revision, "use_auth_token": True if model_args.use_auth_token else None}
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")
    # 加载分词器
    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
        "padding_side":'left'
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    tokenizer.pad_token = tokenizer.eos_token
    logger.info(f"tokenizer.pad_token :{tokenizer.pad_token}")

    # 配置 LoRA 参数
    lora_config = LoraConfig(
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        target_modules =  model_args.target_modules,
        fan_in_fan_out = False,
        lora_dropout=0.05,
        inference_mode=False,
        bias="none",
        task_type="CAUSAL_LM",
    )
    # 配置 BitsAndBytes 参数
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    # 加载预训练的语言模型
    if model_args.model_name_or_path:
        torch_dtype = (
            model_args.torch_dtype
            if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
        )
       
        torch_dtype = torch.float16 #
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            torch_dtype=torch_dtype,
            load_in_8bit=True if model_args.load_in_bits==8 else False,
            quantization_config=bnb_config if model_args.load_in_bits==4 else None,
            # device_map  = 'auto'
            device_map={"": int(os.environ.get("LOCAL_RANK") or 0)}
        )
    else:
        model = AutoModelForCausalLM.from_config(config)
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")

    # 调整模型的嵌入层大小
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
    if model_args.load_in_bits==8:
        model = prepare_model_for_int8_training(model)
    elif model_args.load_in_bits==4:
        model = prepare_model_for_kbit_training(model)
    # 数据预处理：首先对所有文本进行分词
    if training_args.do_train:
        column_names = list(raw_datasets["train"].features)
    else:
        column_names = list(raw_datasets["validation"].features)
        
    train_on_inputs = True
    if len(column_names)==1:
        text_column_name = "text" if "text" in column_names else column_names[0]
    elif len(column_names)==2:
        input_column_name = 'input_text' if 'input_text' in column_names else column_names[0]
        target_column_name = 'target_text' if 'target_text' in column_names else column_names[0]
        train_on_inputs=False
    else:
        raise ValueError('输入文件列数不对')
    
    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

    def tokenize_function(examples):
        with CaptureLogger(tok_logger) as cl:
            output = tokenizer([ item for item in examples[text_column_name]],truncation=True,max_length=data_args.block_size,padding=False,return_tensors=None)
            output['labels'] = output['input_ids'].copy()
            
        return output

    def tokenize(prompt):
        result = tokenizer(prompt,truncation=True,max_length=data_args.block_size,padding=False,return_tensors=None)
        result["labels"] = result["input_ids"].copy()
        return result

    def generate_and_tokenize_prompt(data_point):
        input_text = data_point[input_column_name]
        target_text = data_point[target_column_name]
        full_prompt = input_text+target_text
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = input_text
            tokenized_user_prompt = tokenize(user_prompt)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])
            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ] 
        return tokenized_full_prompt
    # 在主进程中首次进行数据集映射分词
    with training_args.main_process_first(desc="dataset map tokenization"):
        if not data_args.streaming:
            tokenized_datasets = raw_datasets.map(
                tokenize_function if train_on_inputs==True else generate_and_tokenize_prompt,
                batched=True if train_on_inputs==True else False,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )
        else:
            tokenized_datasets = raw_datasets.map(
                tokenize_function if train_on_inputs==True else generate_and_tokenize_prompt,
                batched=True if train_on_inputs==True else False,
                remove_columns=column_names,
            )
    # 设置数据块大小
    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 4096:   #block_size > 2048
            block_size = 4096
    else:
        block_size = min(data_args.block_size, tokenizer.model_max_length)
    logger.info(f"block_size{block_size}")
    
    if training_args.do_train:
        if "train" not in tokenized_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = tokenized_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")
        train_dataset = train_dataset.shuffle(seed=training_args.seed)

    if training_args.do_eval:
        if "validation" not in tokenized_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = tokenized_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

        def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                # Depending on the model and config, logits may contain extra tensors,
                # like past_key_values, but logits always come first
                logits = logits[0]
            return logits.argmax(dim=-1)
    # ========================================================== 模型设置及训练验证 ==========================================================
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # 打印可训练的参数
    # 初始化 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics if training_args.do_eval and not is_torch_tpu_available()else None,
        callbacks=([SavePeftModelCallback] if isinstance(model, PeftModel) else None),
    )

    if training_args.do_train:
        checkpoint = None
        start_train = time.time()
        if training_args.resume_from_checkpoint is not None:
            resume_from_checkpoint = training_args.resume_from_checkpoint
            checkpoint_name = os.path.join(resume_from_checkpoint, "pytorch_model.bin")
            if not os.path.exists(checkpoint_name):
                checkpoint_name = os.path.join(resume_from_checkpoint, "adapter_model.bin")  # 仅适用于LoRA模型
                resume_from_checkpoint = (False)
            if os.path.exists(checkpoint_name):
                print(f"Restarting from {checkpoint_name}")
                adapters_weights = torch.load(checkpoint_name)
                set_peft_model_state_dict(model, adapters_weights)
            else:
                print(f"Checkpoint {checkpoint_name} not found")
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        
        if torch.__version__ >= "2" and sys.platform != "win32":
            model = torch.compile(model)  # 编译模型以提高效率
        
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # 保存模型和分词器
        metrics = train_result.metrics  # 获取训练指标
        max_train_samples = (data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset))
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        end_train = time.time()
        print('***********************耗时：{}***********************'.format((end_train - start_train)))

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])  # 计算困惑度
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        
if __name__ == "__main__":
    main()