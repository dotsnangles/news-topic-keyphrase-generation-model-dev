import os
import re
import warnings
import torch
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
)
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_int8_training
from datasets import load_from_disk
import wandb
from dotenv import load_dotenv
from hydra import compose, initialize
from omegaconf import OmegaConf

warnings.filterwarnings("ignore")

os.environ["TOKENIZERS_PARALLELISM"] = "false"

NGPU = torch.cuda.device_count()
NCPU = os.cpu_count()

# Init Config
load_dotenv()
initialize(version_base=None, config_path="conf", job_name="train")
cfg = compose(config_name="train")

# Use LLM.int8() or Not
int8 = cfg.global_args.int8

# Paths and Names
PROJECT_NAME = cfg.path.PROJECT_NAME
RUN_ID = cfg.path.RUN_ID

TRAIN_DATA_PATH = cfg.path.TRAIN_DATA_PATH
EVAL_DATA_PATH = cfg.path.EVAL_DATA_PATH

MODEL_CHECKPOINT = cfg.path.MODEL_CHECKPOINT
if cfg.path.TOKENIZER_CHECKPOINT != False:
    TOKENIZER_CHECKPOINT = cfg.path.TOKENIZER_CHECKPOINT
else:
    TOKENIZER_CHECKPOINT = cfg.path.MODEL_CHECKPOINT
model_name = re.sub(r"[/-]", r"_", MODEL_CHECKPOINT).lower()

NOTEBOOK_NAME = cfg.path.NOTEBOOK_NAME

ROOT_PATH = "./"
SAVE_PATH = os.path.join(ROOT_PATH, ".log")

run_name = f"{model_name}_{RUN_ID}"
output_dir = os.path.join(SAVE_PATH, run_name)

os.makedirs(SAVE_PATH, exist_ok=True)
os.environ["WANDB_PROJECT"] = PROJECT_NAME
os.environ["WANDB_NOTEBOOK_NAME"] = NOTEBOOK_NAME
os.environ["WANDB_LOG_MODEL"] = "false"
os.environ["WANDB_WATCH"] = "all"

# Training Args
batch_size = cfg.global_args.batch_size
learning_rate = float(cfg.global_args.learning_rate)

training_args_prep = dict(
    **cfg.training_args,
    output_dir=output_dir,
    run_name=run_name,
    learning_rate=learning_rate * (batch_size * NGPU) / 8,
    logging_steps=int(500 / NGPU),
)

wandb.login(key=os.getenv("WANDB_API_KEY"))


def main():
    # Load Model & Tokenizer
    config = AutoConfig.from_pretrained(MODEL_CHECKPOINT)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_CHECKPOINT)

    architectures = config.architectures[0].lower()

    if "conditional" in architectures:
        training_args = Seq2SeqTrainingArguments(**training_args_prep)
        task_type = TaskType.SEQ_2_SEQ_LM
        if int8 == True:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                MODEL_CHECKPOINT,
                load_in_8bit=True,
                device_map="auto",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            )
            model = prepare_model_for_int8_training(model)
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                MODEL_CHECKPOINT,
            )
    elif "causal" in architectures:
        training_args = TrainingArguments(**training_args_prep)
        task_type = TaskType.CAUSAL_LM
        if int8 == True:
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_CHECKPOINT,
                load_in_8bit=True,
                device_map="auto",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            )
            model = prepare_model_for_int8_training(model)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_CHECKPOINT,
            )

    # Prepare Model to Use LoRA
    peft_config = LoraConfig(
        task_type=task_type,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
    )

    model = get_peft_model(model, peft_config)
    model.resize_token_embeddings(len(tokenizer))
    model.print_trainable_parameters()

    # Load Data
    train_dataset = load_from_disk(TRAIN_DATA_PATH)
    eval_dataset = load_from_disk(EVAL_DATA_PATH)

    # Train
    collator_args = dict(tokenizer=tokenizer, model=model, padding=True)
    trainer_args = dict(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=cfg.global_args.early_stopping_patience
            )
        ],
        # compute_metrics=compute_metrics,
    )

    if "conditional" in architectures:
        data_collator = DataCollatorForSeq2Seq(**collator_args)
        trainer = Seq2SeqTrainer(**trainer_args, data_collator=data_collator)
    elif "causal" in architectures:
        collator_args.pop("model")
        collator_args.pop("padding")
        collator_args["mlm"] = False
        data_collator = DataCollatorForLanguageModeling(**collator_args)
        trainer = Trainer(**trainer_args, data_collator=data_collator)

    # model.config.use_cache = (
    #     False  # silence the warnings. Please re-enable for inference!
    # )

    trainer.train()
    wandb.finish()

    # Save LoRA Adapters for Best CKPT and tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    config.save_pretrained(output_dir)


if __name__ == "__main__":
    main()
