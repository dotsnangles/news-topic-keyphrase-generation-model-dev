import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
# os.environ["CUDA_VISIBLE_DEVICES"]= "0"  # Set the GPUs to use

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
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_from_disk
import wandb

warnings.filterwarnings("ignore")


def main():
    NGPU = torch.cuda.device_count()
    NCPU = os.cpu_count()

    # Paths and Names
    PROJECT_NAME = "news-topic-keyphrase-generation-model-dev"
    RUN_ID = "v4_run_1"

    TRAIN_DATA_PATH = "data/model_dev/model_dev_v4_train.hf"
    EVAL_DATA_PATH = "data/model_dev/model_dev_v4_eval.hf"

    MODEL_CHECKPOINT = "paust/pko-t5-base"
    model_name = re.sub(r"[/-]", r"_", MODEL_CHECKPOINT).lower()

    NOTEBOOK_NAME = "./train_seq2seq_plm.ipynb"

    ROOT_PATH = "./"
    SAVE_PATH = os.path.join(ROOT_PATH, ".log")

    run_name = f"{model_name}_{RUN_ID}"
    output_dir = os.path.join(SAVE_PATH, run_name)

    os.makedirs(SAVE_PATH, exist_ok=True)
    os.environ["WANDB_PROJECT"] = PROJECT_NAME
    os.environ["WANDB_NOTEBOOK_NAME"] = NOTEBOOK_NAME
    os.environ["WANDB_LOG_MODEL"] = "false"
    os.environ["WANDB_WATCH"] = "all"

    wandb.login()

    # Training Args
    batch_size = 8

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        run_name=run_name,
        report_to="wandb",
        num_train_epochs=50,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=1,
        # ### AdaFactor
        # optim= 'adafactor',
        # learning_rate=3e-6 * (batch_size * NGPU) / 8,
        # lr_scheduler_type='linear', # 'linear', 'cosine', 'cosine_with_restarts', 'polynomial', 'constant', 'constant_with_warmup'
        # warmup_ratio=0,
        ### AdamW
        optim="adamw_torch",  # 'adamw_torch' or 'adamw_hf'
        learning_rate=3e-6
        * (batch_size * NGPU)
        / 8,  # 3e-6 * (per_device_train_batch_size * NGPU) / 8
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        weight_decay=0.01,
        lr_scheduler_type="linear",  # 'linear', 'cosine', 'cosine_with_restarts', 'polynomial', 'constant', 'constant_with_warmup'
        warmup_ratio=0,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        save_strategy="epoch",
        evaluation_strategy="epoch",
        logging_strategy="steps",
        logging_first_step=True,
        logging_steps=int(500 / NGPU),
        predict_with_generate=False,
        generation_max_length=64,
        # generation_num_beams=generation_num_beams,
        fp16=False,
    )

    # Load Model & Tokenizer
    config = AutoConfig.from_pretrained(MODEL_CHECKPOINT)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_CHECKPOINT, config=config)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

    # Prepare Model to Use LoRA
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Load Data
    train_dataset = load_from_disk(TRAIN_DATA_PATH)
    eval_dataset = load_from_disk(EVAL_DATA_PATH)
    tokenizer.decode(train_dataset["input_ids"][0])
    tokenizer.decode(train_dataset["labels"][0])

    # Train
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=model, padding=True
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        # compute_metrics=compute_metrics,
    )

    trainer.train()
    wandb.finish()

    # Save LoRA Adapters for Best CKPT
    model.save_pretrained(output_dir)

    # Delete Unnecessaries
    keep = [
        "added_tokens.json",
        "config.json",
        "pytorch_model.bin",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.txt" "adapter_config.json",
        "adapter_model.bin",
    ]

    ckpts = os.listdir(output_dir)
    for ckpt in ckpts:
        ckpt = os.path.join(output_dir, ckpt)
        for item in os.listdir(ckpt):
            if item not in keep:
                os.remove(os.path.join(ckpt, item))


if __name__ == "__main__":
    main()
