from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig, TrainingArguments, Trainer
import torch
import time
import evaluate
import pandas as pd
import numpy as np
from pathlib import Path
from datasets import load_dataset
import datasets
from peft import LoraConfig, get_peft_model, TaskType
from peft import PeftModel, PeftConfig


def train_llm_to_generate_questions(datafile: Path, result_path: Path):
    """Description.
    
    Summary.
    
    Args:
        - 
        
    Returns:
    """
    # Load Data and Foundation Model
    dataset = load_dataset("json", data_files=str(datafile))['train'].train_test_split(test_size=0.1)
    model_name='google/flan-t5-base'
    original_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # The dataset actually contains 3 diff splits: train, validation, test.
    # The tokenize_function code is handling all data across all splits in batches.
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    # tokenized_datasets = tokenized_datasets.remove_columns(['id', 'topic', 'dialogue', 'summary',])
    print(f"Shapes of the datasets:")
    print(f"Training: {tokenized_datasets['train'].shape}")
    print(f"Test: {tokenized_datasets['test'].shape}")
    print(tokenized_datasets)


    # Zero-shot Inference as Baseline Performance
    for index in [11, 127]:
        prev_answer = dataset['train'][index]['prev_answer']
        next_question = dataset['train'][index]['next_question']

        prompt = f"""
        Ask an inquisitive interview question in response to the following dialogue.

        {prev_answer}

        Question:
        """
        inputs = tokenizer(prompt, return_tensors='pt')
        output = tokenizer.decode(
            original_model.generate(
                inputs["input_ids"], 
                max_new_tokens=200,
            )[0], 
            skip_special_tokens=True
        )

        dash_line = '-'.join('' for x in range(100))
        print(dash_line)
        print(f'INPUT PROMPT:\n{prompt}')
        print(dash_line)
        print(f'BASELINE HUMAN SUMMARY:\n{next_question}\n')
        print(dash_line)
        print(f'MODEL GENERATION - ZERO SHOT:\n{output}')

    # Low-Rank Adaptation (LoRA) Finetuning
    lora_config = LoraConfig(
        r=4, # Rank
        lora_alpha=32,
        target_modules=["q", "v"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM # FLAN-T5
    )
    peft_model = get_peft_model(original_model, lora_config)
    print(print_number_of_trainable_model_parameters(peft_model))

    peft_training_args = TrainingArguments(
        output_dir=result_path,
        auto_find_batch_size=True,
        learning_rate=1e-3, 
        num_train_epochs=1,
        logging_steps=1,
        max_steps=1    
    )
        
    peft_trainer = Trainer(
        model=peft_model,
        args=peft_training_args,
        train_dataset=tokenized_datasets["train"],
    )

    peft_trainer.train()
    peft_model_path=result_path.joinpath("peft-checkpoint-local")
    peft_trainer.model.save_pretrained(peft_model_path)
    tokenizer.save_pretrained(peft_model_path)

    peft_model_base = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base", torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    peft_model = PeftModel.from_pretrained(peft_model_base, 
                                        peft_model_path, 
                                        torch_dtype=torch.bfloat16,
                                        is_trainable=False)
    
    print(print_number_of_trainable_model_parameters(peft_model))

    # Evaluate PEFT Model
    



def tokenize_function(example):
    model_name='google/flan-t5-base'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    start_prompt = 'Ask an inquisitive interview question in response to the following dialogue.\n\n'
    end_prompt = '\n\Question: '
    prompt = [start_prompt + dialogue + end_prompt for dialogue in example["prev_answer"]]
    example['input_ids'] = tokenizer(prompt, padding="max_length", truncation=True, return_tensors="pt").input_ids
    example['labels'] = tokenizer(example["next_question"], padding="max_length", truncation=True, return_tensors="pt").input_ids
    return example



def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    return f"trainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"
