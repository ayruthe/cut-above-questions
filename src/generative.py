from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig, TrainingArguments, Trainer
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from peft import PeftModel
from datasets import Dataset
import evaluate
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from questions.files import load_json


class InterviewQuestionLLM():
    def __init__(
            self, 
            datafile: Path, 
            result_path: Path,
            lora_rank: int, 
            dataset_subsample_rate: int,
            model_name: str = 'google/flan-t5-base'
        ):
        """LLM for generating interview questions in response to previous answers.

        This class trains a PEFT adapter for an LLM with Interview Q&A data using the
        low-rank adaptation (LoRA) method. Zeroshot inference testing generates and
        saves generated questions along with the sentiment and rouge scores in 
        reference to the actual next question. 
    
        Args:
            datafile: Q&A json dataset file
            result_path: file to save the model checkpoints to
            lora_rank: the rank of the LoRA model
            dataset_subsample_rate: factor for subsampling the dataset
            model_name: the foundation model name as a string

        """
        self.datafile = datafile
        self.result_path = result_path
        self.lora_rank = lora_rank
        self.model_name = model_name
        self.model_checkpoint_path = result_path.joinpath("peft-checkpoint-local")

        data = load_json(str(datafile))
        df = pd.DataFrame(data)
        self.dataset = Dataset.from_pandas(df).train_test_split(test_size=0.1)
        self.model_name=model_name
        self.original_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenized_datasets = self.dataset.map(tokenize_function, batched=True)
        tokenized_datasets = tokenized_datasets.filter(lambda example, index: index % dataset_subsample_rate == 0, with_indices=True)
        self.tokenized_datasets = tokenized_datasets


    def low_rank_adaptation_training(self):
        """Low-rank adaptation (LoRA) PEFT model finetuning procedure."""
        lora_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=32,
            target_modules=["q", "v"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM # FLAN-T5
        )
        peft_base_model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name, torch_dtype=torch.bfloat16)
        peft_model = get_peft_model(peft_base_model, lora_config)

        peft_training_args = TrainingArguments(
            output_dir=self.result_path,
            learning_rate=1e-1, 
            num_train_epochs=5,
            auto_find_batch_size=True,
        )
            
        peft_trainer = Trainer(
            model=peft_model,
            args=peft_training_args,
            train_dataset=self.tokenized_datasets["train"],
        )

        peft_trainer.train()
        peft_trainer.model.save_pretrained(self.model_checkpoint_path)
        self.tokenizer.save_pretrained(self.model_checkpoint_path)


    def zeroshot_test(self):
        """Zeroshot inference of current model."""
        for index in [11, 127]:
            prev_answer = self.dataset['train'][index]['prev_answer']
            next_question = self.dataset['train'][index]['next_question']

            prompt = f"""
            Ask an inquisitive interview question in response to the following dialogue.

            {prev_answer}

            Question:
            """
            inputs = self.tokenizer(prompt, return_tensors='pt')
            output = self.tokenizer.decode(
                self.original_model.generate(
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


    def evaluate_llm(self, datafile: Path, model_path: Path, model_name: str = "google/flan-t5-base"):
        """Evaluate the LLM via sentiment and rouge scores."""
        original_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        peft_base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        peft_model = PeftModel.from_pretrained(peft_base_model, model_path, torch_dtype=torch.bfloat16, is_trainable=False)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Zero-shot Inference as Baseline Performance
        n_entries = len(self.dataset['train']['prev_answer'])
        rouge = evaluate.load('rouge')
        for index in range(n_entries):

            prev_answer = self.dataset['train'][int(index)]['prev_answer']
            prev_answer_sentiment = self.dataset['train'][int(index)]['prev_answer_sentiment']
            next_question = self.dataset['train'][int(index)]['next_question']

            eval_dict = {
                'Previous Interview Answer (Truth)':[],
                'Next Interview Question (Truth)':[],
                'Vanilla FLAN-T5 Question (Generated)':[],
                'LoRA-Finetned FLAN-T5 Question (Generated)':[],
                'LoRA-Finetned FLAN-T5 with Sentiment Question (Generated)':[]
            }
            
            if prev_answer_sentiment > 0:
                pos_neg = 'positive'
            else:
                pos_neg = 'negative'
            prompt = f"""
            Ask an inquisitive interview question in response to the following dialogue.

            {prev_answer}

            Question:
            """
            sentiment_prompt = f"""
            Ask an inquisitive interview question in response to the following dialogue. Respond with {pos_neg} sentiment.

            {prev_answer}

            Question:
            """
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids
            input_ids_sentiment = tokenizer(sentiment_prompt, return_tensors="pt").input_ids

            original_model_outputs = original_model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=200, num_beams=1))
            original_model_text_output = tokenizer.decode(original_model_outputs[0], skip_special_tokens=True)

            peft_model_outputs = peft_model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=200, num_beams=1))
            peft_model_text_output = tokenizer.decode(peft_model_outputs[0], skip_special_tokens=True)

            peft_sentiment_model_outputs = peft_model.generate(input_ids=input_ids_sentiment, generation_config=GenerationConfig(max_new_tokens=200, num_beams=1))
            peft_sentiment_model_text_output = tokenizer.decode(peft_sentiment_model_outputs[0], skip_special_tokens=True)

            dash_line = '-'.join('' for x in range(100))
            print(dash_line)
            print(f'PREVIOUS ANSWER:\n{prev_answer}')
            print(dash_line)
            print(f'BASELINE HUMAN SUMMARY:\n{next_question}')
            print(dash_line)
            print(f'ORIGINAL MODEL:\n{original_model_text_output}')
            print(dash_line)
            print(f'PEFT MODEL: {peft_model_text_output}')
            print(dash_line)
            print(f'PEFT MODEL W/ SENTIMENT: {peft_sentiment_model_text_output}')

            eval_dict['Previous Interview Answer (Truth)'].append(prev_answer)
            eval_dict['Next Interview Question (Truth)'].append(next_question)
            eval_dict['Vanilla FLAN-T5 Question (Generated)'].append(original_model_text_output)
            eval_dict['LoRA-Finetned FLAN-T5 Question (Generated)'].append(peft_model_text_output)
            eval_dict['LoRA-Finetned FLAN-T5 with Sentiment Question (Generated)'].append(peft_sentiment_model_text_output)

            vanilla_scores = rouge_scores(rouge, original_model_text_output, next_question)
            peft_scores = rouge_scores(rouge, peft_model_text_output, next_question)
            peft_w_sent_scores = rouge_scores(rouge, peft_sentiment_model_text_output, next_question)

            prev_answ_sentiment = calc_sentiment(prev_answer)
            next_ques_sentiment = calc_sentiment(next_question)
            vanilla_sentiment = calc_sentiment(original_model_text_output)
            peft_sentiment = calc_sentiment(peft_model_text_output)
            peft_w_sent_sentiment = calc_sentiment(peft_sentiment_model_text_output)

            eval_dict['Previous Interview Answer (Truth)'].append(prev_answ_sentiment)
            eval_dict['Next Interview Question (Truth)'].append(next_ques_sentiment)
            eval_dict['Vanilla FLAN-T5 Question (Generated)'].append(vanilla_sentiment)
            eval_dict['LoRA-Finetned FLAN-T5 Question (Generated)'].append(peft_sentiment)
            eval_dict['LoRA-Finetned FLAN-T5 with Sentiment Question (Generated)'].append(peft_w_sent_sentiment)

            for i in range(2):
                eval_dict['Previous Interview Answer (Truth)'].append(" -- ")
                eval_dict['Next Interview Question (Truth)'].append(" -- ")
            
            for score_name in ['rouge1','rouge2']:
                eval_dict['Vanilla FLAN-T5 Question (Generated)'].append(vanilla_scores[score_name])
                eval_dict['LoRA-Finetned FLAN-T5 Question (Generated)'].append(peft_scores[score_name])
                eval_dict['LoRA-Finetned FLAN-T5 with Sentiment Question (Generated)'].append(peft_w_sent_scores[score_name])


            eval_df = pd.DataFrame(eval_dict)
            eval_df.set_index(pd.Index(["Text", "Sentiment", "Rouge-1", "Rouge-2"]), inplace=True)
            eval_md = eval_df.to_markdown()


            with open(self.result_path.joinpath(f'text_summary_{index}.md'), 'w') as f:
                f.write(eval_md)


def calc_sentiment(text: str):
    """Calculate a text string's sentiment value."""
    sid = SentimentIntensityAnalyzer()    
    return sid.polarity_scores(text)['compound']


def rouge_scores(rouge, generated_text:str, human_text: str):
    """Calculate the rouge score between two text strings."""
    scores = rouge.compute(
        predictions=generated_text[0:len(human_text)],
        references=human_text[0:len(generated_text)],
        use_aggregator=True,
        use_stemmer=True,
    )
    return scores


def tokenize_function(example):
    """Tokenize a dataset."""
    model_name='google/flan-t5-base'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    start_prompt = 'Ask an inquisitive interview question in response to the following dialogue.\n\n'
    end_prompt = '\n\Question: '
    prompt = [start_prompt + dialogue + end_prompt for dialogue in example["prev_answer"]]
    example['input_ids'] = tokenizer(prompt, padding="max_length", truncation=True, return_tensors="pt").input_ids
    example['labels'] = tokenizer(example["next_question"], padding="max_length", truncation=True, return_tensors="pt").input_ids
    return example