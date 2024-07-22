import os
from pathlib import Path

from questions.extraction import interview_named_entity_analysis
from questions.transcipt import TransciptDataset
from questions.language import nlp_preproc
from questions.generative import InterviewQuestionLLM


if __name__ == "__main__":
    """Main script to run project pipeline."""
    # 1. Load, parse, and save json for transcript text files.
    input_filepath = Path().absolute().joinpath('data/transcripts')    
    output_filepath = Path().absolute().joinpath('data/json_datasets')
    if not os.path.exists(output_filepath):
        os.mkdir(output_filepath)
    data_savefile = output_filepath.joinpath('transcript_data.json')        
    metadata_savefile = output_filepath.joinpath('transcript_metadata.json')   
    
    # 2. Preprocess
    # TransciptDataset(input_filepath, data_savefile, metadata_savefile)
    # nlp_preproc(data_savefile)
    
    # # 3. NER and Sentiment Analysis
    # interview_named_entity_analysis(data_savefile)

    # 4. Train LLM
    lora_rank = 32
    dataset_subsample_rate = 1
    model_name = 'google/flan-t5-base'
    result_path = Path().absolute().joinpath(f'results/training-lora-rank-{lora_rank}-subsample-{dataset_subsample_rate}')
    model_path = result_path.joinpath('peft-checkpoint-local')
    train_file = output_filepath.joinpath('training_data.json')
    if not model_path.joinpath('adapter_model.bin').is_file():
        llm = InterviewQuestionLLM(train_file, result_path, lora_rank, dataset_subsample_rate, model_name)
        llm.zeroshot_test()
        llm.low_rank_adaptation_training()
    else:
        print('Loading Model')

    # 5. Evaluate LLM
    llm = InterviewQuestionLLM(train_file, result_path, lora_rank, dataset_subsample_rate, model_name)
    llm.evaluate_llm(train_file, model_path)
