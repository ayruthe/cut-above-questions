from datasets import load_dataset
from pathlib import Path
import time
import os

from questions.generative import train_llm_to_generate_questions


def test_llm_generation_training(
    json_file: Path,
    result_path: Path,
):
    """Test preprocessing data with NLP tasks.

    Preprocess dataset with tokenization, stopwords, lemmatization,
    chunking, and other NLP tasks to prepare it for downstream.

    Args:
        json_file: the path to the json dataset file.
    """
    train_llm_to_generate_questions(json_file, result_path)
    

if __name__ == "__main__":
    """
    Tests for modules and functions. 
    """
    filepath = Path().absolute().joinpath('data/json_datasets')
    result_path = Path().absolute().joinpath(f'results/training-{str(int(time.time()))}')
    datafile = filepath.joinpath('training_data.json')    
    metafile = filepath.joinpath('transcript_metadata.json')    
    test_llm_generation_training(datafile, result_path)
