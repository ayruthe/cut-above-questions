from pathlib import Path

from questions.generative import InterviewQuestionLLM


def test_llm_generation_training(
    json_file: Path,
    result_path: Path,
    lora_rank: int,
    dataset_subsample_rate: int,
    model_name: str,
):
    """Test preprocessing data with NLP tasks.

    Preprocess dataset with tokenization, stopwords, lemmatization,
    chunking, and other NLP tasks to prepare it for downstream.

    Args:
        json_file: the path to the json dataset file.
    """
    llm = InterviewQuestionLLM(json_file, result_path, lora_rank, dataset_subsample_rate, model_name)
    llm.zeroshot_test()
    llm.low_rank_adaptation_training()
    assert result_path.joinpath('peft-checkpoint-local').joinpath('adapter_model.safetensors').is_file()

    
def test_llm_evaluation(
    json_file: Path,
    result_path: Path,
    lora_rank: int,
    dataset_subsample_rate: int,
    model_name: str,
):
    """Test preprocessing data with NLP tasks.

    Preprocess dataset with tokenization, stopwords, lemmatization,
    chunking, and other NLP tasks to prepare it for downstream.

    Args:
        json_file: the path to the json dataset file.
    """
    model_path = result_path.joinpath('peft-checkpoint-local')
    llm = InterviewQuestionLLM(json_file, result_path, lora_rank, dataset_subsample_rate, model_name)
    llm.evaluate_llm(json_file, model_path)

if __name__ == "__main__":
    """
    Tests for modules and functions. 
    """
    lora_rank = 4
    dataset_subsample_rate = 200
    model_name = 'google/flan-t5-base'

    filepath = Path().absolute().joinpath('data/json_datasets')
    result_path = Path().absolute().joinpath(f'results/training-lora-rank-{lora_rank}-subsample-{dataset_subsample_rate}')
    datafile = filepath.joinpath('training_data.json')    
    metafile = filepath.joinpath('transcript_metadata.json')    

    test_llm_generation_training(datafile, result_path, lora_rank, dataset_subsample_rate, model_name)
    test_llm_evaluation(datafile, result_path, lora_rank, dataset_subsample_rate, model_name)
