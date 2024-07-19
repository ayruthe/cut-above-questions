from datasets import load_dataset
from pathlib import Path
import os

from questions.transcipt import TransciptDataset


def test_transcript_parsing(
    input_filepath: Path,
    data_savefile: Path, 
    metadata_savefile: Path,
):
    """Test loading text files into dataset objects.

    This function tests that the text transcripts are properly
    loaded and converted into a huggingface dataset object before
    undergoing preprocessing.

    Args:
        fullfile: the path to the text transcript.
    """
    TransciptDataset(input_filepath, data_savefile, metadata_savefile)
    assert data_savefile.is_file()
    assert metadata_savefile.is_file()


def test_preprocessing(
    json_file: Path,
):
    """Test preprocessing data with NLP tasks.

    Preprocess dataset with tokenization, stopwords, lemmatization,
    chunking, and other NLP tasks to prepare it for downstream.

    Args:
        json_file: the path to the json dataset file.
    """
    test_dataset = load_dataset("json", data_files=str(json_file))
    import pdb
    pdb.set_trace()


if __name__ == "__main__":
    """
    Unit tests for modules and functions. 
    """
    # 1. Load, parse, and save json for transcript text files.
    input_filepath = Path().absolute().joinpath('data/transcripts')    
    output_filepath = Path().absolute().joinpath('data/json_datasets')
    if not os.path.exists(output_filepath):
        os.mkdir(output_filepath)
    data_savefile = output_filepath.joinpath('transcript_data.json')    
    metadata_savefile = output_filepath.joinpath('transcript_metadata.json')    
    test_transcript_parsing(input_filepath, data_savefile, metadata_savefile)
    
    # 2. Preprocess text data in json dataset file.
    # test_preprocessing(savefile)
    