from pathlib import Path
import os

from questions.transcipt import TransciptDataset

def test_data_loading(
    input_filepath: Path,
     output_filepath: Path,
):
    """Test loading text files into dataset objects.

    This function tests that the text transcripts are properly
    loaded and converted into a huggingface dataset object before
    undergoing preprocessing.

    Args:
        fullfile: the path to the text transcript.

    Returns:
        dataset: huggingface dataset object for next tests.
    """
    dataset = TransciptDataset(input_filepath, output_filepath)


if __name__ == "__main__":
    """
    Unit tests for modules and functions. 
    """
    input_filepath = Path().absolute().joinpath('data/transcripts')    
    output_filepath = Path().absolute().joinpath('data/json_datasets')
    if not os.path.exists(output_filepath):
        os.mkdir(output_filepath)    
    test_data_loading(input_filepath, output_filepath)