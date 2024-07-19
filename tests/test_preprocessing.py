from pathlib import Path
import os

from questions.transcipt import TransciptDataset

def test_data_loading(
    fullfile: Path,
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
    dataset = TransciptDataset(fullfile)


if __name__ == "__main__":
    """
    Unit tests for modules and functions. 
    """
    filepath = Path().absolute().joinpath('data/transcripts')    
    test_data_loading(filepath)