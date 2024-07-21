from pathlib import Path

from questions.extraction import interview_named_entity_analysis


def test_extraction(
    json_file: Path,
):
    """Test preprocessing data with NLP tasks.

    Preprocess dataset with tokenization, stopwords, lemmatization,
    chunking, and other NLP tasks to prepare it for downstream.

    Args:
        json_file: the path to the json dataset file.
    """
    interview_named_entity_analysis(json_file)
    

if __name__ == "__main__":
    """
    Tests for modules and functions. 
    """ 
    filepath = Path().absolute().joinpath('data/json_datasets')
    datafile = filepath.joinpath('transcript_data.json')    
    metafile = filepath.joinpath('transcript_metadata.json')    
    test_extraction(datafile)
    