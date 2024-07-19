import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import spacy
from pathlib import Path

from questions.files import load_json, save_json


def nlp_preproc(file: Path) -> None:
    """Natural language processing (NLP) for transcript data.

    This function serves as a basic pipeline for tokenization, 
    part-of-speech (POS) tagging, lemmatization, and name-entity 
    recognition (NER) built with SpaCy library. 
    
    Args:
        file: the json file path to transcript data to preprocess. 
    """
    nltk.downloader.download('vader_lexicon')
    json_data = load_json(file) 
    json_data = remove_newlines(json_data)
    json_data = named_entity_proc(json_data)
    json_data = sentiment_analysis(json_data)
    save_json(file, json_data)

    
def sentiment_analysis(json_data: dict) -> dict:
    """Preprocessing step to label sentiment of sentences.

    Args:
        json_data: dictionary transcript data to process.
    """
    n_qa = len(json_data['questions'])
    q_sentiment = []
    a_sentiment = []
    nlp = spacy.load("en_core_web_sm")
    for qa_str in ['questions', 'answers']:
        for idx in range(n_qa):
            text = json_data[qa_str][idx]
            sid = SentimentIntensityAnalyzer()
            composite_sentiment = sid.polarity_scores(text)['compound']

            if qa_str == 'questions':
                q_sentiment.append(composite_sentiment)
            elif qa_str == 'answers':
                a_sentiment.append(composite_sentiment)

    json_data['q_sentiment'] = q_sentiment
    json_data['a_sentiment'] = a_sentiment
    return json_data


def named_entity_proc(json_data: dict) -> dict:
    """Preprocessing step to label named entities.

    Args:
        json_data: dictionary transcript data to process.
    """
    n_qa = len(json_data['questions'])
    q_entity_types = []
    a_entity_types = []
    q_entity_text = []
    a_entity_text = []
    nlp = spacy.load("en_core_web_sm")
    for qa_str in ['questions', 'answers']:
        for idx in range(n_qa):
            text = json_data[qa_str][idx]
            doc = nlp(text)
            ent_types = [x.label_ for x in doc.ents]
            ent_text = [x.text for x in doc.ents]

            if qa_str == 'questions':
                q_entity_types.append(ent_types)
                q_entity_text.append(ent_text)

            elif qa_str == 'answers':
                a_entity_types.append(ent_types)
                a_entity_text.append(ent_text)

    json_data['q_entity_types'] = q_entity_types
    json_data['q_entity_text'] = q_entity_text
    json_data['a_entity_types'] = a_entity_types
    json_data['a_entity_text'] = a_entity_text

    return json_data
 

def remove_newlines(json_data: dict) -> dict:
    """Preprocessing step to remove newlines in sentences.

    Args:
        json_data: dictionary transcript data to process.
    """
    for idx in range(len(json_data['questions'])):
        json_data['questions'][idx] = json_data['questions'][idx].replace('\n', '')
        json_data['answers'][idx] = json_data['answers'][idx].replace('\n', '')
    return json_data



