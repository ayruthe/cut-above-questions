from collections import Counter
from sklearn import linear_model
from pathlib import Path
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
import plotly.express as px
import pandas as pd

from questions.files import load_json, save_json


def linear_regression(file: Path):
    """Evaluate transcript data with linear models.

    Args:
        file: path to json dictionary storing the transcript data

    """
    data = load_json(file)
    n_entries = len(data['questions'])

    # Sentiment Correlation
    plot_data = {'Question Sentiment': data['q_sentiment'], 'Answer Sentiment': data['a_sentiment']}
    fig = px.scatter(plot_data, x='Question Sentiment', y='Answer Sentiment', trendline="ols", trendline_color_override = 'black')
    a = px.get_trendline_results(fig).px_fit_results.iloc[0].rsquared
    fig.update_layout(title=dict(text=f"Question Sentiment vs Answer Sentiment (R^2={a:.3f})", font=dict(size=18), automargin=True, yref='container', xref='paper', x=0.5, y=0.95))
    fig.write_image("fig1.png")


    # NER Count Box Plots
    box_data = {'Question NER':[], 'Question NER Count':[], 'Answer Length':[], 'Answer Sentiment':[]}
    for idx in range(n_entries):

        answer_length = len(word_tokenize(data['answers'][idx]))
        answer_sentiment = data['a_sentiment'][idx]
        ner_counts = Counter(data['q_entity_text'][idx])
        for (key, value) in ner_counts.items():
            box_data['Question NER'].append(key)
            box_data['Question NER Count'].append(value)
            box_data['Answer Length'].append(answer_length)
            box_data['Answer Sentiment'].append(answer_sentiment)

    box_df = pd.DataFrame(box_data)
    fig = px.box(box_df, y="Question NER", x="Answer Length", width=800, height=1600).update_yaxes(categoryorder='total ascending')
    fig.update_layout(title=dict(text=f"Answer Length by Question Named-Entity", font=dict(size=18), automargin=True, yref='container', xref='paper', x=0.5, y=0.95))
    fig.write_image("fig2.png")

    import pdb
    pdb.set_trace()