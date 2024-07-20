from collections import Counter
from sklearn import linear_model
from pathlib import Path
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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


    # Named-Entity Count Data
    entity_category_mapping = {}
    q_ner_data = {'Question Named-Entity':[], 'Question Named-Entity Count':[], 'Question Named-Entity Category':[], 'Answer Length':[], 'Answer Sentiment':[]}
    a_ner_data = {'Answer Named-Entity':[], 'Answer Named-Entity Count':[], 'Answer Named-Entity Category':[], 'Answer Length':[], 'Answer Sentiment':[]}
    ner_comp_data = {'Named-Entity':[], 'Named-Entity Category':[], 'Question Named-Entity Count':[], 'Answer Named-Entity Count':[], 'Answer Length':[], 'Answer Sentiment':[]}
    for idx in range(n_entries):

        q_entity_category_mapping = {k : v for (k, v) in zip(data['q_entity_text'][idx], data['q_entity_types'][idx])}
        a_entity_category_mapping = {k : v for (k, v) in zip(data['a_entity_text'][idx], data['a_entity_types'][idx])}

        answer_length = len(word_tokenize(data['answers'][idx]))
        answer_sentiment = data['a_sentiment'][idx]
        q_ner_counts = Counter(data['q_entity_text'][idx])
        a_ner_counts = Counter(data['a_entity_text'][idx])

        for (question_ne, question_ne_count) in q_ner_counts.items():
            q_ner_data['Question Named-Entity'].append(question_ne)
            q_ner_data['Question Named-Entity Count'].append(question_ne_count)
            q_ner_data['Question Named-Entity Category'].append(q_entity_category_mapping[question_ne])
            q_ner_data['Answer Length'].append(answer_length)
            q_ner_data['Answer Sentiment'].append(answer_sentiment)

        for (answer_ne, answer_ne_count) in a_ner_counts.items():
            a_ner_data['Answer Named-Entity'].append(answer_ne)
            a_ner_data['Answer Named-Entity Count'].append(answer_ne_count)
            a_ner_data['Answer Length'].append(answer_length)
            a_ner_data['Answer Sentiment'].append(answer_sentiment)
            a_ner_data['Answer Named-Entity Category'].append(a_entity_category_mapping[answer_ne])

        for (question_ne, question_ne_count) in q_ner_counts.items():
            if question_ne in a_ner_counts.keys():
                answer_ne_count = a_ner_counts[question_ne]
                a_ner_counts.pop(question_ne)
            else:
                answer_ne_count = 0
            ner_comp_data['Named-Entity'].append(question_ne)
            ner_comp_data['Named-Entity Category'].append(q_entity_category_mapping[question_ne])
            ner_comp_data['Question Named-Entity Count'].append(question_ne_count)
            ner_comp_data['Answer Named-Entity Count'].append(answer_ne_count)
            ner_comp_data['Answer Length'].append(answer_length)
            ner_comp_data['Answer Sentiment'].append(answer_sentiment)

        for (answer_ne, answer_ne_count) in a_ner_counts.items():
            question_ne_count = 0
            ner_comp_data['Named-Entity'].append(answer_ne)
            ner_comp_data['Named-Entity Category'].append(a_entity_category_mapping[answer_ne])
            ner_comp_data['Question Named-Entity Count'].append(question_ne_count)
            ner_comp_data['Answer Named-Entity Count'].append(answer_ne_count)
            ner_comp_data['Answer Length'].append(answer_length)
            ner_comp_data['Answer Sentiment'].append(answer_sentiment)

    
    for (qa_str, data) in zip(["Question", "Answer"], [q_ner_data, a_ner_data]):
        box_df = pd.DataFrame(data)
        fig = px.box(box_df, y=f"{qa_str} Named-Entity", x=f"Answer Length", width=800, height=1600).update_yaxes(categoryorder='total ascending')
        fig.update_layout(title=dict(text=f"Interview Answer Length by {qa_str} Named-Entity Count", font=dict(size=16), automargin=True, yref='container', xref='paper', x=0.5, y=0.95))
        fig.write_image(f"results/answer_length_by_{qa_str}_ner.png")

    
    for (qa_str, data) in zip(["Question", "Answer"], [q_ner_data, a_ner_data]):
        box_df = pd.DataFrame(data)
        box_df_people = box_df.loc[box_df.index[box_df[f"{qa_str} Named-Entity Category"]=="PERSON"]]
        fig = px.box(box_df_people, y=f"{qa_str} Named-Entity", x=f"Answer Length", width=800, height=1600).update_yaxes(categoryorder='total ascending')
        fig.update_layout(title=dict(text=f"Interview Answer Length by {qa_str} Named-Entity Count", font=dict(size=16), automargin=True, yref='container', xref='paper', x=0.5, y=0.95))
        fig.write_image(f"results/answer_length_by_{qa_str}_ner_people.png")


    bubble_df = pd.DataFrame(ner_comp_data)
    bubble_df_people = bubble_df  #bubble_df.loc[bubble_df.index[bubble_df[f"Named-Entity Category"]=="PERSON"]]
    fig = px.scatter(
        bubble_df_people, 
        y="Answer Named-Entity Count", 
        x="Question Named-Entity Count",
        size="Answer Length",
        width=800, 
        height=1600,
        size_max=60,
        )
    fig.update_layout(title=dict(text=f"Interview Answer Named-Entity Count\nby Question Named-Entity Count", font=dict(size=16), automargin=True, yref='container', xref='paper', x=0.5, y=0.95))
    fig.write_html(f"results/bubble_ner_people.html")
    
    z_grid_avg, z_grid_min, z_grid_max, z_grid_count = meshgrid_stats(bubble_df_people["Question Named-Entity Count"], bubble_df_people["Answer Named-Entity Count"], bubble_df_people["Answer Length"])
    ne_grid = meshgrid_str(bubble_df_people["Question Named-Entity Count"], bubble_df_people["Answer Named-Entity Count"], bubble_df_people["Named-Entity"])
    customdata = np.dstack((z_grid_min, z_grid_max, z_grid_count, ne_grid))

    fig = make_subplots(1, 1, subplot_titles=['Interview Answer Word Count by (Q, A) Named-Entity Count Pairs'])
    fig.add_trace(go.Heatmap(
        z=z_grid_avg,
        customdata=customdata,
        hovertemplate='<b>Avg Answer Word Count:%{z:.1f}</b><br>Min Answer Word Count:%{customdata[0]:d}</b><br>Max Answer Word Count:%{customdata[1]:d}</b><br>Num (#Q-NE,#A-NE) Cases:%{customdata[2]:d}</b><br>Named Entities:%{customdata[3]}</b>',
         name=''),
        1, 1)
    fig.update_layout(xaxis={'title':'Interview Question Named-Entity Count'}, yaxis={'title':'Interview Answer Named-Entity Count'})
    fig.update_layout(title_text='Interivew Q&A Named-Entity Analysis: How Interview Questions Impact Answer Length and Topic')
    fig.write_html(f"results/heatmap_ner_people_names.html")


def meshgrid(x, y, z):
    x_vals, x_idx = np.unique(x, return_inverse=True)
    y_vals, y_idx = np.unique(y, return_inverse=True)
    z_grid = np.empty(x_vals.shape + y_vals.shape)
    z_grid.fill(0) # or whatever yor desired missing data flag is
    z_grid[x_idx, y_idx] = z
    return z_grid

def meshgrid_stats(x, y, z):
    x_vals, x_idx = np.unique(x, return_inverse=True)
    y_vals, y_idx = np.unique(y, return_inverse=True)
    
    z_grid = np.empty(x_vals.shape + y_vals.shape, dtype=float)
    z_grid_min = np.empty(x_vals.shape + y_vals.shape, dtype=float)
    z_grid_max = np.empty(x_vals.shape + y_vals.shape, dtype=float)
    z_grid_count = np.empty(x_vals.shape + y_vals.shape, dtype=int)
    
    z_grid.fill(0.0)
    z_grid_min.fill(0.0)
    z_grid_max.fill(0.0)
    z_grid_count.fill(0)
    
    for (xi, yi, idx) in zip(x_idx, y_idx, x.index):
        c = z_grid_count[xi, yi]
        z_grid[xi, yi] = (c * z_grid[xi, yi] + z.loc[idx]) / (c + 1)
        z_grid_min[xi, yi] = np.min([z_grid[xi, yi], z.loc[idx]])
        z_grid_max[xi, yi] = np.max([z_grid[xi, yi], z.loc[idx]])
        z_grid_count[xi, yi] += 1
    return z_grid, z_grid_min, z_grid_max, z_grid_count

def meshgrid_str(x, y, z):
    x_vals, x_idx = np.unique(x, return_inverse=True)
    y_vals, y_idx = np.unique(y, return_inverse=True)
    z_grid = np.empty(x_vals.shape + y_vals.shape, dtype='U200')
    for (xi, yi, idx) in zip(x_idx, y_idx, x.index):
        if z.loc[idx] not in z_grid[xi, yi]:
            z_grid[xi, yi] = z_grid[xi, yi] + z.loc[idx] + ', '
    for (xi, yi, idx) in zip(x_idx, y_idx, x.index):
        if z_grid[xi, yi][-2:] ==', ':
            z_grid[xi, yi] = z_grid[xi, yi][:-2]
    return z_grid