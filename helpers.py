import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
import spacy
from collections import Counter
from string import punctuation


@st.cache
def load_data():
    """
    Loads a dataframe from memory that has been preprocessed such that each comment contains mention of only one of
    the member columns
    :return: a pandas dataframe
    """
    return pd.read_csv('all_pos_neg_better_targetting_members.csv', lineterminator='\n',
                       parse_dates=['Updated At'],
                       date_parser=lambda x: pd.to_datetime(x),
                       index_col='Updated At')

    return all


@st.cache
def get_by_sentiment(df, val):
    return df.query(f'sentiment_category == {val}')


@st.cache
def get_count_of_comments(df, pattern):
    """
    Because each row mentions exactly one member, we can just sum the boolean columns
    :param df: dataframe containing pattern boolean columns corresponding to whether the comment mentions one in pattern
    :param pattern: a list of boolean columns
    :return: sum of frequencies of truths in boolean columns in dataframe form
    """
    return df[pattern].sum().sort_values(ascending=False).to_frame()


def plot_total_comments(df, pattern):
    count = get_count_of_comments(df, pattern)
    comments = px.bar(x=count.index,
                      y=count,
                      color_discrete_sequence=['pink'],
                      title="Comments per Member or YG"
                      )
    comments.update_xaxes(title="Member or YG")
    comments.update_yaxes(title="Number of Comments")
    comments.update_layout(plot_bgcolor='rgb(0, 0, 0)',
                           xaxis_showgrid=False, yaxis_showgrid=False)
    return comments


def plot_pos_to_neg(x_vals, pos_vals, neg_vals):
    """
    Helper function to plot two dataframes against each other
    :param x_vals:   x-axis values, discrete and categorical, for each member
    :param pos_vals: count of positive comments for each x_val
    :param neg_vals: count of negative comments for each x_val
    :return: a plotly figure
    """
    fig = go.Figure(data=[
        go.Bar(name="Positive Comments",
               x=x_vals,
               y=pos_vals,
               marker_color='pink'
               ),
        go.Bar(name="Negative Comments",
               x=x_vals,
               y=neg_vals,
               marker_color='black'
               )
    ])

    fig.update_layout(
        plot_bgcolor='rgb(255, 255, 255)', xaxis_showgrid=False, yaxis_showgrid=False,
    )
    return fig


@st.cache
def pos_to_neg(df, target_pattern, view_option='Ratio'):
    positives = get_by_sentiment(df, 1)
    negatives = get_by_sentiment(df, -1)

    pos_vals = positives[target_pattern].sum().sort_values(ascending=False)

    # use pos_index to re-order negatives
    pos_index = pos_vals.index
    neg_vals = negatives[target_pattern].sum()[pos_index]

    if view_option == 'As is':
        sum = 1
    else:
        sum = (positives[target_pattern].sum() + negatives[target_pattern].sum())[pos_index]

    return plot_pos_to_neg(pos_index,
                           (pos_vals / sum).tolist(),
                           (neg_vals / sum).tolist()
                           )
