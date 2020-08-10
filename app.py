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

from helpers import *

st.title("Sentiment Analysis of Comments on BlackPink's latest video \"How You Like That\"")
st.write('<style>div.Widget.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
all_solo = load_data()
pattern = ['jennie', 'jisoo', 'rose', 'lisa', 'yg']

st.markdown(
    "Between June 26th and July 20th, I managed to collect more than **a million** comments from BlackPink's latest  "
    "video, \"How You Like That\".")
st.write("After preparing the data, I ultimately ended up with 200,000 comments to work with.")

st.write("With these comments, I wanted to answer several questions:")
st.markdown(">*Which member is most talked about?  \n"
            "What is the general sentiment towards each member?  \n"
            "What are some of the things they're receiving praise for?  \n"
            "Some of the things they've been criticized for?*")
st.markdown("## **Which member is the most talked about?** ")
st.plotly_chart(plot_total_comments(all_solo, pattern))
st.markdown("## **What is the general sentiment towards each member?**")

total_view_option = st.radio(
    "See share of positive and negative comments as is or as ratio",
    ('As is', 'Ratio')
)
st.plotly_chart(pos_to_neg(all_solo, pattern, total_view_option))

st.markdown(
    "Lisa has almost the same number of comments as the sum of Jisoo and Jennie! I was very surprised to find out  "
    "that Rose would receive the least number of comments. Then again, I admit that I don't really know too much  "
    "About each member's popularity. "
    "In terms of raw numbers, Lisa also has the greatest number of negative comments. If you view the ratio option,  "
    "however, you'll see that YG has the greatest proportion of negative to positive comments. (As expected?)"
)
st.markdown(
    "Note that the number of positive and negative comments do not add up to the total number of comments  "
    "for each member. This is because social media texts are inherently noisy, and although the library we used for  "
    "sentiment analysis did a great job, it wasn't able to predict the sentiment of every word. Since the library  "
    "handles these out-of-vocabulary words by treating them as neutral words, it's likely that including neutral  "
    "comments might misrepresent the actual number of positive, neutral, and negative comments."
)
st.write("Thus, for all graphs, the underlying data consists of the positive and negative comments only.")

st.markdown("## **What about sentiment over time?**")


@st.cache
def get_sentiment_isolated(option='Lisa', target_sentiment=1):
    """
    Query for the desired sentiment, and then select comments that mention only one member.
    From this, select the column corresponding to option

    :param option: target member
    :param target_sentiment: desired sentiment
    :return: returns a dataframe, replacing boolean columns with 0 or 1 to signify whether the row has given sentiment
    """
    if target_sentiment == 1:
        label = "positive"
    elif target_sentiment == -1:
        label = "negative"
    else:
        label = "neutral"

    sentiment = get_by_sentiment(all_solo, target_sentiment)  # either all positive or negative or neutral

    return sentiment.mask(sentiment[pattern] == 1,
                          sentiment['sentiment_category'] * target_sentiment,
                          axis=0)[option.lower()] \
        .astype('float64').to_frame().rename({option.lower(): f'{label} comments for {option.lower()}'}, axis=1)


@st.cache
def resample_by(pos, neg, time='10min'):
    return pos.resample(time).sum(), neg.resample(time).sum()


def plot_pos_vs_neg_over_time(member_option='Jennie',
                              view_option='As is',
                              time_range=None,
                              time_interval='1h'):
    time_intervals = {
        '10 min': '10min',
        '1 hr': '1h',
        '6 hr': '6h',
        '12 hr': '12h',
        '1 day': '1d',
        '1 week': '1w'
    }
    pos_subset = get_sentiment_isolated(member_option, target_sentiment=1)
    neg_subset = get_sentiment_isolated(member_option, target_sentiment=-1)

    # resample first to standardized datetimeIndex
    pos_subset, neg_subset = resample_by(pos_subset, neg_subset, time_intervals.get(time_interval))
    joined = pd.concat([neg_subset, pos_subset], axis=1)

    if time_range:
        # constraining joined to the given time_range
        joined = joined.loc[joined.index[joined.index.get_loc(time_range[0], method='nearest')]: \
                            joined.index[joined.index.get_loc(time_range[1], method='nearest')]
                 ]

    if view_option == 'Ratio':
        sum = joined.sum(axis=1)
    else:
        sum = 1

    joined[f'positive comments for {member_option.lower()}'] = joined[
                                                                   f'positive comments for {member_option.lower()}'] / sum
    joined[f'negative comments for {member_option.lower()}'] = joined[
                                                                   f'negative comments for {member_option.lower()}'] / sum

    fig = px.bar(joined,
                 color_discrete_sequence=['black', 'pink'],
                 title=f'Positive to Negative Sentiment for {member_option} over Time',
                 height=600,
                 width=1000,
                 )

    fig.update_xaxes(title="Time",
                     )

    fig.update_yaxes(title="Number of Comments")
    fig.update_layout(hovermode="x",
                      plot_bgcolor='rgb(255, 255, 255)', xaxis_showgrid=False, yaxis_showgrid=False,
                      )
    return fig


member_time_option = st.radio(
    "Choose a member, or the company:",
    ('Jennie', 'Jisoo', 'Lisa', 'Rose', 'YG'),
    key='member_time_option'
)

view_option = st.radio(
    "Choose whether you want to see positive and negative comments as is or as ratio",
    ('As is', 'Ratio')
)

select_time_interval = st.radio(
    "Select a time interval:",
    ('10 min', '1 hr', '6 hr', '12 hr', '1 day', '1 week')
)


def generate_time_range_slider(time_interval):
    if time_interval == '10 min':
        value = (datetime(2020, 6, 25, 00, 00), datetime(2020, 6, 28, 00, 00))
    elif time_interval in ['1 hr', '6 hr', '12 hr']:
        value = (datetime(2020, 6, 25, 00, 00), datetime(2020, 6, 30, 00, 00))
    elif time_interval == '1 day':
        value = (datetime(2020, 6, 25, 00, 00), datetime(2020, 7, 6, 00, 00))
    else:
        value = (datetime(2020, 6, 25, 00, 00), datetime(2020, 7, 21, 00, 00))

    return st.slider(
        label="Time Range",
        min_value=datetime(2020, 6, 30, 00, 00),
        value=value,
        max_value=datetime(2020, 7, 21, 00, 00),
        step=timedelta(hours=6),
        format="MM/DD/YY - hh:mm a"
    )


time_range = generate_time_range_slider(select_time_interval)
fig = plot_pos_vs_neg_over_time(member_time_option, view_option, time_range, select_time_interval)
st.plotly_chart(fig)
st.markdown("Note that the distribution of comments also follows the power law; the majority of the comments can be  "
            "found in the first three days to week since the video has been posted.")

st.markdown("## **A Closer Look**")


@st.cache
def get_ngrams(text, n=1, stop_words=None):
    """

    :param text: iterable of strings
    :param n:    number of words for n-gram
    :param stop_words: stop words to remove, if any
    :return:     n-gram, as specified by n
    """
    if stop_words:
        vectorizer = CountVectorizer(ngram_range=(n, n), analyzer='word', stop_words=stop_words)
    else:
        vectorizer = CountVectorizer(ngram_range=(n, n), analyzer='word')

    sparse = vectorizer.fit_transform(text)
    frequencies = np.sum(sparse, axis=0).T
    return pd.DataFrame(frequencies, index=vectorizer.get_feature_names(), columns=['frequency'])


@st.cache(allow_output_mutation=True)
def vectorize(member='jennie', type='tf'):
    try:
        just_nouns = all_solo[(all_solo[member] == True) & \
                              (all_solo['text_length'] > 10) & \
                              (all_solo['source_lang'] == 'en')]['just_nouns'].fillna('')
    except:
        raise ValueError("Could not produce dataset")

    if type == 'tf':
        tf_vectorizer = CountVectorizer()
    else:
        tf_vectorizer = TfidfVectorizer()
    tf = tf_vectorizer.fit_transform(
        just_nouns
    )

    return tf, tf_vectorizer


def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        st.write(message)
    st.write()


@st.cache
def fit_lda(n_components, tf):
    lda = LatentDirichletAllocation(n_components=n_components,
                                    learning_method='online',
                                    learning_offset=50.,
                                    random_state=0)
    lda.fit(tf)
    return lda


@st.cache
def fit_nmf(n_components, tfidf):
    nmf = NMF(n_components=n_components, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd', max_iter=500).fit(tfidf)
    return nmf


member_topic_option = st.radio(
    "Choose a member, or the company:",
    ('Jennie', 'Jisoo', 'Lisa', 'Rose', 'YG'),
    key='member_topic_option'
)


def get_topics_for_member(member, type='tfidf'):
    # remember to lower
    try:
        vectorized, vectorizer = vectorize(member.lower(), type)
    except:
        st.write("Unsuccessful in finding topics :( ")
    n_components = 8
    n_top_words = 8
    nmf = fit_nmf(n_components, vectorized)
    print_top_words(nmf, vectorizer.get_feature_names(), n_top_words)


if st.checkbox(f'Show me the machine generated topics for {member_topic_option}'):
    get_topics_for_member(member_topic_option)

topic_of_interest = st.text_input("Your topic(s) of interest: \
(you can query multiple topics! Just each topic by a comma and space!)", "🔥, hair, 爱")


def pos_to_neg_topic_for_member(df, topic, member):
    positives = get_by_sentiment(df, 1)
    negatives = get_by_sentiment(df, -1)

    search = [member[0].lower()]
    pos_vals = positives[search].sum()
    neg_vals = negatives[search].sum()
    sum = pos_vals + neg_vals

    fig = plot_pos_to_neg(search,
                          (pos_vals / sum).tolist(),
                          (neg_vals / sum).tolist(),
                          )
    fig.update_layout(
        title=f"Sentiment for All Comments Related to '{topic}' for {member[0]}"
    )

    return fig


@st.cache
def query_comments(df, topic, member, text_length=0):
    try:
        queried = df[
            (df['Original Comment'].str.contains("|".join(topic.split(", ")), case=False)) & \
            (df[member.lower()] == True) & \
            (df['source_lang'] == 'en') & \
            (df['text_length'] > text_length)

            ]
        return queried
    except:
        return pd.DataFrame()


def sample_and_plot_topic_for_member(topic, member, text_length=1):
    data = query_comments(df=all_solo, topic=topic, member=member, text_length=text_length)
    if data.empty:
        st.write(f"Couldn't find anything on *{topic}* for *{member}* :(")
    else:
        st.markdown(f"Found {data.shape[0]} comments regarding **{topic}** for **{member}**. Displaying a few now:")
        st.table(data.sample(3)[['Original Comment', 'sentiment_category']])
        st.plotly_chart(pos_to_neg_topic_for_member(data, topic, [member]))


sample_and_plot_topic_for_member(topic_of_interest, member_topic_option)


@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def load_nlp():
    nlp = spacy.load('en_core_web_md')
    return nlp


@st.cache
def top_sentence(text, limit):
    # create spacy nlp doc out of given text
    pos_tag = ['PROPN', 'ADJ', 'NOUN', 'VERB']
    nlp = load_nlp()
    doc = nlp(text.lower())

    # create list of all words
    keyword = [token.lemma_ for token in doc
               if token.pos_ in pos_tag
               and token.text not in nlp.Defaults.stop_words
               and token.text not in punctuation]

    # generate term-frequency counts
    freq_word = Counter(keyword)
    max_freq = freq_word.most_common(1)[0][1]

    # normalize term frequencies by max_freq
    for w in freq_word:
        freq_word[w] = (freq_word[w] / max_freq)

    # build idf
    num_sentences = 0
    idf = {}
    for sent in doc.sents:
        seen = set()
        num_sentences += 1
        for word in sent:
            # only add words once, even if they appear multiple times in sentence
            if word.lemma_ not in seen:
                if word.lemma_ not in idf:
                    idf[word.lemma_] = 1
                else:
                    idf[word.lemma_] += 1
                seen.add(word.lemma_)

    # combine the normalize(idf) and tfidf calculation step
    tfidf = {}
    for w in freq_word:
        tfidf[w] = freq_word[w] * (np.log((num_sentences + 1) / (idf[w] + 1)) + 1)

    sent_strength = {}
    for sent in doc.sents:
        for word in sent:
            if word.lemma_ in tfidf.keys():
                if sent in sent_strength.keys():
                    sent_strength[sent] += tfidf[word.lemma_]
                else:
                    sent_strength[sent] = tfidf[word.lemma_]

    summary = []

    sorted_x = sorted(sent_strength.items(), key=lambda kv: kv[1], reverse=True)

    i = 0
    while i < len(sorted_x) and i < limit:
        summary.append(" ".join(str(sorted_x[i][0]).capitalize().split()))
        i += 1

    return summary


def get_top_sentences(df, topic, member, text_length=15):
    data = query_comments(df, topic, member, text_length)
    if data.empty:
        st.write(f"Couldn't generate summary on *{topic} for *{member}* :(")
    else:
        sentences = pd.DataFrame(top_sentence(". ".join(data['Original Comment'].tolist()), 5),
                                 columns=[f"5 Sentence Summary on {topic} for {member}"]
                                 )
        st.table(sentences)


get_top_sentences(all_solo, topic_of_interest, member_topic_option)

st.markdown("## **Closing Remarks**")
st.markdown("Remember that people can be inconsistent when it comes to sentiment analysis, and machine approaches  "
            "are no different. We also need to consider the domain under which words are used. For example, typically  "
            "the word 'bias' should be considered negative. But for kpop, and fandoms in general, bias instead means  "
            "the member you favor the most in a group. This explains why the sentiment for 'bias' is roughly evenly  "
            "split. The word 'bias' is usually found with other positive words, but if it stands on its own, it will  "
            "make a statement negative.")
