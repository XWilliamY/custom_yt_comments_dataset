import spacy
import pandas as pd
import numpy as np
import regex
import emoji
import re

def read_in_csvs(list_filenames):
    return pd.concat([pd.read_csv(filename+".csv", names=headers) for filename in list_filenames])


def get_results_of_comparison(target_value, source_column, target_column, dataset):
    # create a boolean mask
    mask = dataset[source_column].values == target_value
    pos = np.flatnonzero(mask) # get idx
    
    return mask, pos, dataset[mask][target_column]


def strip_emojis(text):
    stripped = []
    data = regex.findall(r'\X', text)
    for word in data:
        if any(char in emoji.UNICODE_EMOJI for char in word):
            continue
        else:
            stripped.extend(word)
    return ''.join(stripped)


def strip_symbols(text):
    words = text.split()
    
    # https://stackoverflow.com/questions/58833864/python-regex-removing-all-special-characters-and-numbers-not-attached-to-words
    return ' '.join(re.findall(r'(?:[^\W\d_]+\d|\d+[^\W\d_])[^\W_]*|[^\W\d_]+', text))

def keep_emojis_strip_symbols(text):
    stripped = []
    
    text_to_words = text.split()
    for word in text_to_words:
        data = regex.findall(r'\X', word)
        has_emoji = False
        the_emoji = []
        for char_combo in data:
            if any(char in emoji.UNICODE_EMOJI for char in char_combo):
                has_emoji = True
                break
            else:
                continue
                
        # after analyzing the characters of a word
        if has_emoji:
            stripped.append(''.join(word))
        else:
            stripped.append(strip_symbols(word))
    return ''.join(stripped)


def clean_comments(document):
    nlp = spacy.load('en_core_web_md')
    cleaned = []
    for comment in nlp.pipe(document, disable=['parser']):
        sentence = [keep_emojis_strip_symbols(word.lemma_) \
                    if word.lemma_ != '-PRON-' else word.text \
                    for word in comment]
        cleaned.append(' '.join(sentence))
    return cleaned

def compare_diffs(cleaned_comments, orig_comments, cleaned_sent, orig_sent, indices):
    agree = 0
    disagree = 0
    for i in indices:
        print(cleaned_sent[i], ":", orig_sent[i])
        print(cleaned_comments[i])
        print(orig_comments[i])
        print()
        if cleaned_sent[i] != orig_sent[i]:
            disagree += 1
        else:
            agree += 1
    print(agree, disagree)


def split_pos_neg_neu(comments, sentiments):
    positive = [[index, comment_sentiment[0]] for index, comment_sentiment in \
               enumerate(zip(comments, sentiments)) \
               if comment_sentiment[1] >= 0.05]
    negative = [[index, comment_sentiment[0]] for index, comment_sentiment in \
               enumerate(zip(comments, sentiments)) \
               if comment_sentiment[1] <= -0.05]
    neutral = [[index, comment_sentiment[0]] for index, comment_sentiment in \
               enumerate(zip(comments, sentiments)) \
               if -.05 < comment_sentiment[1] < 0.05]
    return positive, negative, neutral


def get_freqs(**kwargs):
    """
    Given pos, neg, and neu splits, return freqs of each and total freq
    """
    positive_words = [word.lower() for index_comment in kwargs['pos'] \
                      for word in index_comment[1].split() \
                      if word not in stopwords]
    freq_positive_words = Counter(positive_words)
    negative_words = [word.lower() for index_comment in kwargs['neg'] \
                      for word in index_comment[1].split() \
                      if word not in stopwords]
    freq_negative_words = Counter(negative_words)
    neutral_words = [word.lower() for index_comment in kwargs['neu'] \
                     for word in index_comment[1].split() \
                     if word not in stopwords]
    freq_neutral_words = Counter(neutral_words)


def normalize_word_count(words):
    max_freq = max(words.values())
    return {key:value/max_freq for (key, value) in words.items()}

def computeIDF(*args):
    """
    Given parameters of type dict, calculate idf
    """
    import math
    num_docs = 0
    idfs = {}
    for freq_dict in args:
        for word, freq in freq_dict.items():
            if idfs.get(word):
                idfs[word] += 1
            else:
                idfs[word] = 1
        num_docs += 1
    # normalize
    for word, doc_freq in idfs.items():
        idfs[word] = math.log(num_docs / doc_freq)
    return idfs


def compute_tfidf(freqs, idfs):
    return {word:freq * idfs[word] for word, freq in freqs.items()}

def get_tfidf(comments, sentiments):
    pos, neg, neu = split_pos_neg_neu(comments, sentiments)

    freq_pos, freq_neg, freq_neu = get_freqs(pos=pos,
                                             neg=neg,
                                             neu=neu)
    
    freq_pos = normalize_word_count(freq_pos)
    freq_neg = normalize_word_count(freq_neg)
    freq_neu = normalize_word_count(freq_neu)
    
    idf = computeIDF(freq_pos, freq_neg, freq_neu)
    
    tfidf_pos = compute_tfidf(freq_pos, idf)
    tfidf_neg = compute_tfidf(freq_neg, idf)
    tfidf_neu = compute_tfidf(freq_neu, idf)
    
    return tfidf_pos, tfidf_neg, tfidf_neu, pos, neg, neu


def get_sent_strength_of_doc(doc, tfidf):
    from collections import defaultdict
    sent_strength = defaultdict(float)
    index = 0
    # for each sentence
    for sent in doc:
        # handle empty strings
        sent_length = len(sent) if len(sent) > 0 else 1
        
        for word in sent: 
            if word in tfidf.keys(): 
                    sent_strength[sent]+= tfidf[word]
        sent_strength[sent] /= sent_length
        index += 1
    return sent_strength

def get_sentences(comments):
    return [sentence \
            for index_comment in comments \
            for sentence in re.split(r'\s{2,}', index_comment[1])]

def get_comments(comments):
    return [index_comment[1] \
            for index_comment in comments]
