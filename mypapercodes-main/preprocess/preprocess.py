from preprocess.vi_news_preprocess import remove_punctuation_characters, remove_number, remove_whitespace
from preprocess.vi_sentiment_preprocess import (
    handle_negation_form,
    remove_repeated_characters,
)
from pyvi import ViTokenizer
from utils import stopword

from utils.back_translate import back_translate


def execute_preprocess(text):
    text = remove_punctuation_characters(text)
    text = remove_repeated_characters(text)
    text = remove_number(text)
    text = remove_whitespace(text)
    # text = handle_negation_form(text)

    return text

def execute_new_data(text):
    text = remove_underline(text)
    text = back_translate(text)
    text = word_tokenize(text)
    text = execute_preprocess(text)
    text = remove_stopwords(text)
    return text

def word_tokenize(text):

    return ViTokenizer.tokenize(text).lower()

def remove_stopwords(text, stopword=stopword):

    words = []
    for word in text.strip().split():
        if word not in stopword:
            words.append(word)
    return ' '.join(words)

def remove_underline(text):
    return text.replace('_', ' ')
