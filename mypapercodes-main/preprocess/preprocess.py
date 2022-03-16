from preprocess.vi_news_preprocess import remove_punctuation_characters, remove_number, remove_whitespace
from preprocess.vi_sentiment_preprocess import (
    handle_negation_form,
    remove_repeated_characters,

)


def execute_preprocess(text):
    text = remove_punctuation_characters(text)
    text = remove_repeated_characters(text)
    text = remove_number(text)
    text = remove_whitespace(text)
    # text = handle_negation_form(text)

    return text
