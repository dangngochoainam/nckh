from preprocess.vi_news_preprocess import remove_punctuation_characters, remove_number, remove_whitespace
from preprocess.vi_sentiment_preprocess import (
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


if __name__ == '__main__':
    text = "10 . yamaha xsr700 : giống với đàn_anh xsr900 , yamaha xsr700 về cơ_bản là một chiếc mt - 07 với một_vài thay_đổi để tạo ra phong_cách retro hơn . tại châu âu , sức hút của xsr700 là rất lớn khi có hơn 11.000 chiếc đã được bán ra . xe trang_bị động_cơ 2 xy - lanh song_song , dung_tích 689 cc , sản_sinh công_suất 74 mã_lực . đi cùng với động_cơ này là hộp_số 6 cấp và hệ_thống phun xăng điện_tử . xe có_giá bán tại thị_trường mỹ là 8.499 usd , tương_đương 206,4 triệu đồng ."

    print(execute_preprocess(word_tokenize(text)))
