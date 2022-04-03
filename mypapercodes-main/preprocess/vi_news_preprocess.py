import re


def remove_punctuation_characters(text):
    punctuation = """!"#$%&\'()*+,-./:;<=>?@[\\]^`{|}~"""
    translator = str.maketrans(punctuation, ' ' * len(punctuation))
    text = re.sub(r'[^\s\wáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ_]',
                  ' ', text)

    return text.translate(translator)


def remove_number(text):
    return re.sub('[0-9./]+', '', text, flags=re.IGNORECASE)

def remove_whitespace(text):
    return re.sub(r'\s+', ' ', text).strip()

def remove_html(text):
    return re.sub(r'<[^>]*>', '', text)



if __name__ == '__main__':
    text = 'I love this book, this book is nice 3000!'
    print(remove_punctuation_characters(text))
    print(remove_number(text))