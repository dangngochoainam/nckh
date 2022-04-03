from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent

# news_dataset_path = '/Users/huu-thanhduong/Desktop/My-Datasets/News/tokenized_ouvnnews'
news_dataset_path = 'C:/Users/dangn/Downloads/tokenized_ouvnnews'


sentiment1_dataset_path = '/Users/huu-thanhduong/Desktop/My-Datasets/Sentiment/dataset1'
sentiment2_dataset_path = '/Users/huu-thanhduong/Desktop/My-Datasets/Sentiment/dataset1'

not_list = ['không', 'vô', 'chẳng', 'đếch', 'chưa', 'chả', 'đéo', 'kém', 'không thể',
            'không phải', 'không biết', 'không khỏi', 'chưa', 'chưa thể', 'chưa phải',
            'chưa biết', 'chưa có', 'hem được', 'hem', 'không được', 'bỏ', 'trừ', 'mất']

def read_text_file(file_path, encoding="utf-8", is_readlines=False):
    with open("%s/%s" % (BASE_DIR, file_path), "r+", encoding=encoding) as f:
        return [t.replace("\n", "") for t in f.readlines()]


def replace_empty_by_underscore(text=[]):
    for t in text:
        if t.find(" ") >= 0:
            text.append(t.replace(" ", "_"))

    return text


with open('%s/utils/stopwords.txt' %BASE_DIR, 'r', encoding="utf8") as w:
    stopword = w.read().split()

# pos_list = replace_empty_by_underscore(read_text_file(file_path='cached_data/pos.txt'))
# neg_list = replace_empty_by_underscore(read_text_file(file_path='cached_data/neg.txt'))

s1 = u'ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝàáâãèéêìíòóôõùúýĂăĐđĨĩŨũƠơƯưẠạẢảẤấẦầẨẩẪẫẬậẮắẰằẲẳẴẵẶặẸẹẺẻẼẽẾếỀềỂểỄễỆệỈỉỊịỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợỤụỦủỨứỪừỬửỮữỰựỲỳỴỵỶỷỸỹ'
s0 = u'AAAAEEEIIOOOOUUYaaaaeeeiioooouuyAaDdIiUuOoUuAaAaAaAaAaAaAaAaAaAaAaAaEeEeEeEeEeEeEeEeIiIiOoOoOoOoOoOoOoOoOoOoOoOoUuUuUuUuUuUuUuYyYyYyYy'
