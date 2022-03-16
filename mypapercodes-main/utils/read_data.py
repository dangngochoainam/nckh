from utils import news_dataset_path, BASE_DIR
from preprocess.preprocess import execute_preprocess
import pandas as pd
import os
import pickle


# 21/2
import regex as re
#


# def read_news_dataset(path=news_dataset_path):
#
#     total_label = 25
#     vocab = {}
#     label_vocab = {}
#     count = {}
#
#
#     with open('D:/NCKH/Python/mypapercodes-main/utils/stopwords.txt', 'r', encoding="utf8") as w:
#         stopword = w.read().split()
#
#     data = []
#     for folder in os.listdir(path):
#         label_path = '%s/%s' % (path, folder)
#
#         for file in os.listdir(label_path):
#             file_path = '%s/%s' % (label_path, file)
#
#
#     #         with open(file_path, 'r', encoding="utf8") as f:
#     #             data.append({
#     #                 'feature': f.readline(),
#     #                 'target': folder
#     #             })
#     # return pd.DataFrame(data)
#
#
#
#             with open(file_path, 'r', encoding="utf8") as f:
#
#                 text = f.readline()
#                 #
#                 # # 21/2
#                 # xóa bỏ các kí tự đặc biệt
#                 text = re.sub(r'[^\s\wáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ_]',' ',text)
#                 # xóa bỏ khoảng trắng thừa
#                 text = re.sub(r'\s+', ' ', text).strip()
#                 text = execute_preprocess(text=text)
#                 # xóa bỏ các stopword
#                 words = []
#                 for word in text.strip().split():
#                     if word not in stopword:
#                         words.append(word)
#                 feature = ' '.join(words)
#
#                 data.append({
#                     'feature': feature,
#                     'target': folder
#                 })
#     return pd.DataFrame(data)
#
#                  # tìm stop word
#     #             words = text.split()
#     #             # lưu ý từ đầu tiên là nhãn
#     #             label = folder
#     #             if label not in label_vocab:
#     #                 label_vocab[label] = {}
#     #             for word in words:
#     #                 label_vocab[label][word] = label_vocab[label].get(word, 0) + 1
#     #                 if word not in vocab:
#     #                     vocab[word] = set()
#     #                 vocab[word].add(label)
#     #
#     # for word in vocab:
#     #     if len(vocab[word]) == total_label:
#     #
#     #         count[word] = min([label_vocab[x][word] for x in label_vocab])
#     #
#     #
#     #
#     # sorted_count = sorted(count, key=count.get, reverse=True)
#     # for word in sorted_count[:50]:
#     #     print(word, count[word])
#     #
#     # stopword = set()
#     # sorted_count = sorted(count, key=count.get, reverse=True)
#     # with open('stopwords.txt', 'w', encoding="utf8") as fp:
#     #     for word in sorted_count[:50]:
#     #         stopword.add(word)
#     #         fp.write(word + '\n')
#     #
#     # return pd.DataFrame(data)




def read_news_dataset(path=news_dataset_path,
                      is_preprocess=True,
                      is_forced=False,
                      # cached_file='cached_data/news_dataset.sav',
                      ignored_folder=['.DS_Store']):
    # cached_path = os.path.join(BASE_DIR, cached_file)
    # if not is_forced and os.path.exists(cached_path):
    #     data = load_cached_file(cached_file_path=cached_path)
    # else:


    with open('D:/NCKH/Python/22.2/mypapercodes-main/utils/stopwords.txt', 'r', encoding="utf8") as w:
        stopword = w.read().split()


    data = []
    label = 0
    print("===", path)
    for folder in os.listdir(path):
        if folder not in ignored_folder:
            label_path = '%s/%s' % (path, folder)

            for file in os.listdir(label_path):
                file_path = '%s/%s' % (label_path, file)
                with open(file_path, 'r', encoding="utf8") as f:
                    text = f.readline()
                    if is_preprocess:
                        text = execute_preprocess(text=text)
                    # xóa bỏ các stopword
                    words = []
                    for word in text.strip().split():
                        if word not in stopword:
                            words.append(word)
                    feature = ' '.join(words)

                    data.append({
                        'feature': feature,
                        'target': label
                    })

            label = label + 1

        # save_cached_file(data=data, cached_file_path=cached_path)

    return pd.DataFrame(data)


def save_cached_file(data, cached_file_path):
    with open(cached_file_path, 'wb') as f:
        pickle.dump(data, f)


def load_cached_file(cached_file_path):
    with open(cached_file_path, 'rb') as f:
        return pickle.load(f)

if __name__ == '__main__':
    # read_news_dataset()
    print(read_news_dataset().feature[0])

