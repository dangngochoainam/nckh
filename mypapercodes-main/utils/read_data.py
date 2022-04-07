from utils import news_dataset_path, BASE_DIR
from preprocess.preprocess import execute_preprocess, remove_stopwords
import pandas as pd
import os
import pickle



def read_news_dataset(path=news_dataset_path,
                      is_preprocess=True,
                      cached_file='cached_data/dataset_temp.sav',
                      is_forced=False,
                      ignored_folder=['.DS_Store']):


    cached_path = os.path.join(BASE_DIR, cached_file)
    print(cached_path)


    if is_forced and os.path.exists(cached_path):

        data = load_cached_file(cached_file_path=cached_path)

    else:

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

                        feature = remove_stopwords(text)

                        data.append({
                            'feature': feature,
                            'target': label
                        })

                label = label + 1

        save_cached_file(data=data, cached_file_path='%s/cached_data/dataset_temp.sav' %BASE_DIR)

    return pd.DataFrame(data)


def save_cached_file(data, cached_file_path):
    with open(cached_file_path, 'wb') as f:
        pickle.dump(data, f)
    print("Save file cached")


def load_cached_file(cached_file_path):
    with open(cached_file_path, 'rb') as f:
        print("Read file cached")
        return pickle.load(f)

def replace_underline(text):
    return text.replace("_", " ")

if __name__ == '__main__':
    # print(read_news_dataset().feature[0])
    read_news_dataset()




