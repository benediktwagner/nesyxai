"""
utility functions
"""

import json
import re
import requests
import yaml
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
import pandas as pd
from pymystem3 import Mystem
from nltk.util import ngrams
import nltk

mystem = Mystem()
regex = re.compile(r'[^\w\s]')


def get_yaml_config(config_path):

    with open(config_path, 'r', encoding='utf-8') as f:
        params = yaml.load(f)

    return params


def save_dict_to_yaml(d, yaml_path):
    with open(yaml_path, 'w') as file:
        yaml.dump(d, file, default_flow_style=False)


def save_dict_to_json(d, json_path):
    with open(json_path, 'w') as f:
        json.dump(d, f, indent=4)


def save_vocab_to_txt_file(vocab, txt_path):
    with open(txt_path, "w") as f:
        f.write("\n".join(token for token in vocab))


def calculate_metrics(probs, labels):

    y_pred = np.argmax(probs, axis=1)

    metrics = dict()
    metrics['f1'] = f1_score(y_true=labels, y_pred=y_pred,average='micro')
    metrics['acc'] = accuracy_score(y_true=labels, y_pred=y_pred)

    return metrics


def clean(text, lemmatize=True):
    text = regex.sub(r' ', text).strip()
    text = re.sub(r' +', ' ', text)
    text = text.lower()
    if lemmatize:
        text = ' '.join([lemma for lemma in mystem.lemmatize(text) if not lemma.isspace()])
    return text

# def clean(dataframe):
#     dataframe
#     text = regex.sub(r' ', text).strip()
#     text = re.sub(r' +', ' ', text)
#     text = text.lower()
#     if lemmatize:
#         text = ' '.join([lemma for lemma in mystem.lemmatize(text) if not lemma.isspace()])
#     return text

# def input_data(data_path, params, train_time=True):
#
#     data = pd.read_csv(data_path)
#     dataset = tf.data.Dataset.from_tensor_slices(({'tokens': data['text'].values}, data['labels'].values))
#
#     vocab = build_vocab(params['vocab_path'])
#     pad_value = vocab.lookup(tf.constant('<PAD>'))
#     fake_padding = tf.constant(-1, dtype=tf.int64)
#
#     if train_time:
#         dataset = dataset.shuffle(params['train_size'])
#
#     dataset = dataset.map(lambda feats, labs: (vectorize(feats['tokens'], vocab, params['seq_len']), labs))
#
#     padded_shapes = (tf.TensorShape([params['seq_len']]), tf.TensorShape([]))
#     padding_values = (pad_value, fake_padding)
#
#     dataset = dataset.padded_batch(
#         batch_size=params['batch_size'],
#         padded_shapes=padded_shapes,
#         padding_values=padding_values
#     )
#
#     dataset = dataset.map(lambda x, y: ({'x': x, 'y': y}, y))
#     dataset = dataset.prefetch(buffer_size=2)
#
#     return dataset

def chunks(l, n):
    """Yield successive n-sized chunks from l.
    Source: https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
    """
    n_items = len(l)
    if n_items % n:
        n_pads = n - n_items % n
    else:
        n_pads = 0
    l = l + ['<PAD>' for _ in range(n_pads)]
    for i in range(0, len(l), n):
        yield l[i:i + n]

def sequentialise(dataframe, seq_len):
    dataframe['text'] = dataframe['text'].str.split().apply(lambda x: list(chunks(x, 4)))
    dataframe = dataframe.explode('text').reset_index(drop=True)
    dataframe['text'] = dataframe['text'].apply(' '.join)
    return(dataframe)

def get_ngrams(text, n=3):
    words = text.split()
    ngram_tokens = ngrams(words, n)
    return list(ngram_tokens)


def data_seqe(data):
    train = pd.read_csv('/Users/ben/GitHub/nlp_cav/data_path/train.csv')
    train['text'] = train['text'].apply(clean)
    df = train
    longform = pd.DataFrame(columns=['text', 'labels'])

    for idx, content in df.iterrows():
        name_words = (i.lower() for i in content[0].split())
        bla = list(ngrams(name_words,20))
        longform = longform.append(
            [{'word': ng, 'labels': content[1]} for ng in bla],
            ignore_index=True
        )
    longform['test'] = longform['word'].apply(', '.join)
    longform['test1'] = longform['test'].str.replace(',', '')
    longform.to_csv('/Users/ben/GitHub/nlp_cav/data_path/blablu.csv')
    print("DONE")
    # train['bigrams'] = train['text'].apply(lambda row: list(ngrams(row, 2)))
    # # train['text'] = train['text'].apply(clean)
    # new_df = pd.DataFrame()
    # for index, row in train.iterrows():
    #     # print(row[0])
    #     new_df['text'] = get_ngrams(row[0],20)
    #     # new_df['labels'] = row[1]
    #
    #     print(row)
    #     print(labels)
    #
    #     get_ngrams(train['text'],20)
    # for i in train['text']:
    #     get_ngrams(i , 20)
    #
    # for i in train['text']:
    #     pad_sequences(train['text'])
    # labels = train['labels'].values

def vectorize(text, seq_len, vocab, emb_matrix):
    text = clean(text)
    tokens = text.split()

    vectorized = [0] * seq_len  # <PAD>

    # vocab_size = 111230 # MAKE GENERIC
    # embed_size = 300

    # def get_coefs(word, *arr):
    #     return word, np.asarray(arr, dtype='float32')
    # embedding_index = dict(get_coefs(*o.strip().split(" ")) for o in open(emb_matrix))
    # embedding_matrix = np.zeros((vocab_size + 1, embed_size))

    for i, tok in enumerate(tokens):
        # print(i,tok)
        try:
            tok_id = vocab[tok]
            # embedding_vector = embedding_index.get(word)
            # if embedding_vector is not None: embedding_matrix[i] = embedding_vector
        except KeyError:
            tok_id = 1  # <UNK>
        # except IndexError:
        #     print("i",i,"tok",tok,"tokid",tok_id)

        vectorized[i] = tok_id # vectorized.append(tok_id)

    return emb_matrix[vectorized, :] #embedding_matrix


def one_hot(a, num_classes):
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)]).reshape(-1, num_classes)

def parse_kb(input,number=5):

    regex = r"(?<=c/en/)\w*"

    obj = requests.get('http://api.conceptnet.io/related/c/en/'+input+'?filter=/c/en')

    test_string = obj.text
    match = re.findall(regex,test_string)
    lisst = []
    for i in range(len(match)):
        a = match[i].split('_')
        lisst.append(a)

    flat_list = []
    for sublist in lisst:
        for item in sublist:
            if item != str(input) and item != "and" and item != "of":
                flat_list.append(item)

    if flat_list:
        return(flat_list[:number])
    else:
        print("Unsuccess")