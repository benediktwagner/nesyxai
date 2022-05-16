import pickle
import pandas as pd
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('-dd', '--data_dir', default='data_path')
parser.add_argument('-md', '--model_dir', default='experiments')


if __name__ == '__main__':
    args = parser.parse_args()

    n_rows = 1655633
    skip = np.arange(n_rows)
    skip = np.delete(skip, np.arange(0, n_rows, 10))

    data = pd.read_csv(os.path.join(args.data_dir, 'cleaaneed.csv'),skiprows=skip,usecols=['labels'])
    labels = data['labels'].values

    labs_mapping = pickle.load(open(os.path.join('data_path', 'labs_mapping.pkl'), 'rb'))
    labs_mapping_inverse = {val: key for key, val in labs_mapping.items()}

    cav_bottlenecks = pickle.load(open(os.path.join(args.model_dir, 'cav_bottlenecks.pkl'), 'rb'))
    concepts = pickle.load(open(os.path.join(args.model_dir, 'concepts.pkl'), 'rb'))
    cavs = pickle.load(open(os.path.join(args.model_dir, 'cavs.pkl'), 'rb'))
    grads = pickle.load(open(os.path.join(args.model_dir, 'grads.pkl'), 'rb'))

    scores = dict()
    for key, val in concepts.items():
        print(f'TCAV for {key} ...')

        dot_prod = grads.dot(cavs[key])

        tcav_scores = {}
        for lab in labels:
            tcav_score = (dot_prod[labels == lab] < 0).mean() #CHANGE TO POS
            tcav_scores[lab] = tcav_score

        tcav_scores_names = {labs_mapping_inverse[key]: val for key, val in tcav_scores.items()}

        scores[key] = tcav_scores_names

    pickle.dump(scores, open(os.path.join(args.model_dir, 'scores.pkl'), 'wb'))
