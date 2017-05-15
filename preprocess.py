import numpy as np
import random

OUTPUT_DIR = 'data/preprocessed/'

def preprocess_data(filename):
    with open(filename, 'r') as f:
        _, labels, features = zip(*(line.strip().split('\t') for line in f))
        
        num_data = len(labels)
        print('Num data read: %d' % (num_data))

        features = [[f.split('/') for f in sample.split()] for sample in features]

        dev_ix = set(random.sample(xrange(num_data), num_data / 5))

        train_labels = np.array([l for i, l in enumerate(labels) if i not in dev_ix])
        dev_labels = np.array([l for i, l in enumerate(labels) if i in dev_ix])

        train_features = np.array([l for i, l in enumerate(features) if i not in dev_ix])
        dev_features = np.array([l for i, l in enumerate(features) if i in dev_ix])

        train_seq_len = np.array([len(sample) for sample in train_features])
        dev_seq_len = np.array([len(sample) for sample in dev_features])

        np.save(OUTPUT_DIR + 'train_labels', train_labels)
        np.save(OUTPUT_DIR + 'train_features', train_features)
        np.save(OUTPUT_DIR + 'train_seq_len', train_seq_len)
        np.save(OUTPUT_DIR + 'dev_labels', dev_labels)
        np.save(OUTPUT_DIR + 'dev_features', dev_features)
        np.save(OUTPUT_DIR + 'dev_seq_len', dev_seq_len)        

if __name__ == '__main__':
    train_data = 'data/raw/AClassification.train.txt'
    preprocess_data(train_data)