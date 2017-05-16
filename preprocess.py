import numpy as np
import random
from config import Config

OUTPUT_DIR = 'data/preprocessed/'

def preprocess_data(filename):
    with open(filename, 'r') as f:
        _, labels, raw_features = zip(*(line.strip().split('\t') for line in f))
        
        num_data = len(labels)
        print('Num data points read: %d' % (num_data))

        sliced_features = [[f.split('/') for f in sample.split()] for sample in raw_features]
        categorical_features = set()
        for sample in sliced_features:
            for f in sample:
                categorical_features.add(f[1])
        categorical_features_list = list(categorical_features)
        categorical_features_map = {c: (i + 1) for i, c in enumerate(categorical_features_list)}

        features = []
        for sample in sliced_features:
            new_sample = [[int(f[0]), categorical_features_map[f[1]]] for f in sample]
            features.append(new_sample)

        print("Done slicing features")

        dev_ix = set(random.sample(xrange(num_data), num_data / 5))

        train_labels = [l for i, l in enumerate(labels) if i not in dev_ix]
        dev_labels = [l for i, l in enumerate(labels) if i in dev_ix]

        print("Done creating labels")

        train_features = [l for i, l in enumerate(features) if i not in dev_ix]
        dev_features = [l for i, l in enumerate(features) if i in dev_ix]

        print("Done creating features")

        # train_seq_len = np.array([len(sample) for sample in train_features])
        # dev_seq_len = np.array([len(sample) for sample in dev_features])

        # print("Done creating seq_len")

        return train_labels, train_features, dev_labels, dev_features

def create_batch(labels, features, batch_size):
    label_batches = []
    feature_batches = []

    for i in xrange(0, len(labels), batch_size):
        label_batches.append(labels[i:i + batch_size])
        feature_batches.append(features[i:i + batch_size])

    return np.array(label_batches), np.array(feature_batches)

def pad_batches(feature_batches):
    padded_batches = []
    for i in xrange(len(feature_batches)):
        batch = feature_batches[i]
        max_len = max(len(s) for s in batch)

        padded_batch = []
        for s in batch:
            if max_len - len(s) > 0:
                padded_batch.append(np.pad(s, ((0, max_len - len(s)),(0, 0)), 'constant'))
            else:
                padded_batch.append(s)
        padded_batches.append(padded_batch)

    return padded_batches
     

if __name__ == '__main__':
    train_data = 'data/raw/AClassification.train.txt'
    train_labels, train_features, dev_labels, dev_features = preprocess_data(train_data)

    train_label_batch, train_feature_batch = create_batch(train_labels, train_features, Config.batch_size)
    dev_label_batch, dev_feature_batch = create_batch(dev_labels, dev_features, Config.batch_size)

    train_features_padded_batch = pad_batches(train_feature_batch)
    dev_features_padded_batch = pad_batches(dev_feature_batch)

    print(train_features_padded_batch[:5])
    print(dev_features_padded_batch[:5])

    np.save(OUTPUT_DIR + "train_features_batch", train_features_padded_batch)
    np.save(OUTPUT_DIR + "train_label_batch", train_label_batch)

    np.save(OUTPUT_DIR + "dev_features_batch", dev_features_padded_batch)
    np.save(OUTPUT_DIR + "dev_label_batch", dev_label_batch)



