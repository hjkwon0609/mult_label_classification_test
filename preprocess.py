import numpy as np
import random
from config import Config
import hickle as hkl

OUTPUT_DIR = 'data/preprocessed/'

MAKE_SMALLER = True

def preprocess_data(filename):
    with open(filename, 'r') as f:
        _, raw_labels, raw_features = zip(*(line.strip().split('\t') for line in f))
        
        num_data = len(raw_labels)
        print('Num data points read: %d' % (num_data))
        
        ###############################################################################
        # preprocess for smaller data to get model working (BEGIN)
        ###############################################################################
        if MAKE_SMALLER:
            dev_ix = set(random.sample(xrange(num_data), num_data / 1000))
            raw_labels = [l for i, l in enumerate(raw_labels) if i in dev_ix]
            raw_features = [f for i, f in enumerate(raw_features) if i in dev_ix]
            num_data = len(raw_labels)
        ###############################################################################
        # preprocess for smaller data to get model working (END)
        ###############################################################################


        # Preprocess features
        # Slice each filter to its own dimension, create dummy variables for categorical variables
        # e.x) 29/7c = [29, 0, 0, 0, 0, 1, 0] 
        #      (if 7c is 5th cateogry and there are 6 categories as an example. these numbers in data might be different)
        sliced_features = [[f.split('/') for f in sample.split()] for sample in raw_features]
        length_of_features = [len(sample) for sample in sliced_features]
        

        categorical_features = set()
        for sample in sliced_features:
            for f in sample:
                categorical_features.add(f[1])
        categorical_features_list = list(categorical_features)
        num_categorical_variables = len(categorical_features)

        features = []
        for sample in sliced_features:
            new_sample = [[int(f[0])] + [1 if c == f[1] else 0 for c in categorical_features_list] for f in sample]
            features.append(new_sample)

        # sort based on length so that batch consists of similar sizes 
        features = [f for (length, f) in sorted(zip(length_of_features, features), key=lambda pair: pair[0])]

        print("Done preprocessing features")

        # Preprocess labels
        # Cange the target labels to one-hot representations
        # e.x) 0 ==> [1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
        #      0 19 ==> [1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1]
        sliced_labels = [set(int(l) for l in label.split()) for label in raw_labels]
        total_labels = set()
        for sample in sliced_labels:
            for l in sample:
                total_labels.add(int(l))
        total_labels_list = sorted(list(total_labels))
        labels = [[1 if i in sample else 0 for i in total_labels_list] for sample in sliced_labels]
        
        # sort based on length so that batch consists of similar sizes 
        labels = [l for (length, l) in sorted(zip(length_of_features, labels), key=lambda pair: pair[0])]

        print("Done preprocessing labels")
        
        dev_ix = set(random.sample(xrange(num_data), num_data / 5))
        train_labels = [l for i, l in enumerate(labels) if i not in dev_ix]
        dev_labels = [l for i, l in enumerate(labels) if i in dev_ix]

        print("Done splitting labels to train/dev")

        train_features = [l for i, l in enumerate(features) if i not in dev_ix]
        dev_features = [l for i, l in enumerate(features) if i in dev_ix]

        print("Done splitting features to train/dev")

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

    return label_batches, feature_batches

def pad_batches(feature_batches):
    padded_batches = []
    for i in xrange(len(feature_batches)):
        batch = feature_batches[i]
        max_len = max(len(s) for s in batch)

        padded_batch = []
        for s in batch:
            if max_len - len(s) > 0:
                padded_batch.append(np.pad(s, ((max_len - len(s), 0),(0, 0)), 'constant', constant_values=1e-10))
            else:
                padded_batch.append(s)
        padded_batches.append(padded_batch)

    return padded_batches


if __name__ == '__main__':
    train_data = 'data/raw/AClassification.train.txt'
    train_labels, train_features, dev_labels, dev_features = preprocess_data(train_data)

    # np.save(OUTPUT_DIR + "train_features", train_features)
    # np.save(OUTPUT_DIR + "train_label", train_labels)

    # np.save(OUTPUT_DIR + "dev_features", dev_features)
    # np.save(OUTPUT_DIR + "dev_label", dev_labels)

    train_label_batch, train_feature_batch = create_batch(train_labels, train_features, Config.batch_size)
    dev_label_batch, dev_feature_batch = create_batch(dev_labels, dev_features, Config.batch_size)

    train_features_padded_batch = pad_batches(train_feature_batch)
    dev_features_padded_batch = pad_batches(dev_feature_batch)

    print(train_features_padded_batch[:5])
    print(dev_features_padded_batch[:5])

    # train_features_batch_name = 'train_features_batch'
    # train_label_batch_name = 'train_label_batch'
    # dev_features_batch_name = 'dev_features_batch'
    # dev_label_batch_name = 'dev_label_batch'

    # if MAKE_SMALLER:
    #     train_features_batch_name = 'smaller_' + train_features_batch_name
    #     train_label_batch_name = 'smaller_' + train_label_batch_name
    #     dev_features_batch_name = 'smaller_' + dev_features_batch_name
    #     dev_label_batch_name = 'smaller_' + dev_label_batch_name


    # np.save(OUTPUT_DIR + train_features_batch_name, train_features_padded_batch)
    # print('Done saving train_features_batch')
    # np.save(OUTPUT_DIR + train_label_batch_name, train_label_batch)
    # print('Done saving train_label_batch')

    # np.save(OUTPUT_DIR + dev_features_batch_name, dev_features_padded_batch)
    # print('Done saving dev_features_batch')
    # np.save(OUTPUT_DIR + dev_label_batch_name, dev_label_batch)
    # print('Done saving dev_label_batch')

    if MAKE_SMALLER:
        np.savez_compressed(OUTPUT_DIR + 'smaller_data.npz', train_features_padded_batch=train_features_padded_batch,
                                        train_label_batch=train_label_batch,
                                        dev_features_padded_batch=dev_features_padded_batch,
                                        dev_label_batch=dev_label_batch)
    else:
        np.savez_compressed(OUTPUT_DIR + 'data.npz', train_features_padded_batch=train_features_padded_batch,
                                        train_label_batch=train_label_batch,
                                        dev_features_padded_batch=dev_features_padded_batch,
                                        dev_label_batch=dev_label_batch)

    # hkl.dump(train_features_padded_batch, OUTPUT_DIR + "train_features_batch.hkl")
    # hkl.dump(train_label_batch, OUTPUT_DIR + "train_label_batch.hkl")

    # hkl.dump(dev_features_padded_batch, OUTPUT_DIR + "dev_features_batch.hkl")
    # hkl.dump(dev_label_batch, OUTPUT_DIR + "dev_label_batch.hkl")



