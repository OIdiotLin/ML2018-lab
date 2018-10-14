import pickle

import numpy as np
import os
import matplotlib.pyplot as plt

from skimage import io, transform, color
from feature import NPDFeature
from ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

NPD_FEATURED_DATASET_DAT = 'npd_featured_dataset.dat'
SAMPLES_N = 400
SCALE_SIZE = (24, 24)
WEAKERS_LIMIT = 16


def load_dataset():
    def npd_features(imgs):
        features = np.empty(shape=(0, SCALE_SIZE[0] * (SCALE_SIZE[0]-1) // 2))
        for i in range(len(imgs)):
            feature = NPDFeature((imgs[i] * 255).astype(np.int8)).extract()
            features = np.vstack((features, feature))
        print(features.shape)
        return features

    imgs = []
    img_labels = []

    if os.path.exists(NPD_FEATURED_DATASET_DAT):
        with open(NPD_FEATURED_DATASET_DAT, 'rb') as f:
            return pickle.load(f)

    for i in range(SAMPLES_N):
        for category in ['face', 'nonface']:
            img = io.imread('./dataset/original/{c}/{c}_{idx:0>3d}.jpg'.format(c=category, idx=i))
            img_gray = color.rgb2gray(img)
            img_gray = transform.resize(img_gray, SCALE_SIZE, mode='constant', anti_aliasing=True)
            imgs.append(img_gray)
            img_labels.append(1 if category == 'face' else -1)
    img_features = npd_features(imgs)
    img_labels = np.fromiter(img_labels, float)

    with open(NPD_FEATURED_DATASET_DAT, 'wb') as f:
        pickle.dump((img_features, img_labels), f)

    return img_features, img_labels


def split_dataset(dataset, train_ratio=0.8):
    """
    :return: X_train, y_train, X_valid, y_valid
    """
    pivot = int(2*SAMPLES_N * train_ratio)
    train_set = dataset[0][:pivot], dataset[1][:pivot]
    valid_set = dataset[0][pivot:], dataset[1][pivot:]
    return train_set + valid_set


if __name__ == "__main__":
    X_train, y_train, X_valid, y_valid = split_dataset(load_dataset())

    adaBoost = AdaBoostClassifier(DecisionTreeClassifier, WEAKERS_LIMIT)
    accs = adaBoost.fit(X_train, y_train, X_valid, y_valid)

    plt.figure(figsize=[8, 5])
    plt.title('Accuracy')
    plt.xlabel('Num of weak classifiers')
    plt.ylabel('Accuracy')
    plt.plot(accs[0], '--', c='b', linewidth=3, label='train')
    plt.plot(accs[1], c='r', linewidth=3, label='valid')
    plt.legend()
    plt.grid()
    plt.savefig('AdaBoost-accuracy.png')
    plt.show()

    AdaBoostClassifier.save(adaBoost, 'AdaBoost-Model.pkl')
