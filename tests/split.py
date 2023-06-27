from typing import List


def split(feature_vectors: List[List[float]], labels: List[float], fold: int):
    train_vectors: List[List[float]] = []
    test_vectors: List[List[float]] = []
    train_labels: List[float] = []
    test_labels: List[float] = []

    for i, vector in enumerate(feature_vectors):
        if i % 10 == fold:
            test_vectors.append(vector)
            test_labels.append(labels[i])
        else:
            train_vectors.append(vector)
            train_labels.append(labels[i])

    return train_vectors, test_vectors, train_labels, test_labels
