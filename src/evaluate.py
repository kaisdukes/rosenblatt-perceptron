from typing import List

from .perceptron import Perceptron


def evaluate(perceptron: Perceptron, feature_vectors: List[List[float]], labels: List[float]):
    correct = 0

    for i, input in enumerate(feature_vectors):
        prediction = perceptron.predict(input)
        if prediction == labels[i]:
            correct += 1

    return correct / len(feature_vectors)
