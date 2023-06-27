from typing import List

from .perceptron import Perceptron


def train(perceptron: Perceptron, feature_vectors: List[List[float]], labels: List[float], epochs: int):
    for _ in range(epochs):
        for i, input in enumerate(feature_vectors):
            perceptron.update(input, labels[i])
