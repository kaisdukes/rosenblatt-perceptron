import unittest

from split import split
from src.perceptron import Perceptron
from src.train import train
from src.evaluate import evaluate
from src.iris import load_iris_dataset


class PerceptronTest(unittest.TestCase):
    IRIS_FILE = 'data/iris.csv'

    def test_ten_fold_cross_validation(self):
        for fold in range(10):
            self._train_and_test(fold)

    def _train_and_test(self, fold: int):

        # train
        print(f'Fold {fold}')
        feature_vectors, labels = load_iris_dataset(self.IRIS_FILE)
        train_vectors, test_vectors, train_labels, test_labels = split(feature_vectors, labels, fold)

        perceptron = Perceptron(
            weights=[0.0 for _ in feature_vectors[0]],
            bias=0.0,
            learning_rate=0.01)

        train(perceptron, train_vectors, train_labels, epochs=10)

        # test
        accuracy = evaluate(perceptron, test_vectors, test_labels)

        print(f'Accuracy: {accuracy * 100:.2f}%')


if __name__ == '__main__':
    unittest.main()
