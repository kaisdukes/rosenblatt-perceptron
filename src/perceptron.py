from dataclasses import dataclass
from typing import List


@dataclass
class Perceptron:
    weights: List[float]
    bias: float
    learning_rate: float

    def update(self, input: List[float], expected_output: float):
        error = expected_output - self.predict(input)

        for i in range(len(self.weights)):
            self.weights[i] += self.learning_rate * error * input[i]
        self.bias += self.learning_rate * error

    def predict(self, input: List[float]):
        weighted_sum = sum(w * x for w, x in zip(self.weights, input)) + self.bias
        return 1 if weighted_sum >= 0 else 0
