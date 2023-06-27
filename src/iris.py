from typing import List, Tuple


def load_iris_dataset(file_path: str) -> Tuple[List[List[float]], List[float]]:
    feature_vectors = []
    labels = []

    with open(file_path, 'r') as file:
        next(file)  # skip header

        for line in file:
            columns = [x.strip().strip('"') for x in line.strip().split(',')]

            if columns[-1] in ('Setosa', 'Versicolor'):
                features = [float(value) for value in columns[:-1]]
                label = 1 if columns[-1] == 'Setosa' else 0

                feature_vectors.append(features)
                labels.append(label)

    return feature_vectors, labels
