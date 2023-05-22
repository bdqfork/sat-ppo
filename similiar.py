from sat.entity import CNF
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def parse(filepath: str) -> tuple:
    data = np.load(filepath, allow_pickle=True)
    return zip(data['filepaths'], data['pre_assignments'])


def points_cosine() -> float:
    def _points(filepath, pre_assignments) -> np.ndarray:
        pre_assignments = np.abs(pre_assignments)
        pre_assignments = np.sort(pre_assignments)
        cnf = CNF(filepath)
        matrix = np.zeros((len(pre_assignments), cnf.var_num))
        for i, pre_assignment in enumerate(pre_assignments):
            for v in pre_assignment:
                v = abs(v) - 1
                matrix[i][v] = 1
        return matrix

    a = np.array([_points(filepath, pre_assignment)
                  for filepath, pre_assignment in parse('tmp/lookahead.npz')])
    b = np.array([_points(filepath, pre_assignment)
                  for filepath, pre_assignment in parse('tmp/net.npz')])
    cos = []
    for (x, y) in zip(a, b):
        cos.append(cosine_similarity(x, y))
    return np.array(cos).mean()


def sub_cosine() -> float:
    def _sub_matrix(filepath, pre_assignments) -> np.ndarray:
        cnf = CNF(filepath)
        matrix = []
        for pre_assignment in pre_assignments:
            copy: CNF = cnf.copy()
            copy.simpfy(pre_assignment)
            matrix.append(cnf.clause_num - copy.clause_num)
        matrix = np.array(matrix)
        return matrix
    a = np.array([_sub_matrix(filepath, pre_assignment)
                  for filepath, pre_assignment in parse('tmp/lookahead.npz')])
    b = np.array([_sub_matrix(filepath, pre_assignment)
                  for filepath, pre_assignment in parse('tmp/net.npz')])
    cos = cosine_similarity(a, b)
    return cos.mean()


def main():
    print(f'point cosine: {points_cosine()}')
    print(f'sub cosine: {sub_cosine()}')


if __name__ == '__main__':
    main()
