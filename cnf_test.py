from sat.solver import SolverFactory
from sat.entity import CNF

if __name__ == "__main__":
    solver = SolverFactory().create('minisat')
    cnf = CNF('data/satlab/300/sat_000000.npz')
    result = solver._solve(cnf)
    print(result.to_str())
