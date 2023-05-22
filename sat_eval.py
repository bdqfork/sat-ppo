import utils
from sat.solver import SolverFactory
import numpy as np

from sat.entity import CNF
import os
from sat_solver import ParallelSolver


def get_cnfs(path, all=False):
    with open(path, 'r') as f:
        cnfs = f.readlines()
        if not all:
            cnfs = cnfs[:10] + cnfs[len(cnfs)-10:]
        return cnfs


def main(args):
    if not os.path.exists('tmp'):
        os.mkdir('tmp')

    p_solver = ParallelSolver(args.solver, args.level, args.topk,
                              args.core, args.model_path, args.order_model_path, 'tmp',
                              args.worker_num, args.timeout, args.order, args.co_solve, args.force)

    cnfs = get_cnfs(args.data_path, args.all)

    sat_split_times = []
    sat_scores = []
    unsat_scores = []
    unsat_split_times = []

    sat_total_time = 0
    unsat_total_time = 0

    csv_items = []
    for cnf in cnfs:
        cnf = CNF(cnf.strip())

        run_result = SolverFactory(args.timeout).create(
            args.solver)._solve(cnf, timeout=args.timeout)

        if run_result.cpu_time() < 1e-10:
            continue

        if args.verbose:
            print(run_result.to_str())

        p_result = p_solver._solve(cnf, timeout=args.timeout)
        if p_result.result is None:
            print(p_result.to_str())
        csv_items.append([cnf.filepath] + p_result.values())

        p_runtime = p_result.cpu_time()

        runtime = run_result.cpu_time()
        if args.co_solve or not args.force:
            score = (runtime - p_runtime)/runtime
        else:
            score = (runtime - p_runtime + p_result.split_time())/runtime
        if run_result.satisfiable():
            sat_total_time += p_runtime
            sat_split_times.append(p_result.split_time())
            sat_scores.append(score)
        else:
            unsat_total_time += p_runtime
            unsat_split_times.append(p_result.split_time())
            unsat_scores.append(score)
    utils.save_csv(args.output, [], csv_items)
    print(
        f'total {len(cnfs)} instances, solved in {sat_total_time + unsat_total_time} seconds......')

    sat_scores = np.array(sat_scores)
    unsat_scores = np.array(unsat_scores)

    sat_split_times = np.array(sat_split_times)
    unsat_split_times = np.array(unsat_split_times)

    print('SAT......')
    print(
        f'total time: {sat_total_time}, mean split time:{round(np.mean(sat_split_times),2)}, mean score: {round(np.mean(sat_scores),2)},\
             median score: {round(np.median(sat_scores),2)}\
            ， max score: {round(np.max(sat_scores),2)}, min score: {round(np.min(sat_scores),2)}')
    print('UNSAT......')
    print(
        f'total time: {unsat_total_time}, mean split time:{round(np.mean(unsat_split_times),2)}, mean score: {round(np.mean(unsat_scores),2)},\
             median score: {round(np.median(unsat_scores),2)}\
            ， max score: {round(np.max(unsat_scores),2)}, min score: {round(np.min(unsat_scores),2)}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model_path', type=str, default='model/sat/150/model-15099.pt',
                        help='where you store model (default: model/sat/150/model-15099.pt)')
    parser.add_argument('-j', '--order_model_path', type=str, default='model/satisfiable/model.pt',
                        help='where you store the order model (default: model/satisfiable/model.pt)')
    parser.add_argument('-d', '--data_path', type=str, default='data/satlab/150/test.txt',
                        help='where you store test data (default: data/satlab/150/test.txt)')
    parser.add_argument('-a', '--all', action="store_true",
                        help='use all test data (default: False)')
    parser.add_argument('-r', '--save_path', type=str, default='tmp',
                        help='where you store split result (default: tmp)')
    parser.add_argument('-l', '--level', type=int, default=4,
                        help='where you store train data (default: 4)')
    parser.add_argument('-t', '--timeout', type=int, default=1000,
                        help='timeout (default: 1000)')
    parser.add_argument('-k', '--topk', type=int, default=-1,
                        help='topk, must larger than zero or -1, -1 means adaptive (default: -1)')
    parser.add_argument('-c', '--core', type=str, default='net',
                        help='which agent you want to use, net or random, lookahead, net (default: net)')
    parser.add_argument('-s', '--solver', type=str, default='minisat',
                        help='solver (default: minisat)')
    parser.add_argument('-w', '--worker_num', type=int, default=4,
                        help='parallel worker number (default: 4)')
    parser.add_argument('-v', '--verbose', action="store_true",
                        help='verbose (default: False)')
    parser.add_argument('-o', '--order', action="store_true",
                        help='the segmentation results are sorted and solved in order (default: False)')
    parser.add_argument('-e', '--co_solve', action="store_true",
                        help='co_solve (default: False)')
    parser.add_argument('-f', '--force', action="store_true",
                        help='force (default: False)')
    parser.add_argument('-p', '--output', type=str, default='tmp/output.csv',
                        help='output (default: tmp/output.csv)')
    args = parser.parse_args()

    print(args)

    main(args)
