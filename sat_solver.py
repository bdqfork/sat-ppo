from queue import Queue
import time

import torch
from sat.solver import Result, Solver, SolverFactory
from sat.entity import CNF
from tqdm import tqdm
import os
from sat_agent import LookAheadAgent, MutiAgent, NetAgent, RandomAgent
from torch.multiprocessing import Manager, Pool


class ParallelResult(Result):
    def __init__(self, runtime: float, split_time: float, result: Result) -> None:
        super().__init__()
        self.result = result
        self.runtime = runtime
        self.cube_time = split_time

    def cpu_time(self) -> float:
        return self.runtime + self.cube_time

    def split_time(self) -> float:
        return self.cube_time

    def satisfiable(self) -> bool:
        return self.result.satisfiable()

    def assignments(self) -> str:
        return self.result.assignments()

    def to_str(self, assigntments: bool) -> str:
        lines = []
        lines.append(self.result.to_str())
        lines.append(f'split_time: {self.split_time()}')
        lines.append(f'p_cpu_time: {self.cpu_time()}')
        if assigntments:
            lines.append(f'assigntments: {self.assignments()}')
        return '\n'.join(lines)

    def keys(self) -> list:
        return list(filter(lambda k: k != 'assignments', self.result.keys())) + ['split_time', 'p_cpu_time']

    def values(self) -> list:
        return [v for v in self.result.values()] + [self.split_time(), self.cpu_time()]


class ParallelSolver(Solver):
    def __init__(self, solver, level, topk, core, model_path, output_path, worker_num,
                 timeout) -> None:
        super().__init__(timeout=timeout)
        self.worker_num = worker_num
        self.output_path = output_path
        self.level = level

        if core == 'net':
            self.agent = NetAgent(model_path, level)
        elif core == 'lookahead':
            self.agent = LookAheadAgent(level)
        elif core == 'muti':
            self.agent = MutiAgent(model_path, topk, level)
        elif core == 'random':
            self.agent = RandomAgent(level)
        else:
            raise NotImplementedError

        if not os.path.exists(output_path):
            os.makedirs(output_path)
        self.solver = SolverFactory(timeout).create(solver)

    def decision(self, cnf: CNF, pre_assignments, implied_assignments, results: Queue, learnts: list) -> tuple:
        action = self.agent.decision(
            cnf, pre_assignments, implied_assignments)

        t = pre_assignments + [action]
        f = pre_assignments + [-action]

        t_conflict, t_implied_assignments, _ = cnf.unit_propagation(t, learnts)

        f_conflict, f_implied_assignments, _ = cnf.unit_propagation(f, learnts)

        buffer = []
        if t_conflict and f_conflict:
            learnts.append(t)
            learnts.append(f)
            return buffer

        if t_conflict:
            learnts.append(t)
        else:
            if len(t) < self.level:
                buffer.append((t, t_implied_assignments))
            else:
                results.put(t)
        if f_conflict:
            learnts.append(f)
        else:
            if len(f) < self.level:
                buffer.append((f, f_implied_assignments))
            else:
                results.put(f)

        return buffer

    def split(self, cnf: CNF, pool):
        start = time.time()
        cnf = cnf.copy()

        threshold = 10000
        co_threshold = threshold
        m = Manager()

        buffer = []

        results = m.Queue(threshold)
        learnts = m.list()

        buffer.append(([], []))

        filepath = f'{self.output_path}/{cnf.filename()}.out'
        if os.path.exists(filepath):
            os.remove(filepath)
        count = 0
        res = None
        while len(buffer) > 0:
            futures = []
            for _ in range(min(self.worker_num, len(buffer))):
                (pre_assignments, implied_assignments) = buffer.pop()
                f = pool.apply_async(self.decision, args=(
                    cnf, pre_assignments, implied_assignments, results, learnts), error_callback=lambda e: print(e))
                futures.append(f)
            for f in futures:
                buffer += f.get()
            if results.qsize() > co_threshold:
                count += self.save(cnf, filepath, results)

            while len(learnts) > threshold:
                learnts.pop(0)

        if not results.empty():
            count += self.save(cnf, filepath, results)

        runtime = time.time() - start
        return runtime, count, learnts, res

    def save(self, cnf: CNF, filepath, results: Queue):
        with torch.no_grad():
            with open(filepath, "a") as f:
                count = 0
                while not results.empty() and results.qsize() > 0:
                    pre_assignments = results.get()
                    pre_assignments = ' '.join(
                        [str(i) for i in pre_assignments])
                    f.write(f'{pre_assignments}\n')
                    count += 1
                return count

    def _solve(self, cnf: CNF, pre_assignment=[], learnt=[], timeout=60) -> ParallelResult:

        r_queue = Manager().Queue()

        pool = Pool(self.worker_num)

        split_time, count, learnts, run_result = self.split(
            cnf, pool)

        start = time.time()
        filepath = f'{self.output_path}/{cnf.filename()}.out'
        with open(filepath, "r") as f:
            more = True
            epoch = 0
            while more:
                split_results = []
                for _ in range(0, self.worker_num << 1):
                    line = f.readline()
                    split_results.append([int(el)
                                         for el in line.strip().split(' ')])
                    if not line:
                        more = False
                        break
                if not more:
                    break
                for pre_assignment in split_results:
                    pool.apply_async(self.solver._solve, args=(
                        cnf, pre_assignment, learnts, timeout), callback=lambda r: r_queue.put(r),
                        error_callback=lambda e: print(e))

                bar = tqdm(
                    desc=f'{cnf.filepath}-{epoch}-{self.worker_num << 1}', total=len(split_results))
                for _ in range(len(split_results)):
                    run_result: Result = r_queue.get(timeout)
                    if run_result.satisfiable():
                        pool.close()
                        pool.terminate()
                        more = False
                        break
                    bar.update()
                bar.close()
                epoch += 1
        end = time.time()
        runtime = round(end-start, 4)

        split_time = round(split_time, 4)
        return ParallelResult(runtime, split_time, run_result)


def main(args):
    if not os.path.exists('tmp'):
        os.mkdir('tmp')

    if args.agent == 'serial':
        solver = SolverFactory(args.timeout).create(args.solver)
    elif args.agent == 'parallel':
        solver = ParallelSolver(args.solver, args.level, args.topk,
                                args.core, args.model_path,
                                args.output_path, args.worker_num, args.timeout)
    else:
        raise NotImplementedError
    cnf = CNF(args.data_path.strip())
    print(solver._solve(cnf, timeout=args.timeout).to_str(True))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model_path', type=str, default='model/sat/150/model-15099.pt',
                        help='where you store model (default: model/sat/150/model-15099.pt)')
    parser.add_argument('-d', '--data_path', type=str, default='data/satlab/500/sat_000000.npz',
                        help='where you store test data (default: data/satlab/300/sat_000000.npz)')
    parser.add_argument('-r', '--output_path', type=str, default='tmp',
                        help='where you store split result (default: tmp)')
    parser.add_argument('-l', '--level', type=int, default=6,
                        help='where you store train data (default: 4)')
    parser.add_argument('-t', '--timeout', type=int, default=1000,
                        help='timeout (default: 1000)')
    parser.add_argument('-k', '--topk', type=int, default=-1,
                        help='topk, must larger than zero or -1, -1 means adaptive (default: -1)')
    parser.add_argument('-a', '--agent', type=str, default='parallel',
                        help='which agent you want to use, parallel or serial (default: parallel)')
    parser.add_argument('-c', '--core', type=str, default='net',
                        help='which agent you want to use, net or random, lookahead, muti (default: muti)')
    parser.add_argument('-w', '--worker_num', type=int, default=4,
                        help='parallel worker number (default: 4)')
    parser.add_argument('-s', '--solver', type=str, default='minisat',
                        help='solver (default: minisat)')
    parser.add_argument('-o', '--order', action="store_true",
                        help='the segmentation results are sorted and solved in order (default: False)')
    args = parser.parse_args()

    print(args)
    main(args)
