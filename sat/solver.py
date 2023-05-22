from sat.entity import CNF
import signal
import os
import subprocess
from tempfile import NamedTemporaryFile
import re


class Result(object):
    def __init__(self) -> None:
        super().__init__()

    def cpu_time(self) -> float:
        raise NotImplementedError

    def satisfiable(self) -> bool:
        raise NotImplementedError

    def assignments(self) -> str:
        raise NotImplementedError

    def to_str(self, assigntments: bool = False) -> str:
        raise NotImplementedError

    def keys(self) -> list:
        raise NotImplementedError

    def values(self) -> list:
        raise NotImplementedError


class AbortResult(Result):
    def __init__(self) -> None:
        super().__init__()

    def cpu_time(self) -> float:
        return 0.0

    def satisfiable(self) -> bool:
        return False

    def assignments(self) -> str:
        return ''

    def to_str(self, assigntments: bool = False) -> str:
        lines = []
        lines.append('{} : {}'.format('cpu_time', self.cpu_time()))
        lines.append('{} : {}'.format('satisfiable', self.satisfiable()))
        return '\n'.join(lines)

    def keys(self) -> list:
        return ['cpu_time', 'satisfiable']

    def values(self) -> list:
        return [self.cpu_time(), self.satisfiable()]


PATTERN = re.compile(r'[a-z,A-Z].*:.*\d')
NAMES = {
    'Number of variables': 'var_num',
    'Number of clauses': 'clause_num',
    'Parse time': 'parse_time',
    'Eliminated clauses': 'eliminated_clauses',
    'Simplification time': 'simplification_time',
    'restarts': 'restarts',
    'conflicts': 'conflicts',
    'decisions': 'decisions',
    'propagations': 'propagations',
    'conflict literals': 'conflict_literals',
    'Memory used': 'memory_used',
    'CPU time': 'cpu_time'
}


class MinisatResult(Result):
    def __init__(self, result_info: str, assignments: str) -> None:
        super().__init__()
        lines = PATTERN.findall(result_info)
        satisfiable = None
        if result_info.rfind('UNSATISFIABLE') != -1:
            satisfiable = False
        elif result_info.rfind('SATISFIABLE') != -1:
            satisfiable = True
        result = {}
        for line in lines:
            key, value = line.split(':')
            key = key.strip()
            key = NAMES[key]
            value = value.strip()
            if value.find(' ') != -1:
                value = value[:value.find(' ')]
            value = value.strip()
            result[key] = value
        result['satisfiable'] = satisfiable
        result['assignments'] = assignments
        self.result = result

    def var_num(self) -> int:
        if 'var_num' in self.result.keys():
            return int(self.result['var_num'])
        return -1

    def clause_num(self) -> int:
        if 'clause_num' in self.result.keys():
            return int(self.result['clause_num'])
        return -1

    def parse_time(self) -> float:
        if 'parse_time' in self.result.keys():
            return float(self.result['parse_time'])
        return -1.0

    def eliminated_clauses(self) -> float:
        if 'eliminated_clauses' in self.result.keys():
            return float(self.result['eliminated_clauses'])
        return -1.0

    def simplification_time(self) -> float:
        if 'simplification_time' in self.result.keys():
            return float(self.result['simplification_time'])
        return -1.0

    def restarts(self) -> int:
        if 'restarts' in self.result.keys():
            return int(self.result['restarts'])
        return -1

    def conflicts(self) -> int:
        if 'conflicts' in self.result.keys():
            return int(self.result['conflicts'])
        return -1

    def decisions(self) -> int:
        if 'decisions' in self.result.keys():
            return int(self.result['decisions'])
        return -1

    def propagations(self) -> int:
        if 'propagations' in self.result.keys():
            return int(self.result['propagations'])
        return -1

    def conflict_literals(self) -> int:
        if 'conflict_literals' in self.result.keys():
            return int(self.result['conflict_literals'])
        return -1

    def memory_used(self) -> float:
        if 'memory_used' in self.result.keys():
            return float(self.result['memory_used'])
        return -1.0

    def cpu_time(self) -> float:
        if 'cpu_time' in self.result.keys():
            return float(self.result['cpu_time'])
        return -1.0

    def satisfiable(self) -> bool:
        return self.result['satisfiable']

    def assignments(self) -> str:
        return self.result['assignments']

    def to_str(self, assigntments: bool = False) -> str:
        lines = []
        for (k, v) in self.result.items():
            if k == 'assignments' and (not assigntments or not self.satisfiable()):
                continue
            lines.append('{} : {}'.format(k, v))
        return '\n'.join(lines)

    def keys(self) -> list:
        return list(filter(lambda k: k != 'assignments', self.result.keys()))

    def values(self) -> list:
        return [self.result[k] for k in self.keys()]


class Solver(object):
    def __init__(self, timeout=60) -> None:
        super().__init__()
        self.timeout = timeout

    def solve(self, cnf: CNF, pre_assignments: list = [], points: list = None, learnts=[]) -> tuple:
        '''
        批量求解
        '''
        # 直接求解原问题
        if len(pre_assignments) == 0 and points is None:
            return [pre_assignments], [self._solve(cnf, [], [], self.timeout)]
        # 求解子问题
        pre_assignments = split(pre_assignments, points)
        results = [self._solve(cnf, pre_assign, learnts, self.timeout)
                   for pre_assign in pre_assignments]

        return pre_assignments, results

    def _solve(self, cnf: CNF, pre_assignment=[], learnts=[], timeout=60) -> Result:
        raise NotImplementedError


class MinisatSolver(Solver):
    def __init__(self, timeout) -> None:
        super().__init__(timeout=timeout)

    def _solve(self, cnf: CNF, pre_assignment=[], learnts=[], timeout=60) -> Result:
        lines = cnf.to_str(pre_assignment, learnts)
        if lines is None:
            return AbortResult()
        with NamedTemporaryFile(mode="w", dir=r"/tmp") as f:
            f.writelines(lines)
            f.flush()
            out = f.name + '.out'
            try:
                process = subprocess.Popen(
                    'minisat {} {}'.format(f.name, out), stdout=subprocess.PIPE,
                    shell=True, start_new_session=True, encoding='utf-8')
                info, _ = process.communicate(timeout=timeout)
            except subprocess.TimeoutExpired as e:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                raise e
            assignments = ''
            with open(out, 'r') as o:
                lines = o.readlines()
                if len(lines) > 1:
                    assignments = lines[1]
                    assignments = assignments[:assignments.rfind('0')]
                    assignments = assignments.strip()

            os.remove(out)
            return MinisatResult(info, assignments)


class SolverFactory(object):
    def __init__(self, timeout=60) -> None:
        super().__init__()
        self.timeout = timeout

    def create(self, solver) -> Solver:
        if solver == 'minisat':
            return MinisatSolver(self.timeout)
        else:
            raise NotImplementedError


def split(pre_assignments: list, points: list = None) -> list:
    '''
    在pre_assignments的基础上，添加分割点points
    '''
    def _split(points: list, level: int) -> list:
        if level == len(points) - 1:
            return [[points[level]],
                    [-points[level]]]
        res = []
        for item in _split(points, level + 1):
            res.append([points[level]] + item)
            res.append([-points[level]] + item)
        return res

    if points is None:
        return [pre_assignments]

    results = _split(points, 0)
    results = [pre_assignments + res for res in results]
    return results
