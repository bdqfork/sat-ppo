from sat.solver import Result, SolverFactory
from typing import Any
import gym
import numpy as np
import os

from .entity import CNF
from .spaces import Graph
import json


class SatObs(object):
    def __init__(self, graphs: list = []) -> None:
        super().__init__()
        self.graphs = np.array(graphs, dtype=np.object)


class SatAction(object):
    def __init__(self, actions: list = []) -> None:
        super().__init__()
        self.actions = np.array(actions, dtype=np.int)


class Buffer(object):
    def __init__(self, mod: str) -> None:
        super().__init__()
        self._mod = mod
        self.list = []

    def isEmpty(self) -> bool:
        return len(self.list) == 0

    def pop(self) -> Any:
        return self.list.pop(0)

    def append(self, item: Any) -> None:
        if self._mod == 'bfs':
            self.list.append(item)
        elif self._mod == 'dfs':
            self.list.insert(0, item)
        else:
            raise Exception(f"unsupport mod: {self._mod}")

    def clear(self) -> None:
        self.list.clear()


class Sat(gym.Env):
    def __init__(self, config_path: str) -> None:
        self.config_path = config_path
        self.config = {}
        with open(self.config_path, 'r') as f:
            self.config = json.load(f)
        self.cnfs = []

        data_path = self.config['data_path']
        with open(data_path, 'r', encoding='utf-8') as f:
            data_path = data_path[: data_path.rfind('/')]
            for line in f.readlines():
                line = line.strip()
                line = line[line.rfind('/')+1:]
                line = os.path.join(data_path, line)
                self.cnfs.append(line)

        self.solver = SolverFactory(
            self.config['timeout']).create(self.config['solver'])

        self.buffer = Buffer(self.config['mod'])

        self.state = None

        self.level = self.config['level']

        self.observation_space = Graph(dtype=np.float, shape=[2, 2, 1])

        self.action_space = Graph(dtype=np.float, shape=(1,))

        self.p_conflict = self.config['p_conflict']

        self.conflict_buffer = []

        self.p_buffer = self.config['p_buffer']

    def _done(self):
        return self.buffer.isEmpty()

    def step(self, action: np.ndarray) -> tuple:
        (cnf, pre_assignments, run_result) = self.state

        cur_lv = len(pre_assignments)

        reward = 0

        action = action + 1

        if (action[0] in pre_assignments or -action[0]
                in pre_assignments):
            if len(pre_assignments) == self.level:
                reward = 0
            else:
                reward = -1
        else:
            pre_assignments, run_results = self.solver.solve(cnf,
                                                             pre_assignments, action)

            run_results.sort(key=lambda r: r.cpu_time())
            current_result = run_results[-1]

            reward = self._reward(run_result, current_result)

            if len(pre_assignments[0]) <= self.level:

                for pre_assign, run_result in zip(pre_assignments, run_results):
                    self.buffer.append((cnf, pre_assign, run_result))

        obs = None

        done = self._done()

        if not done:
            self.state = self.buffer.pop()
            (cnf, pre_assignments, _) = self.state
            obs = cnf.to_graph(pre_assignments)
        return (obs, reward, done, cur_lv)

    def _reward(self, origin: Result, current: Result) -> float:
        reward = (origin.cpu_time() - current.cpu_time()) / \
            (origin.cpu_time() + 1e-5)

        # cv_ratio = current.clause_num()/current.var_num()
        # score = abs(cv_ratio - 4.26)/4.26
        # reward += score

        # conflict = origin.conflicts() - current.conflicts()
        # if self.config['env'] == 'explore':
        #     if len(self.conflict_buffer) == self.p_buffer:
        #         self.conflict_buffer.pop(0)
        #     self.conflict_buffer.append(abs(conflict))
        #     self._update_p()
        return reward

    def _update_p(self):
        threshold = int(np.median(self.conflict_buffer))
        if threshold > self.p_conflict:
            print(
                f'Adjust p_conflict from {self.p_conflict} to {threshold}')
            self.p_conflict = threshold
            self.config['p_conflict'] = self.p_conflict
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f)

    def _sample(self) -> tuple:
        cnf = np.random.choice(self.cnfs)
        cnf = CNF(cnf)
        return cnf

    def reset(self) -> tuple:
        self.buffer.clear()
        cnf: CNF = self._sample()
        result = self.solver._solve(cnf)
        self.state = (cnf, [], result)
        return cnf.to_graph()


if __name__ == "__main__":
    from gym import make
    from gym.envs.registration import register
    register(
        id="sat-v0", entry_point="sat.minisat:minisat_env", kwargs={'data_path': 'test', 'level': 4, 'timeout': 60})
    solver = make('sat-v0')
    cnf = solver.reset()
    (obs, reward, done) = solver.step([1])
    print(reward)
