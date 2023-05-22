import sys
from queue import Queue
import numpy as np
import torch
from torch.tensor import Tensor

from models import GNN
from sat.spaces import Graph
from sat.entity import CNF
from torch.distributions import Categorical
from sat.utils import batch_graphs
from torch.multiprocessing import Manager
import os


class Agent(object):
    def __init__(self, level=4) -> None:
        super().__init__()
        self.level = level

    def split(self, cnf: CNF):
        import time
        start = time.time()
        pre_assignments, learnts = self.cube(cnf)
        end = time.time()
        runtime = end - start
        return runtime, pre_assignments, learnts

    def cube(self, cnf: CNF):
        cnf = cnf.copy()
        results = []

        buffer = []
        buffer.append(([], []))

        learnts = []
        threshold = 2 ** self.level
        while len(buffer) > 0:
            pre_assignments, implied_assignments = buffer.pop()

            action = self.decision(
                cnf, pre_assignments, implied_assignments)

            t = pre_assignments + [action]
            f = pre_assignments + [-action]

            t_conflict, t_implied_assignments, _ = cnf.unit_propagation(
                t, learnts)

            f_conflict, f_implied_assignments, _ = cnf.unit_propagation(
                f, learnts)

            if t_conflict and f_conflict:
                learnts.append(t)
                learnts.append(f)
                continue

            if t_conflict:
                learnts.append(t)
            else:
                if len(t) < self.level:
                    buffer.append((t, t_implied_assignments))
                else:
                    results.append(t)
            if f_conflict:
                learnts.append(f)
            else:
                if len(f) < self.level:
                    buffer.append((f, f_implied_assignments))
                else:
                    results.append(f)

            while len(learnts) > threshold:
                learnts.pop(0)

        return results, learnts

    def decision(self, cnf: CNF, pre_assignments: list, implied_assignments: list):
        raise NotImplementedError


class RandomAgent(Agent):
    def __init__(self, level=4) -> None:
        super().__init__(level)

    def decision(self, cnf: CNF, pre_assignments: list, implied_assignments: list):
        import numpy as np
        vars = set((i + 1 for i in range(cnf.var_num)))
        candidates = vars.difference(
            set((abs(i) for i in pre_assignments + implied_assignments)))
        return np.random.choice(list(candidates))


class NetAgent(Agent):
    def __init__(self, model_path, level=4) -> None:
        super().__init__(level)
        self.model = GNN.factory().create_eval(Graph(dtype=np.float, shape=[2, 2, 1]), Graph(
            dtype=np.float, shape=(1,)))
        print(f'Loading {model_path}')
        checkpoint = torch.load(
            model_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint['net'])
        self.model.eval()
        self.model.share_memory()

    def decision(self, cnf: CNF, pre_assignments: list, implied_assignments: list):
        with torch.no_grad():
            graph = cnf.to_graph(pre_assignments)
            graph = [torch.tensor(el) for el in graph]
            graph = batch_graphs([graph])
            b_policy, _ = self.model([graph])
            mask = set(pre_assignments + implied_assignments)
            return self.__choose(b_policy, mask)

    def __choose(self, logits, mask):
        with torch.no_grad():
            logits = logits.flatten()
            distribution = Categorical(logits=logits)
            probs = distribution.probs

            if len(mask) > 0:
                for i in mask:
                    probs[abs(i) - 1] = 0

            m_prob = probs.mean()
            for i, prob in enumerate(probs):
                if prob < m_prob:
                    probs[i] = 0

            probs /= probs.sum()
            distribution = Categorical(probs)
            action = distribution.sample()
            action = int(action.detach().numpy()) + 1
            return action


class LookAheadAgent(Agent):
    def __init__(self, level=4) -> None:
        super().__init__(level)

    def decision(self, cnf: CNF, pre_assignments: list, implied_assignments: list) -> int:
        vars = set((i + 1 for i in range(0, cnf.var_num)))
        candidates = vars.difference(
            set((abs(i) for i in pre_assignments + implied_assignments)))
        _, _, cnf = cnf.unit_propagation(pre_assignments + implied_assignments)
        maxScore = -sys.maxsize - 1
        action = 0
        for i in candidates:
            _, _, t_copy = cnf.unit_propagation([i])

            t_score = cnf.clause_num - t_copy.clause_num

            _, _, f_copy = cnf.unit_propagation([-i])

            f_score = cnf.clause_num - f_copy.clause_num

            score = t_score * f_score

            if maxScore < score:
                maxScore = score
                action = i

        return action


class MutiAgent(Agent):
    def __init__(self, model_path, k=-1, level=4) -> None:
        super().__init__(level)

        # 确保不会选择已经选择过的分割点
        if k <= level and k != -1:
            print(f'level = {level} < k = {k}, adjust k = {level << 1}')
            k = level << 1

        self.k = k
        self.model = GNN.factory().create_eval(Graph(dtype=np.float, shape=[2, 2, 1]), Graph(
            dtype=np.float, shape=(1,)))
        print(f'Loading {model_path}')

        checkpoint = torch.load(
            model_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint['net'])
        self.model.eval()
        self.model.share_memory()

    def decision(self, cnf: CNF, pre_assignments: list, implied_assignments: list):
        with torch.no_grad():
            graph = cnf.to_graph(pre_assignments)
            graph = [torch.tensor(el) for el in graph]
            graph = batch_graphs([graph])
            b_policy, _ = self.model([graph])
            return self.__choose(cnf, b_policy, pre_assignments, implied_assignments)

    def __choose(self, cnf: CNF, logits, pre_assignments, implied_assignments):
        with torch.no_grad():
            logits = logits.flatten()
            distribution = Categorical(logits=logits)
            action = distribution.sample()
            action = int(action.detach().numpy()) + 1
            probs = distribution.probs
            m_prob = probs.mean()
            mask = set(pre_assignments + implied_assignments)
            if action in mask or -action in mask or probs[action - 1] < m_prob:
                k = self.k
                if k == -1:
                    k = [i for i in range(
                        int(cnf.var_num*0.05), int(cnf.var_num*0.1))]
                    k = np.random.choice(k)

                _, candidates = torch.topk(logits, k)

                candidates = set(candidates.detach().numpy() + 1)
                candidates = candidates-set((abs(i) for i in pre_assignments))

                scores = []
                for i in candidates:
                    _, _, t_copy = cnf.unit_propagation([i])

                    t_score = cnf.clause_num - t_copy.clause_num

                    _, _, f_copy = cnf.unit_propagation([-i])

                    f_score = cnf.clause_num - f_copy.clause_num

                    score = t_score * f_score
                    scores.append((i, score))

                scores.sort(key=lambda score: score[1], reverse=False)
                action = scores[-1][0]

            return action


class ParallelAgent(Agent):
    def __init__(self, core, model_path, order_model_path, output_path, k=-1, worker_num=4, level=4, order=False) -> None:
        super().__init__(level=level)
        self.worker_num = worker_num
        self.output_path = output_path
        self.order = order
        if order:
            self.dgl_model = torch.load(order_model_path)
            self.dgl_model.eval()
            print(f'Loading order model {order_model_path}')
        if core == 'net':
            self.agent = NetAgent(model_path, level)
        elif core == 'lookahead':
            self.agent = LookAheadAgent(level)
        elif core == 'muti':
            self.agent = MutiAgent(model_path, k, level)
        elif core == 'random':
            self.agent = RandomAgent(level)
        else:
            raise NotImplementedError

    def p_decision(self, cnf: CNF, pre_assignments, implied_assignments, results: Queue, learnts: list) -> tuple:
        action = self.decision(
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
        import time
        start = time.time()
        count, learnts = self.cube(cnf, pool)
        end = time.time()
        runtime = end - start
        return runtime, count, learnts

    def cube(self, cnf: CNF, pool):
        cnf = cnf.copy()
        threshold = 10000
        m = Manager()

        buffer = []

        results = m.Queue(threshold)
        learnts = m.list()

        buffer.append(([], []))

        filepath = f'{self.output_path}/{cnf.filename()}.out'
        if os.path.exists(filepath):
            os.remove(filepath)
        count = 0
        while len(buffer) > 0:
            futures = []
            for _ in range(min(self.worker_num, len(buffer))):
                (pre_assignments, implied_assignments) = buffer.pop()
                f = pool.apply_async(self.p_decision, args=(
                    cnf, pre_assignments, implied_assignments, results, learnts))
                futures.append(f)
            for f in futures:
                buffer += f.get()
            if results.qsize() > threshold:
                count += self.save(cnf, filepath, results)
            while len(learnts) > threshold:
                learnts.pop(0)

        if not results.empty():
            count += self.save(cnf, filepath, results)
        return count, learnts

    def save(self, cnf: CNF, filepath, results: Queue):
        with torch.no_grad():
            with open(filepath, "a") as f:
                count = 0
                while not results.empty() and results.qsize() > 0:
                    pre_assignments = results.get()
                    if self.order:
                        dgl_graph = cnf.to_dgl_graph(pre_assignments)
                        pre_assignments = ' '.join(
                            [str(i) for i in pre_assignments])
                        probs: Tensor = self.dgl_model(dgl_graph).flatten()
                        # 获取可满足的概率
                        prob = probs[1].item()
                        f.write(f'{pre_assignments},{prob}\n')
                        count += 1
                    else:
                        pre_assignments = ' '.join(
                            [str(i) for i in pre_assignments])
                        f.write(f'{pre_assignments}\n')
                        count += 1
                return count

    def decision(self, cnf: CNF, pre_assignments: list, implied_assignments: list):
        return self.agent.decision(cnf, pre_assignments, implied_assignments)
