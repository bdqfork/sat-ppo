import numpy as np
from sat import utils
import dgl
import torch


class Var(object):
    def __init__(self, index: int, val: bool = None) -> None:
        """
        docstring
        """
        self.index = index
        self.val = val

    def copy(self) -> object:
        '''
        深度拷贝
        '''
        return Var(self.index, self.val)


class Lit(object):
    def __init__(self, var: Var, positive: bool):
        """
        docstring
        """
        self.var = var
        self.positive = positive

    def val(self) -> bool:
        '''
        获取文字的值，
        如果尚未赋值，返回None，否则返回对应的值
        '''
        if self.unknow():
            return None
        if self.positive:
            return self.var.val
        else:
            return not self.var.val

    def var_index(self) -> int:
        return self.var.index

    def unknow(self) -> bool:
        return self.var.val is None

    def to_str(self) -> str:
        if self.unknow():
            var_index = self.var_index()
            if self.positive:
                return str(var_index)
            else:
                return '-' + str(var_index)
        return None

    def code(self) -> list:
        if self.positive:
            return [1, 0]
        else:
            return [0, 1]


class Clause(object):
    def __init__(self, lits: list = []) -> None:
        self.lits = lits

    def append(self, lit: Lit) -> None:
        self.lits.append(lit)

    def constant(self) -> bool:
        '''
        判断句子是否恒为真
        '''
        for lit in self.lits:
            var: Var = lit.var
            positive: bool = lit.positive
            if var.val is None:
                continue

            if (var.val and positive) or (not var.val and not positive):
                return True
        return False

    def unit(self) -> Var:
        '''
        判断句子是否为单元子句，
        如果是单元子句，返回对应的文字，否则返回None
        '''
        if len(self.lits) == 1:
            return self.lits[0]
        else:
            return None

    def to_str(self) -> str:
        line = ''
        for lit in self.lits:
            if lit.unknow():
                line += lit.to_str() + ' '
        if len(line) > 0:
            return line + '0\n'
        return None

    def copy(self) -> object:
        copy_lits = [lit.copy() for lit in self.lits]
        return Clause(copy_lits)

    def simpfy(self) -> bool:
        '''
        化简，如果是恒为真，返回False，否则返回True
        '''
        lits = []
        for lit in self.lits:
            val: bool = lit.val()
            if val is None:
                lits.append(lit)
            elif val:
                return False
        self.lits = lits
        return True

    def empty(self) -> bool:
        '''
        判断是否为空集
        '''
        return len(self.lits) == 0


class CNF(object):
    def __init__(self, cnf_file: str, parse: bool = True) -> None:
        super().__init__()

        self.filepath = cnf_file
        self._graph = None

        if not parse:
            return

        var_num, clause_num, indices, values = utils.parse_cnf(
            self.filepath)

        self.var_num = var_num
        self.clause_num = clause_num
        self.lits_num = len(indices)

        self.vars = [Var(i + 1) for i in range(0, self.var_num)]
        self.clauses = [Clause([]) for i in range(0, self.clause_num)]
        self.lits = []
        for ((clause_index, var_index), value) in zip(indices, values):
            lit = Lit(self.vars[var_index], value[0] == 1)
            self.lits.append(lit)
            self.clauses[clause_index].append(lit)

    def filename(self):
        start = self.filepath.rfind('/')
        if start == -1:
            return self.filepath
        return self.filepath[start + 1:]

    def write(self, path: str, pre_assignments: list = [], learnts=[], direct: bool = False) -> bool:
        '''
        返回是否写入成功
        '''
        lines = self.to_str(pre_assignments, learnts, direct)
        if lines is None:
            return False
        with open(path, 'w') as f:
            for line in lines:
                f.write(line)
        return True

    def to_graph(self, pre_assignments: list = []):
        def __to_graph(self) -> tuple:
            edge_data = np.zeros((self.lits_num * 2, 2), dtype=np.float32)
            connectivity = np.zeros((2, edge_data.shape[0]), dtype=np.int)
            ec = 0
            for i, clause in enumerate(self.clauses):
                for lit in clause.lits:
                    edge_data[ec:ec +
                              2] = np.array(lit.code())
                    # from var to clause
                    connectivity[0, ec] = lit.var_index() - 1
                    connectivity[1, ec] = self.var_num + i
                    # from clause to var
                    connectivity[0, ec + 1] = self.var_num + \
                        i
                    connectivity[1, ec + 1] = lit.var_index() - 1
                    ec += 2

            vertex_data = np.zeros(
                (self.var_num + self.clause_num, 2), dtype=np.float32
            )
            vertex_data[:self.var_num, 0] = 1
            vertex_data[self.var_num:, 1] = 1

            global_data = np.zeros((1, 1), dtype=np.float32)

            return (vertex_data, edge_data, connectivity, global_data)

        if self._graph is None:
            self._graph = __to_graph(self)

        (vertex_data, edge_data, connectivity, global_data) = self._graph

        mask = np.zeros((vertex_data.shape[0], 1), dtype=np.float32)

        embbed = np.zeros(vertex_data.shape, dtype=np.float32)

        for pre_assignment in pre_assignments:
            var = abs(pre_assignment) - 1
            mask[var] = mask[var] - 1
            if pre_assignment > 0:
                embbed[var] = np.array([0, 1])
            if pre_assignment < 0:
                embbed[var] = np.array([-1, 0])

        return vertex_data, edge_data, connectivity, global_data, embbed, mask

    def to_dgl_graph(self, pre_assignments: list = []):
        _, _, cnf = self.copy().unit_propagation(pre_assignments)
        # separately count clauses and variables
        clausep, varp = [], []
        clausen, varn = [], []
        for i, clause in enumerate(cnf.clauses):
            for lit in clause.lits:
                if lit.positive:
                    clausep.append(i)
                    varp.append(lit.var_index() - 1)
                else:
                    clausen.append(i)
                    varn.append(lit.var_index() - 1)

        clausep, varp = np.array(clausep), np.array(varp)
        clausen, varn = np.array(clausen), np.array(varn)

        # two kinds of nodes, variable nodes and clause nodes
        ntypes = ('clause', 'var')
        # since DGL has no undirected edges, it uses bidirectional edges to achieve
        etypes = ('cvp', 'cvn', 'vcp', 'vcn')

        # build graph
        graph = dgl.heterograph({
            (ntypes[0], etypes[0], ntypes[1]): (clausep, varp),
            (ntypes[0], etypes[1], ntypes[1]): (clausen, varn),
            (ntypes[1], etypes[2], ntypes[0]): (varp, clausep),
            (ntypes[1], etypes[3], ntypes[0]): (varn, clausen)})

        for i, ntype in enumerate(ntypes):
            graph.nodes[ntype].data['x'] = torch.tensor(
                [i]*graph.number_of_nodes(ntype)).long().reshape(-1, 1)
        return graph

    def unit_propagation(self, pre_assignments: list, learnts=[], direct: bool = False) -> tuple:
        '''
        单元传播，返回是否有冲突以及赋值的变量、cnf副本
        '''
        cnf = self
        if not direct:
            cnf = self.copy()

        for learnt in learnts:
            cnf.append_clause(learnt)

        for item in pre_assignments:
            var_index = abs(item) - 1
            var: Var = cnf.vars[var_index]
            var.val = item > 0

        implied_assignments = []

        while True:
            count = 0
            clauses = filter(lambda clause: clause.simpfy(), cnf.clauses)
            for clause in clauses:
                lit: Lit = clause.unit()
                if lit is None or lit.val() is not None:
                    continue

                count += 1

                var: Var = lit.var
                if lit.positive:
                    var.val = True
                    implied_assignments.append(var.index)
                else:
                    var.val = False
                    implied_assignments.append(-var.index)

            if count == 0:
                cnf.clauses = list(
                    filter(lambda clause: clause.simpfy(), cnf.clauses))

                cnf.lits = [
                    lit for clause in cnf.clauses for lit in clause.lits]

                cnf.clause_num = len(cnf.clauses)
                cnf.lits_num = len(cnf.lits)

                conflict = False
                for clause in cnf.clauses:
                    if clause.empty():
                        conflict = True
                        break

                return (conflict, implied_assignments, cnf)

    def copy(self) -> object:
        vars = [var.copy() for var in self.vars]
        vars.sort(key=lambda v: v.index)

        clauses = [Clause([Lit(vars[lit.var.index - 1], lit.positive)
                           for lit in clause.lits]) for clause in self.clauses]

        lits = [lit for clause in clauses for lit in clause.lits]

        copy_cnf = CNF(self.filepath, parse=False)
        copy_cnf.var_num = len(vars)
        copy_cnf.clause_num = len(clauses)
        copy_cnf.lits_num = len(lits)
        copy_cnf.vars = vars
        copy_cnf.clauses = clauses
        copy_cnf.lits = lits
        return copy_cnf

    def append_clause(self, lits: list) -> None:
        '''
        添加子句
        '''
        if len(lits) == 0:
            return
        lits = [Lit(self.vars[abs(i) - 1], i > 0) for i in lits]
        self.lits += lits
        clause = Clause(lits)
        self.clauses.append(clause)
        self.clause_num += 1

    def to_str(self, pre_assignments: list = [], learnts=[], direct: bool = False) -> list:
        conflict, implied_assignments, cnf = self.unit_propagation(
            pre_assignments, learnts, direct)

        if conflict:
            return None

        pre_assignments += implied_assignments

        lines = []
        for clause in cnf.clauses:
            line = ''
            if clause.constant():
                continue
            line = clause.to_str()
            lines.append(line)
        for pre_assign in pre_assignments:
            lines.append('{} 0\n'.format(pre_assign))

        head = 'p cnf {} {}\n'.format(cnf.var_num, len(lines))
        lines.insert(0, head)
        if not direct:
            del cnf
        return lines
