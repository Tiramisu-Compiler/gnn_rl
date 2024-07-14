from env_api.scheduler.models.action import Action
import numpy as np
from typing import List, Union


class Computation:
    def __init__(self, name, comp_dict):
        self.name = name
        self.absolute_order = comp_dict["absolute_order"]
        self.iterators = comp_dict["iterators"]
        self.write_access_relation = comp_dict["write_access_relation"]
        self.write_buffer_id = comp_dict["write_buffer_id"]
        self.data_type = comp_dict["data_type"]
        self.data_type_size = comp_dict["data_type_size"]
        self.accesses = comp_dict["accesses"]
        self.expression_representation = comp_dict["expression_representation"]
        self.parent_iterator: Iterator = None

    def __str__(self) -> str:
        return f"Computation : {self.name} {self.write_access_relation}"


class Iterator:
    def __init__(self, name, iter_dict):
        self.name = name
        self.lower_bound = iter_dict["lower_bound"]
        self.upper_bound = iter_dict["upper_bound"]
        self.parent_iterator_name = iter_dict["parent_iterator"]
        self.child_iterators_names = iter_dict["child_iterators"]
        self.computations_list_names = iter_dict["computations_list"]
        self.children: List[Union[Iterator, Computation]] = []
        self.parent_iterator: Iterator = None

        # Special tags for some actions :
        is_tiled = False
        is_skewed = False
        is_unrolled = False
        is_parallel = False

    def __str__(self) -> str:
        return f"{self.name} : [{self.lower_bound},{self.upper_bound}] "


class Branch:
    def __init__(self, iterators, num_actions=0):
        self.actions_mask = np.zeros(num_actions)
        self.iterators: list[Iterator] = iterators

    def __eq__(self, other) : 
        if len(self.iterators) != len(other.iterators) : 
            return False 
        for it1, it2 in zip(self.iterators, other.iterators) : 
            if it1.name != it2.name : 
                return False         
        return True


    def __str__(self) -> str:
        s = ""
        for it in self.iterators:
            s += str(it)

        return s


class AST:
    def __init__(self, roots):
        self.roots: list[Iterator] = roots
        self.branches: list[Branch] = extract_branches(roots)

    def __str__(self) -> str:
        return build_str(self.roots)


def extract_branches(nodes: list[Iterator], prev_iters: list = [], branches: list = []):
    level_branches : list[Branch] = []
    for node in nodes : 
        if isinstance(node, Computation) : 
            branch = Branch([*prev_iters])
            if level_branches :
                if level_branches[-1] != branch  :
                    level_branches.append(branch)
            else : 
                level_branches.append(branch)
        else : 
           level_branches.extend(extract_branches(node.children,[*prev_iters, node], [*branches]))

    branches.extend(level_branches)
    return branches


def build_str(nodes: List[Union[Iterator, Computation]], indentation=""):
    tree_str = ""
    for node in nodes:
        tree_str += indentation + str(node) + "\n"
        if isinstance(node, Iterator):
            tree_str += build_str(node.children, indentation + "|\t")

    return tree_str


class State:
    def __init__(self, annotations):
        self.ast = self.generate_ast(annotations=annotations)
        self.actions_list: list[Action] = []

    def generate_legality_code(self):
        pass

    def generate_execution_code(self):
        pass

    def generate_ast(self, annotations):
        computations = annotations["computations"]
        iterators = annotations["iterators"]

        its_dict: dict[str, Iterator] = {}
        comps_dict: dict[str, Computation] = {}

        for it in iterators:
            its_dict[it] = Iterator(name=it, iter_dict=iterators[it])

        for comp in computations:
            comps_dict[comp] = Computation(name=comp, comp_dict=computations[comp])

        # Linking the iterators, computations and building loop nests with the right order of execution
        # keeping track of previous branch
        prev_branch = []
        roots = []
        for comp_name in comps_dict:
            prev_it: str = comps_dict[comp_name].iterators[0]
            if prev_branch:
                if prev_branch[0] != prev_it:
                    roots.append(its_dict[prev_it])
                for idx, it_name in enumerate(
                    comps_dict[comp_name].iterators[1:], start=1
                ):
                    if len(prev_branch) <= idx or it_name != prev_branch[idx]:
                        its_dict[prev_it].children.append(its_dict[it_name])
                    prev_it = it_name
            else:
                roots.append(its_dict[prev_it])
                for it_name in comps_dict[comp_name].iterators[1:]:
                    its_dict[prev_it].children.append(its_dict[it_name])
                    prev_it = it_name

            its_dict[prev_it].children.append(comps_dict[comp_name])
            comps_dict[comp_name].parent_iterator = its_dict[prev_it]
            prev_branch = comps_dict[comp_name].iterators

        return AST(roots)
