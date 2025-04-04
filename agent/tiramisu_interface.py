from collections import namedtuple
import numpy as np
from ray import logger
import tiralib.tiramisu as tiralib
import tiralib.config as tiralib_config
from agent.graph_utils import (
    encode_data_type,
    isl_map_to_write_access_matrix,
    pad_access_matrix,
)
from utils.dataset_actor import TiramisuProgramCache


NEXT_ACTION_INDEX = 55
SECOND = 1000
MAX_ITERATOR_DEPTH = 5
# BUFFER_ACCESS_EMBEDDING_START = (MAX_ITERATOR_DEPTH + 1) * (MAX_ITERATOR_DEPTH +2)
# + 1 (VECTOR_TYPE) + 3 (DATA_TYPE_ENCODING) + 1 (REDUCTION) + 1 (WRITE_BUFFER_ID)
BUFFER_ACCESS_EMBEDDING_START = 49
VECTOR_SIZE = 720


class TiramisuInterface:
    def __init__(
        self,
        cpp_code: str,
        tiralib_config_path: str,
        cache: TiramisuProgramCache | None = None,
        machine: str = "jubail",
        use_server: bool = True,
    ):
        tiralib_config.BaseConfig.init(tiralib_config_path)
        self.current_branch_index = 0
        self.action_indices: list[int] = []
        self._initial_execution_time: float | None = None
        self.cache = cache
        self.machine = machine
        self.use_server = use_server
        if self.cache:
            self.tiramisu_program = tiralib.TiramisuProgram.from_annotations(
                self.cache.program_annotation, cpp_code=cpp_code, load_tree=True
            )
        else:
            if not self.use_server:
                raise ValueError("Server must be used when cache is not provided")
            self.tiramisu_program = tiralib.TiramisuProgram.init_server(
                cpp_code=cpp_code,
                load_isl_ast=True,
                load_tree=True,
                load_annotations=True,
                reuse_server=True,
            )

        self.schedule = tiralib.Schedule(self.tiramisu_program)
        assert self.initial_execution_time, "Getting initial execution time failed"

        # self.branches = self.schedule_branches

    @property
    def server(self):
        if not self.tiramisu_program.server:
            self.tiramisu_program.server = tiralib.FunctionServer(
                self.tiramisu_program, reuse_server=True
            )
        return self.tiramisu_program.server

    @property
    def initial_execution_time(self):
        if not self._initial_execution_time:
            if self.cache:
                self._initial_execution_time = self.cache.execution_time(
                    self.machine, "empty"
                )
            if not self._initial_execution_time:
                self.init_server()
                self._initial_execution_time = median_execution_time(self.schedule)
                if self.cache:
                    self.cache.add_execution_time(
                        self.machine,
                        "empty",
                        self._initial_execution_time,
                    )
        return self._initial_execution_time

    @property
    def branches(self) -> list[list[tiralib.IteratorIdentifier]]:
        branches = []

        for sections in self.tree.get_candidate_sections().values():
            branches.extend(sections)

        return branches

    @property
    def current_branch(self) -> list[tiralib.IteratorIdentifier]:
        return self.branches[self.current_branch_index]

    @property
    def tree(self):
        return self.schedule.tree

    def init_server(self):
        if self.use_server:
            _ = self.server

    def get_mask(self, mask_size: int = 56):
        mask = np.zeros(mask_size)

        # TODO break out of this order
        for optim in self.schedule.optims_list:
            match type(optim):
                case tiralib.tiramisu_actions.Skewing:
                    # Mask interchange, Reversal, and Skewing
                    mask[ActionSlices.INTERCHANGE] = 1
                    mask[ActionSlices.REVERSAL] = 1
                    mask[ActionSlices.SKEWING] = 1
                    # Mask Tiling2D
                    mask[ActionSlices.TILING2D] = 1
                case tiralib.tiramisu_actions.Parallelization:
                    # Mask interchange, Reversal, and Skewing
                    mask[ActionSlices.INTERCHANGE] = 1
                    mask[ActionSlices.REVERSAL] = 1
                    mask[ActionSlices.SKEWING] = 1
                case tiralib.tiramisu_actions.Tiling2D:
                    # Mask interchange, Reversal, Skewing, Parallelization, and Tiling2D
                    mask[ActionSlices.INTERCHANGE] = 1
                    mask[ActionSlices.REVERSAL] = 1
                    mask[ActionSlices.SKEWING] = 1
                    mask[ActionSlices.PARALLELIZATION] = 1
                    mask[ActionSlices.TILING2D] = 1
                case tiralib.tiramisu_actions.Unrolling:
                    # Mask all actions
                    mask[0:55] = 1
                case (
                    tiralib.tiramisu_actions.Interchange
                    | tiralib.tiramisu_actions.Reversal
                ):
                    # No masks
                    pass
                case _:
                    raise ValueError(f"Unsupported action type {type(optim)}")

        # hide all previous actions
        for action_index in self.action_indices:
            if action_index == NEXT_ACTION_INDEX:
                continue
            mask[action_index] = 1

        if len(self.current_branch) == 1:
            mask[ActionSlices.INTERCHANGE] = 1
            mask[ActionSlices.SKEWING] = 1
            mask[ActionSlices.TILING2D] = 1

        # check if the most in depth node in the branch is a leaf iterator
        # if node has children then mask Unrolling
        iterator = self.tree.iterators[self.current_branch[-1]]
        if iterator.child_iterators:
            mask[ActionSlices.UNROLLING] = 1

        # mask levels that are not in current branch
        # TODO this is a temporary solution, we need to find a better way handle iterator depth
        levels = [level for level in range(MAX_ITERATOR_DEPTH)]
        for iterator in self.current_branch:
            # if iterator[1] not in levels:
            # print(f"iterator {iterator} not in levels {levels}")
            levels.remove(iterator[1])

        for level in levels:
            # REVERSAL has actions that work on a single iterator
            mask[ActionSlices.REVERSAL.start + level] = 1
            # UNROLLING is always applied to the leaf iterator
            # the parameter is thus used for factor not level

            # only 2 actions for parallelization
            if level < 2:
                mask[ActionSlices.PARALLELIZATION.start + level] = 1

            # interchange and tiling2d have actions that work on successive tuples
            # (0,1), (1,2), (2,3), etc.
            tuple_actions_start_indices = [
                ActionSlices.INTERCHANGE.start,
                ActionSlices.SKEWING.start,
            ] + [
                i
                for i in range(
                    ActionSlices.TILING2D.start, ActionSlices.TILING2D.stop, 4
                )
            ]

            for tuple_action_start_index in tuple_actions_start_indices:
                for action_index in _get_level_action_indices_tuple_actions(
                    level, tuple_action_start_index
                ):
                    mask[action_index] = 1
        return mask

    def _tree_to_iterator_vectors(self):
        schedule_tree = self.tree
        it_dict = {}
        for it in schedule_tree.iterators:
            single_iter_vector = -np.ones(VECTOR_SIZE)
            # Type 0 for iterators
            single_iter_vector[IteratorTags.TYPE_TAG] = 0
            # Initialize the non paddings tags to 0
            # The focus tag is the first valid tag
            single_iter_vector[IteratorTags.FOCUS_TAG :] = 0

            lower_bound_is_int = isinstance(
                schedule_tree.iterators[it].lower_bound, int
            )
            # TODO Create a better embedding for non rectangular domains
            single_iter_vector[IteratorTags.LOWER_BOUND_IS_INT_TAG] = (
                1 if lower_bound_is_int else 0
            )
            single_iter_vector[IteratorTags.LOWER_BOUND_VALUE_TAG] = (
                (schedule_tree.iterators[it].lower_bound) if lower_bound_is_int else 0
            )
            upper_bound_is_int = isinstance(
                schedule_tree.iterators[it].upper_bound, int
            )
            # TODO Create a better embedding for non rectangular domains
            single_iter_vector[IteratorTags.UPPER_BOUND_IS_INT_TAG] = (
                1 if upper_bound_is_int else 0
            )
            single_iter_vector[IteratorTags.UPPER_BOUND_VALUE_TAG] = (
                (schedule_tree.iterators[it].upper_bound) if upper_bound_is_int else 0
            )
            it_dict[it] = single_iter_vector

        return it_dict

    # TODO recompute the annotations in tiramisu after schedule is applied
    def _get_comp_annotations(self, comp: str):
        if comp in self.annotations["computations"]:
            return self.annotations["computations"][comp]
        else:
            for initial_comp in self.annotations["computations"]:
                if initial_comp in comp:
                    return self.annotations["computations"][initial_comp]
        raise ValueError(f"Computation {comp} not found in annotations")

    def _annotations_to_comps_vectors(self):
        max_depth = MAX_ITERATOR_DEPTH
        dict_comp = {}
        comps = self.tree.computations
        for comp in comps:
            single_comp_vector = -np.ones(VECTOR_SIZE)
            # vector type 0 for iterators and 1 for computations
            single_comp_vector[0] = 1

            comp_dict = self._get_comp_annotations(comp)
            # This field represents the absolute order of execution of computations
            single_comp_vector[1] = self.tree.computations_absolute_order[comp]
            # a vector of one-hot encoding of possible 3 data-types
            single_comp_vector[2:5] = encode_data_type(comp_dict["data_type"])
            single_comp_vector[5] = +comp_dict["comp_is_reduction"]
            # The write-to buffer id
            single_comp_vector[6] = +comp_dict["write_buffer_id"]
            # We add a vector of write access
            write_matrix = isl_map_to_write_access_matrix(
                comp_dict["write_access_relation"]
            )
            padded_matrix = pad_access_matrix(write_matrix, max_depth).reshape(-1)
            single_comp_vector[7 : 7 + padded_matrix.shape[0]] = padded_matrix
            # We add vector of read access
            for index, read_access_dict in enumerate(comp_dict["accesses"]):
                read_access_matrix = pad_access_matrix(
                    np.array(read_access_dict["access_matrix"]), max_depth
                ).reshape(-1)
                read_access_matrix = np.append(
                    read_access_matrix, +read_access_dict["access_is_reduction"]
                )
                read_access_matrix = np.append(
                    read_access_matrix, read_access_dict["buffer_id"] + 1
                )
                read_access_size = read_access_matrix.shape[0]
                single_comp_vector[
                    BUFFER_ACCESS_EMBEDDING_START
                    + index * read_access_size : BUFFER_ACCESS_EMBEDDING_START
                    + (index + 1) * read_access_size
                ] = read_access_matrix
            dict_comp[comp] = single_comp_vector
        return dict_comp

    @property
    def graph(self):
        it_vector_dict = self._tree_to_iterator_vectors()
        comp_vector_dict = self._annotations_to_comps_vectors()
        it_index = {}
        comp_index = {}
        tree = self.tree
        num_iterators = len(tree.iterators)
        for i, iterator_id in enumerate(it_vector_dict):
            it_index[iterator_id] = i
        for i, comp in enumerate(comp_vector_dict):
            comp_index[comp] = i

        edge_index = []
        node_feats = None

        for iterator_id in tree.iterators:
            iterator_node = tree.iterators[iterator_id]
            for child_it in iterator_node.child_iterators:
                edge_index.append([it_index[iterator_id], it_index[child_it]])

            for child_comp in iterator_node.computations_list:
                edge_index.append(
                    [it_index[iterator_id], num_iterators + comp_index[child_comp]]
                )
        node_feats = np.stack(
            [
                *[arr for arr in it_vector_dict.values()],
                *[arr for arr in comp_vector_dict.values()],
            ],
        )

        # focus on current branch
        for iterator in self.current_branch:
            index = it_index[iterator]
            node_feats[index][IteratorTags.FOCUS_TAG] = 1

        return node_feats, np.array(edge_index), it_index, comp_index

    def apply_action(self, action: int):
        done = False
        mask = self.get_mask()
        if np.all(mask):
            logger.warning("All actions are masked")
            return ApplyActionResult(is_legal=True, speedup=1, done=True, crashed=False)
        self.action_indices.append(action)
        tmp_schedule = self.schedule.copy()

        # if action is next action, move to next branch
        if action == NEXT_ACTION_INDEX:
            if self.current_branch_index + 1 >= len(self.branches):
                return ApplyActionResult(
                    is_legal=True, speedup=1, done=True, crashed=False
                )
            self.current_branch_index += 1
            return ApplyActionResult(
                is_legal=True, speedup=1, done=False, crashed=False
            )

        # if action is not next action, apply the action
        try:
            tmp_schedule.add_optimizations(
                [self.action_index_to_tiralib_action(action)]
            )
            is_legal = self.schedule_is_legal(tmp_schedule)
            if not is_legal:
                return ApplyActionResult(
                    is_legal=False, speedup=1, done=False, crashed=False
                )
            tmp_schedule_str = str(tmp_schedule)
            if self.cache and (
                cached_exec_time := self.cache.execution_time(
                    self.machine, tmp_schedule_str
                )
                is not None
            ):
                current_execution_time = cached_exec_time
            else:
                self.init_server()
                try:
                    current_execution_time = median_execution_time(tmp_schedule)
                except tiralib.function_server.ServerExecutionFailedError:
                    return ApplyActionResult(
                        is_legal=False, speedup=1, done=False, crashed=True
                    )
                if self.cache:
                    self.cache.add_execution_time(
                        self.machine, tmp_schedule_str, current_execution_time
                    )

            speedup = self.initial_execution_time / current_execution_time
            self.schedule = tmp_schedule

            # TODO Handle the depth of the tree dyamically or in a better way
            if self.schedule.tree.depth > MAX_ITERATOR_DEPTH:
                done = True

            return ApplyActionResult(
                is_legal=True, speedup=speedup, done=done, crashed=False
            )
        except Exception as e:
            logger.error(f"Error applying action {action}: {e}")
            return ApplyActionResult(
                is_legal=False, speedup=1, done=False, crashed=True
            )

    def action_index_to_tiralib_action(self, action_index: int):
        computation = self.current_branch[0][0]
        if (
            ActionSlices.INTERCHANGE.start
            <= action_index
            < ActionSlices.INTERCHANGE.stop
        ):
            level = action_index - ActionSlices.INTERCHANGE.start
            # check if level is in current branch we exclude the last iterator because the action uses
            # 2 successive iterators
            if level not in [iterator[1] for iterator in self.current_branch[:-1]]:
                raise ValueError(
                    f"Invalid level {level} for interchange: current branch {self.current_branch} and level {level}"
                )
            return tiralib.tiramisu_actions.Interchange(
                params=[(computation, level), (computation, level + 1)]
            )
        elif ActionSlices.REVERSAL.start <= action_index < ActionSlices.REVERSAL.stop:
            level = action_index - ActionSlices.REVERSAL.start
            if level not in [iterator[1] for iterator in self.current_branch]:
                raise ValueError(
                    f"Invalid level {level} for reversal: current branch {self.current_branch} and level {level}"
                )
            return tiralib.tiramisu_actions.Reversal(params=[(computation, level)])
        elif ActionSlices.SKEWING.start <= action_index < ActionSlices.SKEWING.stop:
            level = action_index - ActionSlices.SKEWING.start
            # action uses 2 successive iterators
            if level not in [iterator[1] for iterator in self.current_branch[:-1]]:
                raise ValueError(
                    f"Invalid level {level} for skewing: current branch {self.current_branch} and level {level}"
                )
            return tiralib.tiramisu_actions.Skewing(
                params=[(computation, level), (computation, level + 1), 0, 0]
            )
        elif (
            ActionSlices.PARALLELIZATION.start
            <= action_index
            < ActionSlices.PARALLELIZATION.stop
        ):
            level = action_index - ActionSlices.PARALLELIZATION.start
            if level not in [iterator[1] for iterator in self.current_branch]:
                raise ValueError(
                    f"Invalid level {level} for parallelization: current branch {self.current_branch} and level {level}"
                )
            return tiralib.tiramisu_actions.Parallelization(
                params=[(computation, level)]
            )
        elif ActionSlices.TILING2D.start <= action_index < ActionSlices.TILING2D.stop:
            level = (action_index - ActionSlices.TILING2D.start) % 4
            if level not in [iterator[1] for iterator in self.current_branch[:-1]]:
                raise ValueError(
                    f"Invalid level {level} for tiling2D: current branch {self.current_branch} and level {level}"
                )
            return tiralib.tiramisu_actions.Tiling2D(
                params=[
                    (computation, level),
                    (computation, level + 1),
                    *ActionSlices.tiling_size(action_index),
                ]
            )

        elif ActionSlices.UNROLLING.start <= action_index < ActionSlices.UNROLLING.stop:
            factor = action_index - ActionSlices.UNROLLING.start + 1
            # check if leaf of current branch does not have child iterators
            iterator_id = self.current_branch[-1]
            iterator = self.tree.get_iterator_of_computation(*iterator_id)
            if iterator.child_iterators:
                raise ValueError(
                    f"Cannot unroll iterator {iterator.name} with child iterators {iterator.child_iterators}"
                )
            return tiralib.tiramisu_actions.Unrolling(params=[iterator_id, 2**factor])
        else:
            raise ValueError(f"Invalid action index {action_index}")

    @property
    def annotations(self):
        return self.tiramisu_program.annotations

    def schedule_is_legal(self, schedule: tiralib.Schedule):
        """
        Checks if the schedule is legal.

        Returns
        -------
        Boolean indicating if the schedule is legal.
        """
        schedule_str = str(schedule)
        is_legal = None
        if self.cache:
            is_legal = self.cache.schedules_legality.get(schedule_str, None)
            isl_ast_str = self.cache.isl_ast.get(schedule_str, None)
            skewing_factors = self.cache.schedules_solver.get(schedule_str, None)

        if is_legal is None:
            if self.use_server:
                result = self.server.run("legality", schedule)
                is_legal: bool = result.legality
                isl_ast_str: str = result.isl_ast
                skewing_factors = None
                if (
                    result.additional_info
                    and "skewing_factors" in result.additional_info
                ):
                    skewing_factors = [
                        int(factor)
                        for factor in result.additional_info.replace(
                            "skewing_factors:", ""
                        ).split(",")
                    ]
                if self.cache:
                    self.cache.add(schedule_str, is_legal, isl_ast_str, skewing_factors)
            else:
                is_legal = schedule.is_legal(with_ast=True)
                isl_ast_str = schedule.tree.get_isl_ast_string()
                if (
                    schedule.optims_list[-1].is_skewing()
                    and schedule.optims_list[-1].params[2] == 0
                ):
                    skewing_action: tiralib.tiramisu_actions.Skewing = (
                        schedule.optims_list[-1]
                    )
                    copy_schedule = schedule.copy()
                    copy_schedule.optims_list.pop()
                    result = tiralib.tiramisu_actions.Skewing.get_factors(
                        copy_schedule,
                        [iterator[1] for iterator in skewing_action.iterators],
                        skewing_action.comps,
                    )
                    skewing_factors = list(result) if result else None

        schedule.legality = is_legal
        schedule.tree = tiralib.tiramisu_tree.TiramisuTree.from_isl_ast_string_list(
            isl_ast_string_list=isl_ast_str.split("\n")
        )
        # Update the skewing factors if they are not set
        if skewing_factors:
            for action in schedule.optims_list:
                if action.type == tiralib.tiramisu_actions.TiramisuActionType.SKEWING:
                    if action.params[2] == 0:
                        action.params[2] = skewing_factors[0]
                        action.params[3] = skewing_factors[1]
                        action.factors = skewing_factors
                        action.set_string_representations(schedule.tree)
        return is_legal


def median_execution_time(
    schedule: tiralib.Schedule,
    min_runs: int = 1,
    max_runs: int = 30,
    time_budget_in_seconds: int = 30,
):
    execution_times = schedule.execute(
        min_runs=min_runs,
        max_runs=max_runs,
        time_budget=time_budget_in_seconds * SECOND,
    )
    return float(np.median(execution_times))


class ActionSlices:
    INTERCHANGE = slice(0, 4)  # (0,1), (1,2), (2,3), (3,4)
    REVERSAL = slice(4, 9)  # 0, 1, 2, 3, 4
    SKEWING = slice(9, 12)  # (0,1), (1,2), (2,3)
    PARALLELIZATION = slice(12, 14)  # 0, 1
    TILING2D = slice(14, 50)  # (0,1), (1,2), (2,3), (3,4) *
    # [(32, 32), (64, 64), (128, 128), (32, 64), (32, 128), (64, 32), (64, 128), (128, 32), (128, 64)]
    UNROLLING = slice(50, 55)  # 2, 4, 8, 16, 32

    @classmethod
    def tiling_size(cls, action_index: int):
        size_dict = {
            (14, 18): (32, 32),
            (18, 22): (64, 64),
            (22, 26): (128, 128),
            (26, 30): (32, 64),
            (30, 34): (32, 128),
            (34, 38): (64, 32),
            (38, 42): (64, 128),
            (42, 46): (128, 32),
            (46, 50): (128, 64),
        }
        for start, stop in size_dict:
            if start <= action_index < stop:
                return size_dict[start, stop]

        raise ValueError(f"Invalid action index {action_index} for tiling2D")


# named tuple to hold the return of apply_action result
# is_legal: bool, is the schedule legal after applying the action
# speedup: float, the speedup after applying the action
# done: bool, is the schedule done
# crashed: bool, did the schedule crash after applying the action
ApplyActionResult = namedtuple(
    "ApplyActionResult", ["is_legal", "speedup", "done", "crashed"]
)


def program_compatible_with_model(annotations):
    max_accesses = 15
    min_accesses = 0
    max_iterators = MAX_ITERATOR_DEPTH
    computations_dict = annotations["computations"]

    # Making sure every computation doesn't exceed the limit of the cost model , if the model is updated change the conditions
    for comp_name in computations_dict:
        comp_dict = computations_dict[comp_name]
        if (
            len(comp_dict["accesses"]) > max_accesses
            or len(comp_dict["accesses"]) < min_accesses
        ):
            return False
        if len(comp_dict["iterators"]) > max_iterators:
            return False

    return True


class IteratorTags:
    """The tags of the iterator embeddings.

    The embedding is of size VECTOR_SIZE. The rest of the tags are set to -1 as padding."""

    TYPE_TAG = 0
    FOCUS_TAG = -11
    LOWER_BOUND_IS_INT_TAG = -10
    LOWER_BOUND_VALUE_TAG = -9
    UPPER_BOUND_IS_INT_TAG = -8
    UPPER_BOUND_VALUE_TAG = -7
    PARALLELIZATION_TAG = -6
    REVERSAL_TAG = -5
    UNROLLING_FACTOR_TAG = -4
    TILE_SIZE_TAG = -3
    SKEWING_FACTOR_1_TAG = -2
    SKEWING_FACTOR_2_TAG = -1


def _get_level_action_indices_tuple_actions(level: int, start_index: int):
    size_of_action = 3 if start_index == ActionSlices.SKEWING.start else 4
    if level > size_of_action:
        return []

    if level == 0:
        return [start_index]
    if level == size_of_action:
        return [start_index + level - 1]

    return [start_index + level - 1, start_index + level]
