from collections import namedtuple
import numpy as np
from ray import logger
import tiralib.tiramisu as tiralib
import tiralib.config as tiralib_config
from agent.graph_utils import encode_data_type, isl_to_write_matrix, pad_access_matrix


NEXT_ACTION_INDEX = 55
SECOND = 1000
MAX_ITERATOR_DEPTH = 5
VECTOR_SIZE = 718


class TiramisuInterface:
    def __init__(self, cpp_code: str, tiralib_config_path: str):
        tiralib_config.BaseConfig.init(tiralib_config_path)
        self.tiramisu_program = tiralib.TiramisuProgram.init_server(
            original_code=cpp_code,
            load_isl_ast=True,
            load_tree=True,
            load_annotations=True,
            reuse_server=True,
        )

        self.schedule = tiralib.Schedule(self.tiramisu_program)
        self.transformed_iterators: set[tiralib.IteratorIdentifier] = set()
        self.current_branch_index = 0
        self.action_indices: list[int] = []
        self.initial_execution_time = median_execution_time(self.schedule)
        # self.branches = self.schedule_branches

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

    def get_mask(self, mask_size: int = 56):
        mask = np.zeros(mask_size)

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

        # hide all previous actions
        for action_index in self.action_indices:
            if action_index == NEXT_ACTION_INDEX:
                continue
            mask[action_index] = 1

        if len(self.current_branch) == 1:
            mask[ActionSlices.INTERCHANGE] = 1
            mask[ActionSlices.SKEWING] = 1
            mask[ActionSlices.TILING2D] = 1

            # get the leaf iterator of the current branch
            iterator = self.tree.get_iterator_of_computation(
                self.current_branch[0][0], self.current_branch[0][1]
            )
            # if node has children then mask Unrolling
            if iterator.child_iterators:
                mask[ActionSlices.UNROLLING] = 1

        # mask levels that are not in current branch
        # TODO this is a temporary solution, we need to find a better way handle iterator depth
        levels = [level for level in range(MAX_ITERATOR_DEPTH)]
        for iterator in self.current_branch:
            levels.remove(iterator[1])

        for level in levels:
            mask[ActionSlices.REVERSAL.start + level] = 1

            # only 2 actions for parallelization
            if level < 2:
                mask[ActionSlices.PARALLELIZATION.start + level] = 1

            # interchange and tiling2d have actions that work on successive tuples
            # (0,1), (1,2), (2,3), etc.
            tuple_actions_start_indices = [
                ActionSlices.INTERCHANGE.start,
                ActionSlices.SKEWING.start,
            ]
            tuple_actions_start_indices.extend(
                [
                    i
                    for i in range(
                        ActionSlices.TILING2D.start, ActionSlices.TILING2D.stop, 4
                    )
                ]
            )
            for tuple_action_start_index in tuple_actions_start_indices:
                for action_index in self._get_level_action_indices_tuple_actions(
                    level, tuple_action_start_index
                ):
                    mask[action_index] = 1
        return mask

    def _get_level_action_indices_tuple_actions(self, level: int, start_index: int):
        size_of_action = 3 if start_index == ActionSlices.SKEWING.start else 4
        if level > size_of_action:
            return []

        if level == 0:
            return [start_index]
        if level == size_of_action:
            return [start_index + level - 1]

        return [start_index + level - 1, start_index + level]

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
            single_iter_vector[IteratorTags.LOWER_BOUND_IS_INT_TAG] = (
                1 if lower_bound_is_int else 0
            )
            single_iter_vector[IteratorTags.LOWER_BOUND_VALUE_TAG] = (
                schedule_tree.iterators[it].lower_bound
            )
            upper_bound_is_int = isinstance(
                schedule_tree.iterators[it].upper_bound, int
            )
            single_iter_vector[IteratorTags.UPPER_BOUND_IS_INT_TAG] = (
                1 if upper_bound_is_int else 0
            )
            single_iter_vector[IteratorTags.UPPER_BOUND_VALUE_TAG] = (
                schedule_tree.iterators[it].upper_bound
            )
            it_dict[it] = single_iter_vector

        return it_dict

    def _annotations_to_comps_vectors(self):
        annotations = self.tiramisu_program.annotations
        max_depth = MAX_ITERATOR_DEPTH
        dict_comp = {}
        for comp in annotations["computations"]:
            single_comp_vector = -np.ones(VECTOR_SIZE)
            # This means that this vector has data related to a computation and not an iterator
            single_comp_vector[0] = 1
            comp_dict = annotations["computations"][comp]
            # This field represents the absolute order of execution of computations
            single_comp_vector[1] = (
                self.tiramisu_program.tree.computations_absolute_order[comp]
            )
            # a vector of one-hot encoding of possible 3 data-types
            single_comp_vector[2:5] = encode_data_type(comp_dict["data_type"])
            single_comp_vector[5] = +comp_dict["comp_is_reduction"]
            # The write-to buffer id
            single_comp_vector[6] = +comp_dict["write_buffer_id"]
            # We add a vector of write access
            write_matrix = isl_to_write_matrix(comp_dict["write_access_relation"])
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
                    49 + index * read_access_size : 49 + (index + 1) * read_access_size
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
        if not tree:
            raise ValueError("Tree is not initialized")
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

        ## Removed because we are using the ISL tree to update the graph
        ## and we do not pass any information of the applied transformations to the model.
        # apply previous actions of the schedule
        # for optim in self.schedule.optims_list:
        #     match type(optim):
        #         case tiralib.tiramisu_actions.Interchange:
        #             iterator_1 = tree.get_iterator_of_computation(*optim.params[0])
        #             iterator_2 = tree.get_iterator_of_computation(*optim.params[1])
        #             it1 = it_index[iterator_1.name]
        #             it2 = it_index[iterator_2.name]
        #             for edge in edge_index:
        #                 if edge[0] == it1:
        #                     edge[0] = it2
        #                 elif edge[0] == it2:
        #                     edge[0] = it1
        #                 if edge[1] == it1:
        #                     edge[1] = it2
        #                 elif edge[1] == it2:
        #                     edge[1] = it1
        #         case tiralib.tiramisu_actions.Reversal:
        #             iterator = tree.get_iterator_of_computation(*optim.iterator_id)
        #             index = it_index[iterator.name]
        #             node_feats[index][-5] = 1
        #         case tiralib.tiramisu_actions.Skewing:
        #             for iterator_id in optim.iterators:
        #                 iterator = tree.get_iterator_of_computation(*iterator_id)
        #                 index = it_index[iterator.name]
        #                 node_feats[index][-2:] = optim.factors
        #         case tiralib.tiramisu_actions.Parallelization:
        #             iterator = tree.get_iterator_of_computation(*optim.iterator_id)
        #             index = it_index[iterator.name]
        #             node_feats[index][-6] = 1
        #         case tiralib.tiramisu_actions.Tiling2D:
        #             for iterator_id, tile_size in zip(
        #                 optim.iterators, optim.tile_sizes
        #             ):
        #                 iterator = tree.get_iterator_of_computation(*iterator_id)
        #                 index = it_index[iterator.name]
        #                 node_feats[index][-3] = tile_size
        #         case tiralib.tiramisu_actions.Unrolling:
        #             assert isinstance(optim, tiralib.tiramisu_actions.Unrolling)
        #             iterator = tree.get_iterator_of_computation(*optim.iterator_id)
        #             index = it_index[iterator.name]
        #             node_feats[index][-4] = optim.unrolling_factor
        #         case _:
        #             raise ValueError(f"Unsupported action {optim}")

        return node_feats, np.array(edge_index), it_index, comp_index

    def apply_action(self, action: int):
        mask = self.get_mask()
        if np.all(mask == 1):
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
            is_legal = tmp_schedule.is_legal(with_ast=True)
            if not is_legal:
                return ApplyActionResult(
                    is_legal=False, speedup=1, done=False, crashed=False
                )

            speedup = median_execution_time(tmp_schedule) / self.initial_execution_time
            self.schedule = tmp_schedule

            return ApplyActionResult(
                is_legal=True, speedup=speedup, done=False, crashed=False
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
            if level not in [iterator[1] for iterator in self.current_branch]:
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
            if level not in [iterator[1] for iterator in self.current_branch]:
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
            if level not in [iterator[1] for iterator in self.current_branch]:
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
            factor = action_index - ActionSlices.UNROLLING.start
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
    return np.median(execution_times)


class ActionSlices:
    INTERCHANGE = slice(0, 4)
    REVERSAL = slice(4, 9)
    SKEWING = slice(9, 12)
    PARALLELIZATION = slice(12, 14)
    TILING2D = slice(14, 50)
    UNROLLING = slice(50, 55)

    @classmethod
    def all_actions(cls):
        return (
            [
                cls.INTERCHANGE,
                cls.REVERSAL,
                cls.SKEWING,
                cls.PARALLELIZATION,
            ]
            + [slice(i, i + 4) for i in range(cls.TILING2D.start, cls.TILING2D.stop, 4)]
            + [cls.UNROLLING]
        )

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

    The embedding is of size 718. The rest of the tags are set to -1 as padding."""

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
