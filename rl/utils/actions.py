import numpy as np 

def apply_parallelization(iterator, node_feats, it_index):
    index = it_index[iterator]
    node_feats[index][-6] = 1


def apply_reversal(iterator, node_feats, it_index):
    index = it_index[iterator]
    node_feats[index][-5] = 1


def apply_unrolling(iterator, unrolling_factor, node_feats, it_index):
    index = it_index[iterator]
    node_feats[index][-4] = unrolling_factor


def apply_tiling(iterators, tile_sizes, node_feats, it_index):
    for it, tile in zip(iterators, tile_sizes):
        index = it_index[it]
        node_feats[index][-3] = tile


def apply_skewing(iterators, skewing_factors, node_feats, it_index):
    for it in iterators:
        index = it_index[it]
        node_feats[index][-2:] = skewing_factors


def apply_interchange(iterators, edge_index, it_index):
    it1, it2 = it_index[iterators[0]], it_index[iterators[1]]
    for edge in edge_index:
        if edge[0] == it1:
            edge[0] = it2
        elif edge[0] == it2:
            edge[0] = it1
        if edge[1] == it1:
            edge[1] = it2
        elif edge[1] == it2:
            edge[1] = it1


def focus_on_iterators(iterators, node_feats, it_index):
    # We reset the value for all the nodes
    node_feats[: len(it_index), -9:-8] = 0
    # We focus on the branches' iterators
    for it in iterators:
        index = it_index[it]
        node_feats[index][-9] = 1
    return node_feats



def apply_flattened_action(tiramisu_api, action, node_feats, edge_index, it_index):
    done = False
    if action < 4:
        loop_level = action
        # Interchange of loops (0,1) (1,2) (2,3) (3,4)
        (
            speedup,
            embedded_tensor,
            legality,
            actions_mask,
        ) = tiramisu_api.interchange(
            loop_level1=loop_level,
            loop_level2=loop_level + 1,
            env_id=action,
        )
        if legality:
            branch = tiramisu_api.scheduler_service.current_branch
            its = tiramisu_api.scheduler_service.branches[branch].common_it
            apply_interchange(
                [its[loop_level], its[loop_level + 1]], edge_index, it_index
            )
    elif action < 9:
        loop_level = action - 4
        # Reversal from 0 to 4
        (
            speedup,
            embedded_tensor,
            legality,
            actions_mask,
        ) = tiramisu_api.reverse(
            loop_level=loop_level,
            env_id=action,
        )
        if legality:
            branch = tiramisu_api.scheduler_service.current_branch
            its = tiramisu_api.scheduler_service.branches[branch].common_it
            apply_reversal(its[loop_level], node_feats, it_index)
    elif action < 12:
        loop_level = action - 9
        # Skewing 0,1 to 2,3
        speedup, embedded_tensor, legality, actions_mask = tiramisu_api.skew(
            loop_level1=loop_level,
            loop_level2=loop_level + 1,
            env_id=action,
        )

        if legality:
            skewing_params = (
                tiramisu_api.scheduler_service.schedule_object.schedule_list[
                    -1
                ].action.params[-2:]
            )
            branch = tiramisu_api.scheduler_service.current_branch
            its = tiramisu_api.scheduler_service.branches[branch].common_it
            apply_skewing(
                [its[loop_level], its[loop_level + 1]],
                skewing_params,
                node_feats,
                it_index,
            )
    elif action < 14:
        loop_level = action - 12
        # For parallelization 0 and 1
        (
            speedup,
            embedded_tensor,
            legality,
            actions_mask,
        ) = tiramisu_api.parallelize(
            loop_level=loop_level,
            env_id=action,
        )
        if legality:
            branch = tiramisu_api.scheduler_service.current_branch
            its = tiramisu_api.scheduler_service.branches[branch].common_it
            apply_parallelization(its[loop_level], node_feats, it_index)
    elif action < 18:
        loop_level = action - 14
        speedup, embedded_tensor, legality, actions_mask = tiramisu_api.tile2D(
            loop_level1=loop_level,
            loop_level2=loop_level + 1,
            size_x=128,
            size_y=64,
            env_id=action,
        )
        if legality:
            branch = tiramisu_api.scheduler_service.current_branch
            its = tiramisu_api.scheduler_service.branches[branch].common_it
            apply_tiling(
                [its[loop_level], its[loop_level + 1]], [128, 64], node_feats, it_index
            )
    elif action < 22:
        loop_level = action - 18
        speedup, embedded_tensor, legality, actions_mask = tiramisu_api.tile2D(
            loop_level1=loop_level,
            loop_level2=loop_level + 1,
            size_x=64,
            size_y=128,
            env_id=action,
        )
        if legality:
            branch = tiramisu_api.scheduler_service.current_branch
            its = tiramisu_api.scheduler_service.branches[branch].common_it
            apply_tiling(
                [its[loop_level], its[loop_level + 1]], [64, 128], node_feats, it_index
            )
    elif action < 26:
        loop_level = action - 22

        speedup, embedded_tensor, legality, actions_mask = tiramisu_api.tile2D(
            loop_level1=loop_level,
            loop_level2=loop_level + 1,
            size_x=64,
            size_y=64,
            env_id=action,
        )
        if legality:
            branch = tiramisu_api.scheduler_service.current_branch
            its = tiramisu_api.scheduler_service.branches[branch].common_it
            apply_tiling(
                [its[loop_level], its[loop_level + 1]], [64, 64], node_feats, it_index
            )
    elif action < 31:
        factor = action - 24
        speedup, embedded_tensor, legality, actions_mask = tiramisu_api.unroll(
            unrolling_factor=2**factor,
            env_id=action,
        )
        if legality:
            branch = tiramisu_api.scheduler_service.current_branch
            its = tiramisu_api.scheduler_service.branches[branch].common_it
            apply_unrolling(its[-1], 2**factor, node_feats, it_index)
    else:
        # Next case
        next_branch = tiramisu_api.scheduler_service.next_branch()
        if next_branch == None:
            speedup, embedded_tensor, legality, actions_mask = (
                1,
                None,
                True,
                np.zeros(32),
            )
            done = True
        else:
            speedup, embedded_tensor, legality, actions_mask = (
                1,
                next_branch[0],
                True,
                next_branch[1],
            )
            branch = tiramisu_api.scheduler_service.current_branch
            node_feats = focus_on_iterators(
                tiramisu_api.scheduler_service.branches[branch].common_it,
                node_feats,
                it_index,
            )

    return speedup, node_feats, edge_index, legality, actions_mask, done