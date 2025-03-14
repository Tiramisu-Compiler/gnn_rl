from unittest.mock import MagicMock, patch

from agent.rollout_worker import RolloutWorker, TimePerStage
from agent.tiramisu_interface import ApplyActionResult
from config.config import Config


@patch("agent.rollout_worker.RolloutWorker.reset")
def test_init(reset, dataset_actor):
    worker = RolloutWorker(dataset_worker=dataset_actor, config=Config.config)
    assert worker.dataset_worker == dataset_actor
    assert worker.tiramisu_interface is None
    assert worker.tiralib_config_path == Config.config.tiralib_config_path
    assert worker.worker_id == 0
    assert worker.current_program is None
    assert worker.times_per_stages == TimePerStage()
    assert worker.state is None
    assert worker.previous_speedup is None
    reset.assert_called_once()


def test_init_with_reset(dataset_actor):
    worker = RolloutWorker(
        dataset_worker=dataset_actor,
        config=Config.config,
        function_name="function_cvtcolor_MEDIUM",
    )
    assert worker.dataset_worker == dataset_actor
    assert worker.current_program == "function_cvtcolor_MEDIUM"
    assert worker.tiramisu_interface
    assert worker.state


@patch("ray.get", lambda x: x)
def test_init_with_reset_remote_dataset_actor(dataset_actor):
    remote_dataset_actor = MagicMock()
    remote_dataset_actor.get_next_function.remote.return_value = (
        dataset_actor.get_function_by_name("function_cvtcolor_MEDIUM")
    )

    worker = RolloutWorker(
        dataset_worker=remote_dataset_actor,
        config=Config.config,
    )
    assert worker.dataset_worker == remote_dataset_actor
    assert worker.current_program == "function_cvtcolor_MEDIUM"
    assert worker.tiramisu_interface
    assert worker.state


def test_reward_process(dataset_actor):
    worker = RolloutWorker(
        dataset_worker=dataset_actor,
        config=Config.config,
        function_name="function_cvtcolor_MEDIUM",
    )
    worker.previous_speedup = 1
    assert worker.reward_process(0, True, 2) == 0.5
    assert worker.previous_speedup == 2

    assert worker.reward_process(55, True, 2) == 0
    assert worker.previous_speedup == 2

    assert worker.reward_process(0, False, 2) == 0
    assert worker.previous_speedup == 2


# def rollout(self, model: nn.Module, device: str):
#     rollout_start_time = time()
#     start_time = rollout_start_time
#     model.to(device)
#     model.eval()
#     trajectory = []
#     done = False
#     log_trajectory = "#" * 50
#     log_trajectory += f"\nFunction  : {self.current_program}"
#     print("\n")
#     print("#" * 50)
#     print(f"Function : {self.current_program}")
#     print(f"trajectory : {trajectory}")
#     end_time = time()
#     self.times_per_stages.model_eval = end_time - start_time
#     start_time = end_time
#     while not done:
#         prev_actions_mask = self.tiramisu_interface.get_mask()
#         end_time = time()
#         self.times_per_stages.get_mask.append(end_time - start_time)
#         start_time = end_time
#         self.steps += 1
#         (node_feats, edge_index, it_index) = self.state
#         data = Data(
#             x=torch.tensor(node_feats, dtype=torch.float32),
#             edge_index=torch.tensor(edge_index, dtype=torch.int)
#             .transpose(0, 1)
#             .contiguous(),
#         ).to(device)

#         with torch.no_grad():
#             action, action_log_prob, entropy, value = model(
#                 data, torch.tensor(prev_actions_mask).to(device)
#             )
#             action = action.item()
#             action_log_prob = action_log_prob.item()
#             value = value.item()
#         end_time = time()
#         self.times_per_stages.get_action.append(end_time - start_time)
#         start_time = end_time

#         print(f"Running Action : {action}")

#         result = self.tiramisu_interface.apply_action(action)
#         end_time = time()
#         self.times_per_stages.apply_action.append(end_time - start_time)
#         start_time = end_time

#         done = result.done
#         if result.crashed:
#             logger.info("Crashed applying the action. Skipping this action")
#             continue

#         reward = self.reward_process(action, result.is_legal, result.speedup)
#         end_time = time()
#         self.times_per_stages.reward_process.append(end_time - start_time)
#         start_time = end_time

#         trajectory.append(
#             (
#                 (np.copy(node_feats), np.copy(edge_index)),
#                 action,
#                 reward,
#                 value,
#                 action_log_prob,
#                 entropy,
#                 prev_actions_mask,
#             )
#         )
#         if self.steps == 40:
#             done = True

#         if not done:
#             new_node_feats, new_edge_index, it_index, _ = (
#                 self.tiramisu_interface.graph
#             )

#             self.state = (new_node_feats, new_edge_index, it_index)
#             end_time = time()
#             self.times_per_stages.get_next_state.append(end_time - start_time)
#         start_time = time()

#         current_log = (
#             f"\nStep : {self.steps}"
#             + f"\nAction ID : {action}"
#             + f"\nLegality : {result.is_legal}"
#             + f"\nActions Sequence So far : {self.tiramisu_interface.action_indices}"
#             + "\n"
#         )
#         print(current_log)
#         log_trajectory += current_log

#     else:
#         start_time = time()
#         self.times_per_stages.rollout = start_time - rollout_start_time
#         if self.tiramisu_interface.cache:
#             if type(self.dataset_worker) is DatasetActor:
#                 self.dataset_worker.update_dataset(
#                     self.current_program, self.tiramisu_interface.cache.to_dict()
#                 )
#             else:
#                 ray.get(
#                     self.dataset_worker.update_dataset.remote(
#                         self.current_program,
#                         self.tiramisu_interface.cache.to_dict(),
#                     )
#                 )
#             self.times_per_stages.update_dataset = time() - start_time

#     # clean up created files
#     # delete files with the filename in workspace
#     if Config.config.clean_files:
#         for filename in Path(Config.config.tiramisu.workspace).glob("*"):
#             if self.current_program in filename.name:
#                 filename.unlink()
#     print(f"End of episode : {self.current_program}")
#     print(f"Schedule : {self.tiramisu_interface.schedule}")
#     print(f"actions : {self.tiramisu_interface.action_indices}")
#     print(f"Speedup : {self.previous_speedup}")
#     print(f"Time per stage for {self.current_program} : {self.times_per_stages}")
#     return RolloutResult(
#         self.current_program,
#         trajectory,
#         self.previous_speedup,
#         str(self.tiramisu_interface.schedule),
#         log_trajectory,
#     )


def test_rollout_with_model(gnn_model, dataset_actor):
    worker = RolloutWorker(
        dataset_worker=dataset_actor,
        config=Config.config,
        function_name="function_cvtcolor_MEDIUM",
    )

    result = worker.rollout(gnn_model, "cpu")
    assert result.function_name == "function_cvtcolor_MEDIUM"
    assert result.trajectory
    assert result.speedup
    assert result.schedule
    assert result.log_trajectory


@patch(
    "agent.tiramisu_interface.TiramisuInterface.apply_action",
    return_value=ApplyActionResult(
        done=True,
        crashed=True,
        speedup=0,
        is_legal=True,
    ),
)
@patch("agent.rollout_worker.RolloutWorker.reward_process")
def test_rollout_action_crashes(reward, apply_action, dataset_actor):
    worker = RolloutWorker(
        dataset_worker=dataset_actor,
        config=Config.config,
        function_name="function_cvtcolor_MEDIUM",
    )

    model = MagicMock()
    model.to = MagicMock()
    model.eval = MagicMock()
    action = MagicMock()
    action.item = MagicMock(return_value=0)
    model.return_value = (action, MagicMock(), MagicMock(), MagicMock())

    result = worker.rollout(model, "cpu")
    assert result.function_name == "function_cvtcolor_MEDIUM"
    assert reward.call_count == 0


@patch(
    "agent.tiramisu_interface.TiramisuInterface.apply_action",
    return_value=ApplyActionResult(
        done=False,
        crashed=False,
        speedup=2,
        is_legal=True,
    ),
)
def test_rollout_action_stopped_at_40(apply_action, dataset_actor):
    worker = RolloutWorker(
        dataset_worker=dataset_actor,
        config=Config.config,
        function_name="function_cvtcolor_MEDIUM",
    )

    model = MagicMock()
    model.to = MagicMock()
    model.eval = MagicMock()
    action = MagicMock()
    action.item = MagicMock(return_value=0)
    model.return_value = (action, MagicMock(), MagicMock(), MagicMock())

    result = worker.rollout(model, "cpu")
    assert result.function_name == "function_cvtcolor_MEDIUM"
    assert worker.steps == 40


@patch("pathlib.Path.unlink")
def test_rollout_with_model_clean_files(unlink, gnn_model, dataset_actor):
    Config.config.clean_files = True
    worker = RolloutWorker(
        dataset_worker=dataset_actor,
        config=Config.config,
        function_name="function_cvtcolor_MEDIUM",
    )

    result = worker.rollout(gnn_model, "cpu")
    assert result.function_name == "function_cvtcolor_MEDIUM"
    assert result.trajectory
    assert result.speedup
    assert result.schedule
    assert result.log_trajectory
    assert unlink.call_count > 0
