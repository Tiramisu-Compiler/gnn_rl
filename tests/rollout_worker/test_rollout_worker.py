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


@patch("tiralib.tiramisu.schedule.Schedule.execute", return_value=[3.0, 2.0, 5.0])
def test_init_with_reset(_, dataset_actor):
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
@patch("tiralib.tiramisu.schedule.Schedule.execute")
def test_init_with_reset_remote_dataset_actor(_, dataset_actor):
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


@patch("tiralib.tiramisu.schedule.Schedule.execute")
def test_reward_process(_, dataset_actor):
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


@patch("tiralib.tiramisu.schedule.Schedule.execute", return_value=[2.0])
def test_rollout_with_model(_, gnn_model, dataset_actor):
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
@patch("tiralib.tiramisu.schedule.Schedule.execute", return_value=[2.0])
def test_rollout_with_model_clean_files(_, unlink, gnn_model, dataset_actor):
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
