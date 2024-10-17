from env_api.core.services.compiling_service import CompilingService
from env_api.scheduler.models.schedule import Schedule
from env_api.utils.exceptions import ExecutingFunctionException
import subprocess
from config.config import Config


class PredictionService:
    def get_initial_time(self, schedule_object: Schedule, worker_id: str):
        INIT_TIMEOUT = (
            5 * Config.config.experiment.max_time_in_minutes * 60 + 4
        )  # number of executions * minutes * seconds + seconds of starting the script

        initial_execution = schedule_object.prog.get_execution_time(
            "initial_execution", Config.config.machine
        )
        if initial_execution is None:
            try:
                # We need to run the program to get the value
                if Config.config.test.skip_execute_schedules:
                    initial_execution = 1
                else:
                    initial_execution = CompilingService.execute_code(
                        tiramisu_program=schedule_object.prog,
                        optims_list=[],
                        timeout=INIT_TIMEOUT,
                        worker_id=worker_id,
                    )
                if initial_execution:
                    schedule_object.prog.execution_times[Config.config.machine][
                        "initial_execution"
                    ] = initial_execution
                else:
                    raise ExecutingFunctionException
            except subprocess.TimeoutExpired:
                return None
            except ExecutingFunctionException:
                return None
            except Exception:
                return None

        return initial_execution

    def get_real_speedup(self, schedule_object: Schedule, worker_id):
        SLOWDOWN_TIMEOUT = Config.config.experiment.max_slowdown

        initial_execution = self.get_initial_time(schedule_object, worker_id)

        schedule_execution = schedule_object.prog.get_execution_time(
            schedule_object.schedule_str, Config.config.machine
        )
        if schedule_execution is None:
            try:
                # We need to run the program to get the value
                if Config.config.test.skip_execute_schedules:
                    schedule_execution = 1
                else:
                    schedule_execution = CompilingService.execute_code(
                        tiramisu_program=schedule_object.prog,
                        optims_list=schedule_object.schedule_list,
                        worker_id=worker_id,
                        timeout=((initial_execution / 1000) * SLOWDOWN_TIMEOUT) * 5 + 4,
                    )

                if schedule_execution:
                    schedule_object.prog.execution_times[Config.config.machine][
                        schedule_object.schedule_str
                    ] = schedule_execution
                else:
                    raise ExecutingFunctionException

            except subprocess.TimeoutExpired:
                schedule_execution = initial_execution * SLOWDOWN_TIMEOUT

            except Exception:
                raise ExecutingFunctionException

        return initial_execution / schedule_execution
