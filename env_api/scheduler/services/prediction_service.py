from env_api.core.services.compiling_service import CompilingService
from env_api.scheduler.models.schedule import Schedule
from env_api.utils.exceptions import ExecutingFunctionException
import subprocess
from config.config import Config




class PredictionService:
    def get_initial_time(self, schedule_object: Schedule):

        INIT_TIMEOUT = 5 * Config.config.experiment.max_time_in_minutes * 60 + 4 # number of executions * minutes * seconds + seconds of starting the script

        if "initial_execution" in schedule_object.prog.execution_times:
            # Original execution time of the program already exists in the dataset so we read the value directly
            initial_execution = schedule_object.prog.execution_times[
                "initial_execution"
            ]
        else:
            try:
                # We need to run the program to get the value
                if Config.config.test.skip_execute_schedules :
                    initial_execution = 1
                else:
                    initial_execution = CompilingService.execute_code(
                        tiramisu_program=schedule_object.prog,
                        optims_list=[],
                        timeout=INIT_TIMEOUT,
                    )
                if initial_execution:
                    schedule_object.prog.execution_times[
                        "initial_execution"
                    ] = initial_execution
                else:
                    raise ExecutingFunctionException
            except subprocess.TimeoutExpired as e:
                schedule_object.prog.execution_times["initial_execution"] = None
                return None
            except ExecutingFunctionException as e :
                return None
            except Exception as e :
                return None

        return initial_execution

    def get_real_speedup(self, schedule_object: Schedule):

        SLOWDOWN_TIMEOUT = Config.config.experiment.max_slowdown

        
        initial_execution = self.get_initial_time(schedule_object)

        if schedule_object.schedule_str in schedule_object.prog.execution_times:
            schedule_execution = schedule_object.prog.execution_times[
                schedule_object.schedule_str
            ]

        else:
            try:
                # We need to run the program to get the value
                if Config.config.test.skip_execute_schedules :
                    schedule_execution = 1
                else:
                    schedule_execution = CompilingService.execute_code(
                        tiramisu_program=schedule_object.prog,
                        optims_list=schedule_object.schedule_list,
                        timeout=((initial_execution / 1000) * SLOWDOWN_TIMEOUT) * 5 + 4
                    )
                if schedule_execution:
                    schedule_object.prog.execution_times[
                        schedule_object.schedule_str
                    ] = schedule_execution
                else:
                    raise ExecutingFunctionException
                
            except subprocess.TimeoutExpired as e:
                schedule_execution = initial_execution * SLOWDOWN_TIMEOUT
            
            except Exception as e :
                raise ExecutingFunctionException
            
        return initial_execution / schedule_execution
