from env_api.tiramisu_api import TiramisuEnvAPI
from config.config import Config
from env_api.utils.exceptions import *
import random, traceback, time
from env_api.core.services.compiling_service import CompilingService
from agent.graph_utils import build_graph
from torch_geometric.data import Data
import torch
from agent.policy_value_nn import GAT

if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    start = time.time()
    # Init global config to run the Tiramisu env
    Config.init()
    t_api = TiramisuEnvAPI(local_dataset=True)
    # Get a list of the program names in the database
    programs = t_api.get_programs()
    try:
        # Select a program randomly for example program = "function025885"
        program: str = random.choice(programs)
        print("Selected f  unction : ", program)
        # set_program(str) creates all the necessary objects to start doing operations on a program
        # it returns an encoded representation specific to the RL system
        # This representation has a shape and type of torch.Size([180])
        # ast = t_api.initialize_optimzer(name=program)
        # print(t_api.optimizer.tiramisu_prog.original_str)
        # print(ast)
        # init_exec_time = t_api.optimizer.execute_original_code()
        # t_api.apply_tiling(loops=[0,1], sizes=[2,2])
        # ast.get_fusion_levels()


        actions_mask = t_api.set_program(name=program)
        # There is some programs that are not supported so we need to check our representation first
        if True:
            # After setting a program and checking if it is fully supported by our RL system, you can apply any action on it in any order
            # And expect to get the speedup of the whole schedule, the representation and the result of legality check of the last operation

            # (speedup, legality, actions_mask) = t_api.skew(
            #     loop_level1=0, loop_level2=1, env_id=2, worker_id="0"
            # )

            # (speedup,
            #  legality,actions_mask) = t_api.skew(loop_level1=0,loop_level2=1,env_id=2)
            # (
            #     speedup,
            #     legality,
            #     actions_mask,
            # ) = t_api.interchange(loop_level1=1,loop_level2=2, env_id=7)
            # (
            #     speedup,
            #     legality,
            #     actions_mask,
            # ) = t_api.parallelize(loop_level=1, env_id=1, worker_id = "test")
            # t_api.scheduler_service.next_branch()
            # (speedup, legality, actions_mask) = t_api.tile2D(
            #     loop_level1=1, loop_level2=2, size_x=6, size_y=6, env_id=4
            # )
            # (
            #     speedup,
            #     legality,
            #     actions_mask,
            # ) = t_api.reverse(loop_level=2, env_id=1)
            # n, e, i, c = build_graph(
            #     t_api.scheduler_service.schedule_object.prog.annotations
            # )

            # data = Data(
            #     x=torch.tensor(n, dtype=torch.float32),
            #     edge_index=torch.tensor(e, dtype=torch.int)
            #     .transpose(0, 1)
            #     .contiguous(),
            # )

            # ppo_agent = GAT(input_size=718, num_heads=4, hidden_size=198, num_outputs=56)

            # print(ppo_agent(data))

            # (
            #     speedup,
            #     legality,
            #     actions_mask,
            # ) = t_api.parallelize(loop_level=0, env_id=1)
            (speedup, legality, actions_mask,
            ) = t_api.unroll(unrolling_factor=10, env_id=7, worker_id="1")

            # t_api.scheduler_service.next_branch()
            # (speedup,
            #  legality,actions_mask) = t_api.tile2D(loop_level1=0,loop_level2=1,size_x=32,size_y=64,env_id=4)
            # (speedup, embedding_tensor, legality, actions_mask,
            # ) = t_api.unroll(unrolling_factor=16, env_id=7)

            # # (speedup, embedding_tensor,
            # # legality,actions_mask) = t_api.tile3D(loop_level1=0 , loop_level2=1,loop_level3=2,
            # #     size_x=128,size_y=128,size_z=128,env_id=17)
            print("Speedup : ", speedup, " ", "Legality : ", legality)

        print("Time : ", time.time() - start)
    except Exception as e:
        print("Traceback of the error : " + 60 * "-")
        print(traceback.print_exc())
        print(80 * "-")
