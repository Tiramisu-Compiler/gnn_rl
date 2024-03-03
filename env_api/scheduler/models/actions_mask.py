import numpy as np
from env_api.scheduler.models.action import *
import math 
import torch 

class ActionsMask:
    def __init__(
        self, num_types=7, loop_levels = 4, tiling_params = 9, unrolling_params = 3, interchange_loops = 10
    ):
        self.loop_levels = loop_levels
        self.types_mask = np.zeros(num_types)
        self.loop_param_actions = np.zeros((3, loop_levels))
        self.tiling_mask = np.zeros((loop_levels, tiling_params))
        self.interchange_mask = np.zeros(interchange_loops)
        self.unrolling_mask = np.zeros(unrolling_params)

    def to(self, device):
        dup = ActionsMask()
        dup.loop_param_actions = torch.Tensor(self.loop_param_actions).to(device)
        dup.types_mask = torch.Tensor(self.types_mask).to(device)
        dup.tiling_mask = torch.Tensor(self.tiling_mask).to(device)
        dup.interchange_mask = torch.Tensor(self.interchange_mask).to(device)
        dup.unrolling_mask = torch.Tensor(self.unrolling_mask).to(device)
        return dup
        

    def update_mask(self, action : Action, applied : bool = True):
        skewing_index = 0 
        reversal_index = 1 
        parallel_index = 2 
        interch_index = 3 
        unrolling_index = 4
        tiling_index = 5 
        if isinstance(action, Skewing): 
            loop_level1, loop_level2 = action.params[:2]
            self.loop_param_actions[skewing_index][loop_level1] = 1 
            self.types_mask[skewing_index] = int(np.all(self.loop_param_actions[skewing_index]))

            if applied : 
                # Do not allow any interchange to happen to the skewed loops : 
                indices = [self.combination_index(l1, l2) for  l1, l2 in self.combination_list(loop_level1, loop_level2)]
                self.interchange_mask[indices] = 1
                self.types_mask[interch_index] = int(np.all(self.interchange_mask))

                # Do not allow tilings to happen on the skewed loops : 
                self.tiling_mask[loop_level1][:] = 1 
                if loop_level2 < self.loop_levels:
                    self.tiling_mask[loop_level2][:] = 1
                if loop_level1 > 0 : 
                    self.tiling_mask[loop_level1-1][:] = 1

                self.types_mask[tiling_index] = int(np.all(self.tiling_mask))

                



        if isinstance(action, Reversal): 
            loop_level = action.params[0]
            self.loop_param_actions[reversal_index][loop_level] = 1 
            self.types_mask[reversal_index] = int(np.all(self.loop_param_actions[reversal_index]))

        
        if isinstance(action, Interchange):
            loop_level1, loop_level2 = action.params
            i = self.combination_index(loop_level1, loop_level2)
            self.interchange_mask[i] = 1 
            self.types_mask[interch_index] = int(np.all(self.interchange_mask))

        if isinstance(action, Parallelization):
            loop_level = action.params[0]
            self.loop_param_actions[parallel_index][loop_level] = 1 
            self.types_mask[parallel_index] = int(np.all(self.loop_param_actions[parallel_index]))
            if applied : 
                self.types_mask[skewing_index] = 1 
                self.types_mask[reversal_index] = 1 
                self.types_mask[interch_index] = 1 

        if isinstance(action, Tiling):
            loop_level1, loop_level2, size_x, size_y = action.params 
            idx = self.tiling_index(size_x, size_y)
            self.tiling_mask[loop_level1][idx] = 1
            self.types_mask[tiling_index] = int(np.all(self.tiling_mask))
            
            if applied : 
                self.types_mask[skewing_index] = 1 
                self.types_mask[parallel_index] = 1 
                self.types_mask[reversal_index] = 1 
                self.types_mask[interch_index] = 1 
                self.types_mask[tiling_index] = 1 

            

        if isinstance(action, Unrolling):
            _, factor = action.params
            idx = int(math.log(factor,2)) - 4
            self.unrolling_mask[idx] = 1 

            if applied : 
                self.types_mask[:6] = 1 
            




    def combination_list(self, l1, l2):
        comb = []
        for i in range(self.loop_levels):
            for j in range(i+1, self.loop_levels+1):
                if i == l1 or i == l2 or j == l1 or j ==l2:
                    comb.append((i,j))

        return comb

    def combination_index(self, a , b):
        if a > b : a,b = b,a
        if a == 0 : 
            return b-1
        if a == 1 : 
            return 2 + b
        if a == 2 : 
            return 4 + b
        if a == 3 : 
            return 9

    def tiling_index(self, x, y ):
        a = int(x >= 32) + int(x >= 64) + int(x >= 128) - 1 
        b = int(y >= 32) + int(y >= 64) + int(y >= 128) - 1

        return a * 3 + b