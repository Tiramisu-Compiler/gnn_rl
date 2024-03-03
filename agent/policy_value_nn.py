from torch_geometric.nn import (
    GATConv,
    global_mean_pool,
    Linear,
    global_max_pool,
)
import torch
import torch.nn as nn

from torch.distributions import Categorical
import numpy as np

from env_api.scheduler.models.actions_mask import ActionsMask



class GAT(nn.Module):
    def __init__(
        self,
        input_size=260,
        hidden_size=64,
        num_heads=4,
        num_outputs=32,
    ):
        super(GAT, self).__init__()

        self.conv_layer1 = GATConv(
            in_channels=input_size,
            out_channels=hidden_size,
            heads=num_heads,
        )
        self.linear1 = Linear(
            in_channels=hidden_size * num_heads,
            out_channels=hidden_size,
        )
        self.conv_layer2 = GATConv(
            in_channels=hidden_size,
            out_channels=hidden_size,
            heads=num_heads,
        )
        self.linear2 = Linear(
            in_channels=hidden_size * num_heads,
            out_channels=hidden_size,
        )

        convolutions_layers = [
            self.conv_layer1,
            self.conv_layer2,
        ]
        for convlayer in convolutions_layers:
            for name, param in convlayer.named_parameters():
                if "weight" in name:
                    nn.init.xavier_uniform_(param)
        linear_layers = [
            self.linear1,
            self.linear2,
        ]
        for linearlayer in linear_layers:
            nn.init.xavier_uniform_(linearlayer.weight)

        self.convs_summarizer = Linear(
            in_channels=4 * hidden_size,
            out_channels=hidden_size * 2,
        )

        self.shared_linear1 = Linear(
            in_channels=hidden_size * 2, out_channels=hidden_size
        )
        nn.init.xavier_uniform_(self.shared_linear1.weight)

        self.π = nn.Sequential(
            self.init_layer(nn.Linear(hidden_size, hidden_size)),
            nn.SELU(),
            self.init_layer(nn.Linear(hidden_size, hidden_size)),
            nn.SELU(),
            self.init_layer(nn.Linear(hidden_size, num_outputs), std=0.1),
        )

        self.v = nn.Sequential(
            self.init_layer(nn.Linear(hidden_size, hidden_size)),
            nn.SELU(),
            self.init_layer(nn.Linear(hidden_size, hidden_size)),
            nn.SELU(),
            self.init_layer(nn.Linear(hidden_size, 1)),
        )

    def init_layer(self, layer, std=np.sqrt(2), bias_const=0.0):
        nn.init.orthogonal_(layer.weight, std)
        nn.init.constant_(layer.bias, bias_const)
        return layer

    def shared_layers(self, data):
        x, edges_index, batch_index = (data.x, data.edge_index, data.batch)

        x = self.conv_layer1(x, edges_index)
        x = nn.functional.selu(self.linear1(x))
        x1 = torch.concat(
            (global_mean_pool(x, batch_index), global_max_pool(x, batch_index)), dim=-1
        )

        x = self.conv_layer2(x, edges_index)
        x = nn.functional.selu(self.linear2(x))
        x2 = torch.concat(
            (global_mean_pool(x, batch_index), global_max_pool(x, batch_index)), dim=-1
        )

        x = torch.concat(
            (
                x1,
                x2,
            ),
            dim=-1,
        )

        x = self.convs_summarizer(x)

        x = nn.functional.selu(self.shared_linear1(x))
        return x

    def forward(self, data, actions_mask=None, action=None):
        weights = self.shared_layers(data)
        logits = self.π(weights)
        if actions_mask != None:
            logits = logits - actions_mask * 1e10
        probs = Categorical(logits=logits)
        if action == None:
            action = probs.sample()
        value = self.v(weights)
        return action, probs.log_prob(action), probs.entropy(), value


class GATMultiDiscrete(nn.Module):
    def __init__(
        self,
        input_size=260,
        hidden_size=64,
        num_heads=4,
        **kwargs
    ):
        super(GATMultiDiscrete, self).__init__()

        self.actions_split = [*kwargs.values()]

        self.conv_layer1 = GATConv(
            in_channels=input_size,
            out_channels=hidden_size,
            heads=num_heads,
        )
        self.linear1 = Linear(
            in_channels=hidden_size * num_heads,
            out_channels=hidden_size,
        )
        self.conv_layer2 = GATConv(
            in_channels=hidden_size,
            out_channels=hidden_size,
            heads=num_heads,
        )
        self.linear2 = Linear(
            in_channels=hidden_size * num_heads,
            out_channels=hidden_size,
        )

        convolutions_layers = [
            self.conv_layer1,
            self.conv_layer2,
        ]
        for convlayer in convolutions_layers:
            for name, param in convlayer.named_parameters():
                if "weight" in name:
                    nn.init.xavier_uniform_(param)
        linear_layers = [
            self.linear1,
            self.linear2,
        ]
        for linearlayer in linear_layers:
            nn.init.xavier_uniform_(linearlayer.weight)

        self.convs_summarizer = Linear(
            in_channels=4 * hidden_size,
            out_channels=hidden_size * 2,
        )

        self.shared_linear1 = Linear(
            in_channels=hidden_size * 2, out_channels=hidden_size
        )
        nn.init.xavier_uniform_(self.shared_linear1.weight)

        self.π = nn.Sequential(
            self.init_layer(nn.Linear(hidden_size, hidden_size)),
            nn.SELU(),
            self.init_layer(nn.Linear(hidden_size, hidden_size)),
            nn.SELU(),
            self.init_layer(nn.Linear(hidden_size, np.sum(self.actions_split)), std=0.1),
        )

        self.v = nn.Sequential(
            self.init_layer(nn.Linear(hidden_size, hidden_size)),
            nn.SELU(),
            self.init_layer(nn.Linear(hidden_size, hidden_size)),
            nn.SELU(),
            self.init_layer(nn.Linear(hidden_size, 1)),
        )

    def init_layer(self, layer, std=np.sqrt(2), bias_const=0.0):
        nn.init.orthogonal_(layer.weight, std)
        nn.init.constant_(layer.bias, bias_const)
        return layer

    def shared_layers(self, data):
        x, edges_index, batch_index = (data.x, data.edge_index, data.batch)

        x = self.conv_layer1(x, edges_index)
        x = nn.functional.selu(self.linear1(x))
        x1 = torch.concat(
            (global_mean_pool(x, batch_index), global_max_pool(x, batch_index)), dim=-1
        )

        x = self.conv_layer2(x, edges_index)
        x = nn.functional.selu(self.linear2(x))
        x2 = torch.concat(
            (global_mean_pool(x, batch_index), global_max_pool(x, batch_index)), dim=-1
        )

        x = torch.concat(
            (
                x1,
                x2,
            ),
            dim=-1,
        )

        x = self.convs_summarizer(x)

        x = nn.functional.selu(self.shared_linear1(x))
        return x

    def forward(self, data, input_actions=None, action_mask : ActionsMask = None):
        mask_num = 1e8
        weights = self.shared_layers(data)
        raw_logits = self.π(weights)
        splitted_logits = torch.split(raw_logits , self.actions_split, dim=1)
        size = data.batch[-1].item() + 1 
        final_logprob = torch.zeros((size))
        final_entropy = torch.zeros((size))

        reduced_action = None
        if action_mask == None : 
            multi_categorical = [Categorical(logits=logits) for logits in splitted_logits]
            if input_actions == None : 
                raw_actions = torch.stack([categorical.sample() for categorical in multi_categorical])
            else : 
                raw_actions = torch.clone(input_actions)
            logprob = torch.stack([categorical.log_prob(a) for a, categorical in zip(raw_actions, multi_categorical)])
            entropy = torch.stack([categorical.entropy() for categorical in multi_categorical])
            
            for batch_index, act in enumerate(raw_actions[0]):
                if act < 3  : 
                    final_logprob[batch_index] = logprob[0][batch_index] + logprob[1][batch_index]
                    final_entropy[batch_index] = entropy[0][batch_index] + entropy[1][batch_index]
                elif act < 5  : 
                    a = act - 1
                    final_logprob[batch_index] = logprob[0][batch_index] + logprob[a][batch_index]
                    final_entropy[batch_index] = entropy[0][batch_index] + entropy[a][batch_index]
                    
                elif act == 5: 
                    final_logprob[batch_index] = logprob[0][batch_index] + logprob[1][batch_index] + logprob[4][batch_index]
                    final_entropy[batch_index] = entropy[0][batch_index] + entropy[1][batch_index] + entropy[4][batch_index]

                else : 
                    final_logprob[batch_index] = logprob[0][batch_index]
                    final_entropy[batch_index] = entropy[0][batch_index]

        else : 
            raw_actions = torch.zeros((5,1))
            type_distribution = Categorical(logits= splitted_logits[0] - action_mask.types_mask * mask_num)
            type = type_distribution.sample()
            raw_actions[0] = type
            if type < 3 : 
                level_distribution = Categorical(logits=splitted_logits[1] - action_mask.loop_param_actions[type] * mask_num)
                level = level_distribution.sample()
                raw_actions[1] = level
                reduced_action = [z.item() for z in [type, level]]
                final_logprob[0] = type_distribution.log_prob(type) + level_distribution.log_prob(level)
                final_entropy[0] = type_distribution.entropy() + level_distribution.entropy()
            elif type == 3 : 
                int_loops_dist = Categorical(logits=splitted_logits[2] - action_mask.interchange_mask * mask_num) 
                interchange_loops = int_loops_dist.sample()
                raw_actions[2] = interchange_loops
                reduced_action = [z.item() for z in [type,interchange_loops]]
                final_logprob[0] = type_distribution.log_prob(type) + int_loops_dist.log_prob(interchange_loops)
                final_entropy[0] = type_distribution.entropy() + int_loops_dist.entropy()
            elif type == 4 : 
                unrolling_dist = Categorical(logits=splitted_logits[3] - action_mask.unrolling_mask * mask_num)
                unrolling_factor = unrolling_dist.sample()
                raw_actions[3] = unrolling_factor
                reduced_action = [z.item() for z in [type,unrolling_factor]]
                final_logprob[0] = type_distribution.log_prob(type) + unrolling_dist.log_prob(unrolling_factor)
                final_entropy[0] = type_distribution.entropy() + unrolling_dist.entropy()
            elif type == 5 : 
                levels_mask = torch.all(action_mask.tiling_mask, dim=1)
                tiling_lvl_dist = Categorical(logits=splitted_logits[1] - levels_mask * mask_num)
                level = tiling_lvl_dist.sample()
                raw_actions[1] = level
                factors_dist = Categorical(logits=splitted_logits[4] - action_mask.tiling_mask[level] * mask_num)
                factor = factors_dist.sample()
                raw_actions[4] =  factor
                reduced_action  = [z.item() for z in [type,level,factor]]
                final_logprob[0] = type_distribution.log_prob(type) + tiling_lvl_dist.log_prob(level) + factors_dist.log_prob(factor)
                final_entropy[0] = type_distribution.entropy() + tiling_lvl_dist.entropy() + factors_dist.entropy()
            else : 
                reduced_action = [6]
                final_logprob[0] = type_distribution.log_prob(type)
                final_entropy[0] = type_distribution.entropy() 

        value = self.v(weights)
        return reduced_action, raw_actions, final_logprob, final_entropy, value
