from torch_geometric.nn import (
    GATConv,
    global_mean_pool,
    Linear,
    global_max_pool,
)
import torch
import torch.nn as nn


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
        self.shared_linear2 = Linear(in_channels=hidden_size, out_channels=hidden_size)
        self.shared_linear3 = Linear(in_channels=hidden_size, out_channels=hidden_size)

        shared_layers = [
            self.shared_linear1,
            self.shared_linear2,
            self.shared_linear3,
        ]
        for child in shared_layers:
            if isinstance(child, Linear):
                nn.init.xavier_uniform_(child.weight)

        self.regression_layer = Linear(
            in_channels=hidden_size, out_channels=num_outputs
        )
        nn.init.xavier_uniform_(self.regression_layer.weight)

    def forward(self, data):
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
        x = nn.functional.selu(self.shared_linear2(x))
        x = nn.functional.selu(self.shared_linear3(x))

        x = self.regression_layer(x)

        return x

