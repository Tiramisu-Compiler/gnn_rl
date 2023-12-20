
from rl.utils.replay_buffer import Transition
import torch
from torch_geometric.data import Data, Batch 

def update_weights(replay_buffer, target_dqn, train_dqn, criterion, optimizer, device, batch_size, lambdaa, tau):
    if len(replay_buffer) < batch_size:
        return

    sampled_data = replay_buffer.sample(batch_size)
    batch = Transition(*zip(*sampled_data))

    states_batch = []
    next_states_batch = []
    masked_states = []
    for entry in sampled_data:
        graph_entry = Data(
            x=torch.tensor(entry.state[0], dtype=torch.float32),
            edge_index=torch.tensor(entry.state[1], dtype=torch.int)
            .transpose(0, 1)
            .contiguous(),
        )
        states_batch.append(graph_entry)

        if entry.next_state == None:
            masked_states.append(False)
            graph_entry = Data(
                x=torch.zeros(1, 718, dtype=torch.float32),
                edge_index=torch.tensor([[0], [0]], dtype=torch.int).contiguous(),
            )
        else:
            graph_entry = Data(
                x=torch.tensor(entry.next_state[0], dtype=torch.float32),
                edge_index=torch.tensor(entry.next_state[1], dtype=torch.int)
                .transpose(0, 1)
                .contiguous(),
            )
            masked_states.append(True)
        next_states_batch.append(graph_entry)

    states = Batch.from_data_list(states_batch).to(device)
    next_states = Batch.from_data_list(next_states_batch).to(device)
    actions = torch.tensor(batch.action, dtype=torch.int64, device=device)
    rewards = torch.tensor(batch.reward, dtype=torch.float32, device=device)

    train_dqn.train()
    estimated_values = train_dqn(states).gather(1, actions.view(-1, 1))

    mask = torch.tensor(masked_states, device=device).float()
    target_dqn.eval()
    with torch.no_grad():
        target_values = lambdaa * mask * target_dqn(next_states).max(-1).values.view(-1)

    target_values = (rewards + target_values).view(-1, 1)

    optimizer.zero_grad()
    loss = criterion(target_values, estimated_values)
    loss.backward()
    # torch.nn.utils.clip_grad_value_(train_dqn.parameters(), 20)
    optimizer.step()

    target_net_state_dict = target_dqn.state_dict()
    policy_net_state_dict = train_dqn.state_dict()
    for key in policy_net_state_dict:
        target_net_state_dict[key] = policy_net_state_dict[
            key
        ] * tau + target_net_state_dict[key] * (1 - tau)
    target_dqn.load_state_dict(target_net_state_dict)

    return loss.item()