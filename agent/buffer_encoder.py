import torch.nn as nn 
import torch 
class NN(nn.Module):
    def __init__(self) -> None:
        super(NN,self).__init__()
        self.encoder1 = nn.Linear(702,100)
        self.encoder2 = nn.Linear(100,20)
        self.decoder1 = nn.Linear(20,702)

    def forward(self, x):
        y = nn.functional.selu(self.encoder1(x))
        y = nn.functional.selu(self.encoder2(y))
        y = self.decoder1(y)
        return y 

encoder = NN()
encoder.load_state_dict(torch.load("./agent/rw_buf_encoder.pt"))
encoder = encoder.eval()