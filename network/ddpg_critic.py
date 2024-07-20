import torch
from torch import nn 

class DDPGCritic(nn.Module):

    def __init__(self,args):
        super(DDPGCritic,self).__init__()

        self.args = args
        input_shape = args.input_shape
        n_action = args.n_actions

        self.stateNet = nn.Sequential(
            nn.Linear(input_shape,512),
            nn.ReLU(),
            nn.Linear(512,128),
            nn.ReLU(),
        )

        self.actionNet = nn.Sequential(
            nn.Linear(n_action,128),
            nn.ReLU(),
        )

        self.QNet = nn.Sequential(
            nn.Linear(256,512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,1)
        )
    
    def forward(self,state,action):
        
        stateProcess = self.stateNet(state)
        actionProcess = self.actionNet(action)
        ipt = torch.cat((stateProcess,actionProcess),dim=1)
        Qval = self.QNet(ipt)
        return Qval
