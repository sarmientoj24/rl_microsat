import torch.nn.functional as F
import torch
import torch.nn as nn
import os
import torch.optim as optim
from src.commons.utils import linear_weights_init


class BaseValueNetwork(nn.Module):
    def __init__(self, alpha=0.0001, state_dim=50, hidden_dim=128, init_w=3e-3, 
            name='value_net', chkpt_dir='./tmp/', method=''):
        super(BaseValueNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.q = nn.Linear(hidden_dim, 1)

        self.q.weight.data.uniform_(-init_w, init_w)
        self.q.bias.data.uniform_(-init_w, init_w)

        self.name = name
        self.method = method
        self.checkpoint_dir = chkpt_dir + method
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_' + method)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = 'cpu'

        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.q(x)
        return x

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))


class BaseQNetwork(nn.Module):
    def __init__(self, alpha=0.0001,  state_dim=50, action_dim=4, hidden_dim=128, init_w=3e-3, 
            name='q_net', chkpt_dir='./tmp/', method=''):
        super(BaseQNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.q = nn.Linear(hidden_dim, 1)
        
        self.q.weight.data.uniform_(-init_w, init_w)
        self.q.bias.data.uniform_(-init_w, init_w)

        self.name = name
        self.method = method
        self.checkpoint_dir = chkpt_dir + method
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_' + method)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = 'cpu'

        self.to(self.device)

    def forward(self, state, action):
        x = torch.cat([state, action], 1) # the dim 0 is number of samples
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.q(x)
        return x

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file, map_location=torch.device(self.device)))
