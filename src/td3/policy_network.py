import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import os
import numpy as np
from src.commons.base_networks import BasePolicyNetwork


class PolicyNetwork(BasePolicyNetwork):
    def __init__(self, alpha=0.0001, state_dim=50, action_dim=4, action_range=1, 
            log_std_min=-20, log_std_max=2, hidden_dim=128, init_w=3e-3, 
            name='policy', chkpt_dir='./tmp/', method=''):
        super(PolicyNetwork, self).__init__(method=method, hidden_dim=hidden_dim)
        self.to('cpu')

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        mean  = torch.tanh(self.mu(x))
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max) # clip the log_std into reasonable range
        
        return mean, log_std
    
    def sample_normal(self, state, deterministic=True, eval_noise_scale=0.5, epsilon=1e-6):
        '''
        generate sampled action with state as input wrt the policy network;
        deterministic evaluation provides better performance according to the original paper;
        '''
        mean, log_std = self.forward(state)
        std = log_std.exp() # no clip in evaluation, clip affects gradients flow
        
        normal = Normal(0, 1)
        z      = normal.sample() 
        action_0 = torch.tanh(mean + std*z.to('cpu')) # TanhNormal distribution as actions; reparameterization trick

        action = self.action_range*mean if deterministic else self.action_range*action_0

        log_prob = Normal(mean, std).log_prob(mean+ std*z.to('cpu')) - torch.log(1. - action_0.pow(2) + epsilon) -  np.log(self.action_range)
        # both dims of normal.log_prob and -log(1-a**2) are (N,dim_of_action); 
        # the Normal.log_prob outputs the same dim of input features instead of 1 dim probability, 
        # needs sum up across the features dim to get 1 dim prob; or else use Multivariate Normal.
        log_prob = log_prob.sum(dim=1, keepdim=True)
        ''' add noise '''
        eval_noise_clip = 2*eval_noise_scale
        noise = normal.sample(action.shape) * eval_noise_scale
        noise = torch.clamp(
                noise,
                -eval_noise_clip,
                eval_noise_clip)
        action = action + noise.to('cpu')

        return action, log_prob

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file, map_location=T.device('cpu')))

    def choose_action(self, state, deterministic=False, explore_noise_scale=0.5):
        state = torch.FloatTensor(state).unsqueeze(0).to('cpu')
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        normal = Normal(0, 1)
        z      = normal.sample().to('cpu')
        action = mean.detach().cpu().numpy()[0] if deterministic else torch.tanh(mean + std*z).detach().cpu().numpy()[0]
        ''' add noise '''
        noise = normal.sample(action.shape) * explore_noise_scale
        action = self.action_range * action + noise.numpy()

        return action

    def sample_action(self):
        action = torch.FloatTensor(self.action_dim).uniform_(-1, 1)
        action = self.action_range * action

        return (action).numpy()


