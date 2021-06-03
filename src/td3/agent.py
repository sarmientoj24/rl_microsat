from src.commons.base_networks import BaseQNetwork
from src.td3.policy_network import PolicyNetwork
from src.commons.utils import ReplayBuffer
from src.commons.trainer import BaseAgent
import torch
import torch.nn.functional as F
import numpy as np


class Agent(BaseAgent):
    def __init__(self, alpha=0.0001, state_dim=50, env=None, gamma=0.995,
            action_dim=4, action_range=1, max_size=1000000, tau=1e-2,
            hidden_dim=128, batch_size=512, reward_scale=1, policy_target_update_interval=1,
            device='cpu'):
        super(Agent, self).__init__(batch_size=batch_size, hidden_dim=hidden_dim, reward_scale=reward_scale)

        self.policy_net = PolicyNetwork(alpha, state_dim, action_dim=action_dim, device=device,
                    name='policy_net', hidden_dim=hidden_dim, action_range=action_range, method='td3')
        self.target_policy_net = PolicyNetwork(alpha, state_dim, action_dim=action_dim, device=device,
                    name='target_policy_net', hidden_dim=hidden_dim, action_range=action_range, method='td3')
        self.q_net1 = BaseQNetwork(alpha, state_dim, action_dim=action_dim, hidden_dim=hidden_dim, 
                    name='q_net1', method='td3', device=device,)
        self.q_net2 = BaseQNetwork(alpha, state_dim, action_dim=action_dim, hidden_dim=hidden_dim, 
                    name='q_net2', method='td3', device=device,)
        self.target_q_net1 = BaseQNetwork(alpha, state_dim, action_dim=action_dim, hidden_dim=hidden_dim, 
                    name='target_q_net1', method='td3', device=device,)
        self.target_q_net2 = BaseQNetwork(alpha, state_dim, action_dim=action_dim, hidden_dim=hidden_dim, 
                    name='target_q_net2', method='td3', device=device,)

        self.reward_scale = reward_scale
        self.target_q_net1 = self.update_network_parameters(
            self.q_net1, self.target_q_net1,
            soft_tau=1
        )

        self.target_q_net2 = self.update_network_parameters(
            self.q_net2, self.target_q_net2,
            soft_tau=1
        )

        self.policy_net = self.update_network_parameters(
            self.policy_net, self.target_policy_net,
            soft_tau=1
        )

        self.update_cnt = 0
        self.policy_target_update_interval = policy_target_update_interval

    def choose_action(self, state, deterministic=True, explore_noise_scale=0.5):
        return self.policy_net.choose_action(
            state, deterministic=deterministic, explore_noise_scale=explore_noise_scale)

    def remember(self, state, action, reward, new_state, done):
        self.memory.push(state, action, reward, new_state, done)

    def save_models(self):
        print('.... saving models ....')
        self.policy_net.save_checkpoint()
        self.target_q_net1.save_checkpoint()
        self.target_q_net2.save_checkpoint()
        self.q_net1.save_checkpoint()
        self.q_net2.save_checkpoint()

    def load_models(self):
        print('.... loading models ....')
        self.policy_net.load_checkpoint()
        self.target_q_net1.load_checkpoint()
        self.target_q_net2.load_checkpoint()
        self.q_net1.load_checkpoint()
        self.q_net2.load_checkpoint()

    def learn(self, eval_noise_scale, deterministic=True, debug=False):
        state, action, reward, new_state, done = \
                self.memory.sample(self.batch_size)

        reward = torch.tensor(reward, dtype=torch.float).unsqueeze(1).to(self.policy_net.device)
        done = torch.tensor(np.float32(done)).unsqueeze(1).to(self.policy_net.device)
        next_state = torch.tensor(new_state, dtype=torch.float).to(self.policy_net.device)
        state = torch.tensor(state, dtype=torch.float).to(self.policy_net.device)
        action = torch.tensor(action, dtype=torch.float).to(self.policy_net.device)

        predicted_q_value1 = self.q_net1(state, action)
        predicted_q_value2 = self.q_net2(state, action)
        new_action, log_prob = self.policy_net.sample_normal(state, deterministic, eval_noise_scale=0.0)  # no noise, deterministic policy gradients
        new_next_action, _ = self.target_policy_net.sample_normal(next_state, deterministic, eval_noise_scale=eval_noise_scale) # clipped normal noise

        reward = self.reward_scale * (reward - reward.mean(dim=0)) / (reward.std(dim=0) + 1e-6) # normalize with batch mean and std; plus a small number to prevent numerical problem

        # Training Q Function
        target_q_min = torch.min(self.target_q_net1(next_state, new_next_action),self.target_q_net2(next_state, new_next_action))
        target_q_value = reward + (1 - done) * self.gamma * target_q_min # if done==1, only reward
        q_value_loss1 = ((predicted_q_value1 - target_q_value.detach())**2).mean()  # detach: no gradients for the variable
        q_value_loss2 = ((predicted_q_value2 - target_q_value.detach())**2).mean()
        self.q_net1.optimizer.zero_grad()
        q_value_loss1.backward()
        self.q_net1.optimizer.step()
        self.q_net2.optimizer.zero_grad()
        q_value_loss2.backward()
        self.q_net2.optimizer.step()

        policy_loss = None
        if self.update_cnt % self.policy_target_update_interval == 0:
        # Training Policy Function
            predicted_new_q_value = self.q_net1(state, new_action)
            policy_loss = - predicted_new_q_value.mean()
            self.policy_net.optimizer.zero_grad()
            policy_loss.backward()
            self.policy_net.optimizer.step()
        
        # Soft update the target nets
            self.target_q_net1 = self.update_network_parameters(self.q_net1, self.target_q_net1)
            self.target_q_net2 = self.update_network_parameters(self.q_net2, self.target_q_net2)
            self.target_policy_net = self.update_network_parameters(self.policy_net, self.target_policy_net)

        self.update_cnt+=1
        
        if debug:
            print('q loss: ', q_value_loss1, q_value_loss2)
        return (q_value_loss1, q_value_loss2, policy_loss)