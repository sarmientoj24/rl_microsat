from src.commons.base_networks \
    import BaseValueNetwork as ValueNetwork, BaseQNetwork as QNetwork
from src.sac.policy_network import PolicyNetwork
from src.commons.utils import ReplayBuffer
import torch
import torch.nn.functional as F
import numpy as np
from src.commons.trainer import BaseAgent


class Agent(BaseAgent):
    def __init__(self, alpha=0.0001, state_dim=50, env=None, gamma=0.995,
            action_dim=4, action_range=1, max_size=1000000, tau=1e-2,
            hidden_dim=128, batch_size=256, reward_scale=1, device='cpu'):
        super(Agent, self).__init__(batch_size=batch_size, reward_scale=reward_scale)

        self.policy_net = PolicyNetwork(alpha, state_dim, action_dim=action_dim,
                    name='policy_net', hidden_dim=hidden_dim,
                    action_range=action_range, method='sac')
        self.q_net1 = QNetwork(alpha, state_dim, action_dim=action_dim,
                    name='q_net1', method='sac')
        self.q_net2 = QNetwork(alpha, state_dim, action_dim=action_dim,
                    name='q_net2', method='sac')
        self.value_net = ValueNetwork(alpha, state_dim, name='value', method='sac')
        self.target_value_net = ValueNetwork(alpha, state_dim, name='target_value', method='sac')

        self.target_value_net = self.update_network_parameters(
            self.value_net, self.target_value_net,
            soft_tau=1
        )

    def choose_action(self, state, deterministic=False):
        if self.memory.get_length() < self.batch_size:
            return self.policy_net.sample_action()
        else:
            return self.policy_net.choose_action(state, deterministic=deterministic)

    def remember(self, state, action, reward, new_state, done):
        self.memory.push(state, action, reward, new_state, done)

    def save_models(self):
        print('.... saving models ....')
        self.policy_net.save_checkpoint()
        self.value_net.save_checkpoint()
        self.target_value_net.save_checkpoint()
        self.q_net1.save_checkpoint()
        self.q_net2.save_checkpoint()

    def load_models(self):
        print('.... loading models ....')
        self.policy_net.load_checkpoint()
        self.value_net.load_checkpoint()
        self.target_value_net.load_checkpoint()
        self.q_net1.load_checkpoint()
        self.q_net2.load_checkpoint()


    def learn(self, deterministic=False, debug=False):
        alpha = 1.0

        state, action, reward, new_state, done = \
                self.memory.sample(self.batch_size)

        reward = torch.tensor(reward, dtype=torch.float).unsqueeze(1).to(self.policy_net.device)
        done = torch.tensor(np.float32(done)).unsqueeze(1).to(self.policy_net.device)
        next_state = torch.tensor(new_state, dtype=torch.float).to(self.policy_net.device)
        state = torch.tensor(state, dtype=torch.float).to(self.policy_net.device)
        action = torch.tensor(action, dtype=torch.float).to(self.policy_net.device)

        predicted_q_value1 = self.q_net1(state, action)
        predicted_q_value2 = self.q_net2(state, action)
        predicted_value    = self.value_net(state)
        new_action, log_prob = self.policy_net.sample_normal(state)

        reward = self.reward_scale*(reward - reward.mean(dim=0)) / (reward.std(dim=0) +  1e-6) # normalize with batch mean and std
        # Training Q Function
        target_value = self.target_value_net(next_state)
        target_q_value = reward + (1 - done) * self.gamma * target_value # if done==1, only reward
        q_value_loss1 = F.mse_loss(predicted_q_value1, target_q_value.detach())  # detach: no gradients for the variable
        q_value_loss2 = F.mse_loss(predicted_q_value2, target_q_value.detach())

        self.q_net1.optimizer.zero_grad()
        q_value_loss1.backward()
        self.q_net1.optimizer.step()
        self.q_net2.optimizer.zero_grad()
        q_value_loss2.backward()
        self.q_net2.optimizer.step()  

        # Training Value Function
        predicted_new_q_value = torch.min(self.q_net1(state, new_action), self.q_net2(state, new_action))
        target_value_func = predicted_new_q_value - alpha * log_prob # for stochastic training, it equals to expectation over action

        value_loss = F.mse_loss(predicted_value, target_value_func.detach())

        self.value_net.optimizer.zero_grad()
        value_loss.backward()
        self.value_net.optimizer.step()

        # Training Policy Function
        policy_loss = (alpha * log_prob - predicted_new_q_value).mean()

        self.policy_net.optimizer.zero_grad()
        policy_loss.backward()
        self.policy_net.optimizer.step()

        self.target_value_net = self.update_network_parameters(
            self.value_net, self.target_value_net
        )

        return (value_loss, q_value_loss1, q_value_loss2, policy_loss)