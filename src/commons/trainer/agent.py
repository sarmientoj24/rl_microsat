from src.commons.utils import ReplayBuffer


class BaseAgent():
    def __init__(self, alpha=0.0001, state_dim=50, env=None, gamma=0.995,
            action_dim=4, action_range=1, max_size=1000000, tau=1e-2,
            hidden_dim=128, batch_size=256, reward_scale=1):
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.memory = ReplayBuffer(max_size)
        self.batch_size = batch_size
        self.action_dim = action_dim
        self.reward_scale = reward_scale
        self.action_range = action_range


    def choose_action(self, state, deterministic=False):
        pass

    def remember(self, state, action, reward, new_state, done):
        self.memory.push(state, action, reward, new_state, done)

    def update_network_parameters(self, net, target_net, soft_tau=None):
        if soft_tau is None:
            soft_tau = self.tau
        for target_param, param in zip(target_net.parameters(), net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )

        return target_net

    def save_models(self):
        pass

    def load_models(self):
        pass

    def learn(self):
        pass