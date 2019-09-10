import numpy as np
import torch.nn as nn
import random
from collections import namedtuple, deque
import torch
import torch.nn.functional as F
import torch.optim as optim
import json


def parse_params(params_dir):
    with open(params_dir) as fp:
        params = json.load(fp)
    return params


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed, device):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience",
                                     field_names=["state", "action", "reward",
                                                  "next_state", "done"])
        random.seed(seed)
        self.device = device

    def add(self, state, action, reward, next_state, done, num_agents):
        """Add a new experience to memory."""
        for i in range(num_agents):
            e = self.experience(state[i], action[i], reward[i], next_state[i],
                                done[i])
            self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class Actor_critic_model(nn.Module):
    def __init__(self, params_dir, input_dim, act_size):
        super().__init__()
        self.input_dim = input_dim
        self.act_size = act_size
        self.params = parse_params(params_dir)
        self.actor = self.create_actor()
        self.critic = self.create_critic()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            if m.out_features <= 2:
                m.weight.data.uniform_(-3e-3, 3e-3)
            else:
                nn.init.kaiming_normal_(m.weight, mode='fan_in')

    def create_actor(self):
        torch.manual_seed(self.params['seed'])
        module_list = nn.ModuleList()
        layer = nn.Sequential()
        fc = nn.Linear(self.input_dim, self.params['hidden_dim'])
        layer.add_module(f"fc_layer_1", fc)
        layer.add_module(f"RELU_layer_1", nn.ReLU())
        module_list.append(layer)
        self.add_hidden_layer(module_list, self.params['actor_h_num'],
                              self.params['hidden_dim'],
                              int(self.params['hidden_dim']-100))
        out_put_layer = nn.Sequential()
        out_layer = nn.Sequential(nn.Linear(int(self.params['hidden_dim']-100),
                                            self.act_size))
        out_put_layer.add_module(f"out_put_layer", out_layer)
        out_put_layer.add_module(f"Tanh_out", nn.Tanh())
        module_list.append(out_put_layer)
        module_list.apply(self._init_weights)
        return module_list

    def create_critic(self):
        torch.manual_seed(self.params['seed'])
        module_list = nn.ModuleList()
        layer = nn.Sequential()
        fc = nn.Linear(self.input_dim, self.params['hidden_dim'])
        layer.add_module(f"fc_layer_1", fc)
        # layer.add_module(f"bn_layer_1",
                        # nn.BatchNorm1d(self.params['hidden_dim']))
        # layer.add_module(f"RELU_layer_1", nn.LeakyReLU())
        layer.add_module(f"RELU_layer_1", nn.ReLU())
        module_list.append(layer)
        self.add_hidden_layer(module_list, self.params['critic_h_num'],
                              self.params['hidden_dim']+self.act_size,
                              int(self.params['hidden_dim'])-100)
        out_put_layer = nn.Sequential()
        out_layer = nn.Sequential(nn.Linear(int(self.params['hidden_dim'])-100,
                                            1))
        out_put_layer.add_module(f"out_put_layer", out_layer)
        module_list.append(out_put_layer)
        module_list.apply(self._init_weights)
        return module_list

    def add_hidden_layer(self, module_list, num_hidden_layer,
                         input_dim, output_dim):
        if num_hidden_layer == 0:
            return
        for i in range(1, num_hidden_layer+1):
            layer = nn.Sequential()
            fc = nn.Linear(input_dim, output_dim)
            layer.add_module(f"fc_layer_{i}", fc)
            # layer.add_module(f"bn_layer_{i}",
                          #    nn.BatchNorm1d(output_dim))
            # layer.add_module(f"RELU_layer_{i}", nn.LeakyReLU())
            layer.add_module(f"RELU_layer_{i}", nn.ReLU())
            module_list.append(layer)
            input_dim = output_dim

    def forward(self, states, action=None, actor=True, train=True):
        '''
            If actor is True, output actions
            If Critic (actor = False), output state value
        '''
        x_ = states
        if actor:
            for m in self.actor:
                x_ = m(x_)
            return x_
        # forward in value path
        for idx, v in enumerate(self.critic):
            if idx == 1:
                x_ = torch.cat((x_, action), dim=1)
            x_ = v(x_)
        return x_


class Agent:
    def __init__(self, params_dir, state_size, action_size, num_agents,
                 device):
        self.params = parse_params(params_dir)
        random.seed(self.params['seed'])
        self.num_agents = num_agents
        self.device = device
        self.epsilon = 1
        # Local Model
        self.model = Actor_critic_model(params_dir, state_size,
                                        action_size).to(device)
        self.actor_optimizer = optim.Adam(self.model.actor.parameters(),
                                          lr=self.params["actor_lr"])
        # Target Model
        self.target = Actor_critic_model(params_dir, state_size,
                                         action_size).to(device)
        self.critic_optimizer = optim.Adam(self.model.critic.parameters(),
                                           lr=self.params["critic_lr"],
                                           weight_decay=0.0)
        # Replay Buffer
        self.memory = ReplayBuffer(action_size, self.params["buffer_size"],
                                   self.params["batch_size"],
                                   self.params["seed"], device)

        # Make local and target models are identical during initialization
        self.hard_update()
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done, self.num_agents)

        if len(self.memory):
            self.t_step = (self.t_step + 1) % self.params["update_freq"]
            if self.t_step == 0:
                # Only start learning when there are enough samples in the memory
                if len(self.memory) > self.params["batch_size"]:
                    for _ in range(self.params["update_freq"]):
                        experiences = self.memory.sample()
                        self.learn(experiences)

    def act(self, state):
        state = torch.from_numpy(state).float().to(self.device)
        self.model.eval()
        with torch.no_grad():
            action = self.model(state).cpu().data.numpy()
        self.model.train()
        return action

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.target(next_states)
        Q_targets_next = self.target(next_states, actions_next,
                                     actor=False)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (self.params['gamma'] * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.model(states, actions, actor=False)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.model(states)

        # we want to maximizde the state value using predicted actions,
        # However, since Pytorch is designed to find the minumum, we add
        # neagative in front of the actor loss
        actor_loss = -self.model(states, actions_pred, actor=False).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update()                 
        self.epsilon = max(self.epsilon - 0.00001, 0.1)

    def soft_update(self):
        for tp, lp in zip(self.target.parameters(),
                          self.model.parameters()):
            tp.data.copy_(self.params['TAU']*lp.data +
                          (1.0-self.params['TAU'])*tp.data)

    def hard_update(self):
        for tp, lp in zip(self.target.parameters(),
                          self.model.parameters()):
            tp.data.copy_(lp.data)

