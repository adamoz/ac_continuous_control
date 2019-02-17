import abc
import copy
import numpy as np
from ac_continuous_control.model import Actor, Critic
from ac_continuous_control.replay_buffer import ReplayBuffer
import random
import torch
import torch.nn.functional as F
import torch.optim as optim


class AgentInterface(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def step(self, state, action, reward, next_state, done):
        """Save state to replay buffer and train if needed."""

    @abc.abstractmethod
    def act(self, state, add_noise=True):
        """Return actions for given state as per current policy.

        Params:
            state (array_like): current state
            add_noise (bool): add noise to argmax over actions

        """


class Agent(AgentInterface):
    """Interacts with and learns from the environment."""
    def __init__(self, state_size, action_size, num_agents, buffer_size=int(1e5),
                 batch_size=128, tau=1e-3, weight_decay=0, gamma=0.99, lr_actor=0.0001, lr_critic=0.0001, seed=0, device='cpu'):
        """Initialize an Agent object.

        Params:
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
            fc_units (list): List of unit counts in each layer of q-netwokr
            buffer_size (int): Replay buffer size
            batch_size (int): Size of sampled batches from replay buffer
            lr_actor (float): Learning rate
            lr_critic (float): Learning rate
            gamma (float): Reward discount
            tau (float): For soft update of target network parameters
            weight_decay (float): l2 loss during adam optimization

        """
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.state_size = state_size
        self.action_size = action_size
        self.weight_decay = weight_decay
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.num_agents = num_agents

        # Actor Network
        self.actor_local = Actor(state_size, action_size, seed).to(device)
        self.actor_target = Actor(state_size, action_size, seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.lr_actor)

        # Critic Network
        self.critic_local = Critic(state_size, action_size, seed).to(device)
        self.critic_target = Critic(state_size, action_size, seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.lr_critic, weight_decay=self.weight_decay)

        self.noise = OUNoise((num_agents, action_size), seed)
        self.memory = ReplayBuffer(buffer_size=buffer_size, batch_size=batch_size, seed=seed, device=device)

    def __repr__(self):
        return f'Agent(state_size={self.state_size}, action_size={self.action_size}, num_agents={self.num_agents}, device="{self.device}")'

    def step(self, states, actions, rewards, next_states, dones):
        # Save experience in replay memory
        self.memory.add(states, actions, rewards, next_states, dones)

        # Learn under update_rate.
        if self.memory.is_ready_to_sample():
            experiences = self.memory.sample()
            self.learn(experiences)

    def act(self, states, add_noise=True):
        """Return actions for given state as per current policy.

        Params:
            state (array_like): current state
            add_noise (bool): add noise to argmax over actions
        """

        states = torch.from_numpy(states).float().to(self.device)
        self.actor_local.eval()
        with torch.no_grad():
            actions = self.actor_local(states).cpu().data.numpy()
        self.actor_local.train()

        if add_noise:
            actions += self.noise.sample()
        return np.clip(actions, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma=None, tau=None):
        """Update value parameters using given batch of experience tuples.

        Params:
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
            tau (float): For soft update of target network parameters
        """

        if gamma is None:
            gamma = self.gamma

        if tau is None:
            tau = self.tau

        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        Q_expected = self.critic_local(states, actions)

        # Compute critic loss
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, tau)
        self.soft_update(self.actor_local, self.actor_target, tau)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """

        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def save(self, file_name):
        torch.save(self.actor_local.state_dict(), file_name + '_actor.pth')
        torch.save(self.critic_local.state_dict(), file_name + '_critic.pth')

    def load(self, file_name):
        self.actor_local.load_state_dict(torch.load(file_name + '_actor.pth'))
        self.actor_target.load_state_dict(torch.load(file_name + '_actor.pth'))
        self.critic_local.load_state_dict(torch.load(file_name + '_critic.pth'))
        self.critic_target.load_state_dict(torch.load(file_name + '_critic.pth'))


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0.0, theta=0.15, sigma=0.15, sigma_min=0.05, sigma_decay=.975):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.sigma_min = sigma_min
        self.sigma_decay = sigma_decay
        self.seed = random.seed(seed)
        self.size = size
        self.reset()

    def reset(self):
        self.state = copy.copy(self.mu)
        self.sigma = max(self.sigma_min, self.sigma * self.sigma_decay)

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx
        return self.state
