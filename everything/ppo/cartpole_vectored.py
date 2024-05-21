import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from multiprocessing import Process, Pipe


# Hyperparameters
gamma = 0.99
lr = 0.002
clip_epsilon = 0.2
update_epochs = 10
rollout_steps = 2048
batch_size = 64
entropy_coef = 0.01  # Coefficient for the entropy bonus
num_envs = 4  # Number of parallel environments


# Define the policy network
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return Categorical(logits=self.fc3(x))

# Define the value network
class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return self.fc3(x)


# Function to run an environment in a separate process
def worker(remote, parent_remote, env_fn):
    parent_remote.close()
    env: gym.Env = env_fn()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            state, reward, done, trunced, info = env.step(data)
            if done:
                state, _ = env.reset()
            remote.send((state, reward, done, info))
        elif cmd == 'reset':
            state, _ = env.reset()
            remote.send(state)
        elif cmd == 'close':
            remote.close()
            break


# Function to create a vectorized environment
def make_env(env_name):
    return lambda: gym.make(env_name)


class VecEnv:
    def __init__(self, env_fns):
        self.n_envs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(self.n_envs)])
        self.processes = [Process(target=worker, args=(work_remote, remote, env_fn))
                          for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for process in self.processes:
            process.start()
        for work_remote in self.work_remotes:
            work_remote.close()

    def step(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        results = [remote.recv() for remote in self.remotes]
        states, rewards, dones, infos = zip(*results)
        return states, rewards, dones, infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return [remote.recv() for remote in self.remotes]

    def close(self):
        for remote in self.remotes:
            remote.send(('close', None))
        for process in self.processes:
            process.join()


# Initialize environment, policy network, and value network
env_name = 'CartPole-v1'
envs = VecEnv([make_env(env_name) for _ in range(num_envs)])
state_dim = gym.make(env_name).observation_space.shape[0]
action_dim = gym.make(env_name).action_space.n

policy_net = PolicyNetwork(state_dim, action_dim)
value_net = ValueNetwork(state_dim)
policy_optimizer = optim.Adam(policy_net.parameters(), lr=lr)
value_optimizer = optim.Adam(value_net.parameters(), lr=lr)


# Function to collect rollouts in parallel
@torch.no_grad()
def collect_rollouts(policy_net: PolicyNetwork, value_net: ValueNetwork, envs: VecEnv, rollout_steps: int):
    states, actions, rewards, log_probs, values, dones = [], [], [], [], [], []
    state = envs.reset()
    for _ in range(rollout_steps):
        state_tensor = torch.tensor(state, dtype=torch.float32)
        dist = policy_net(state_tensor)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        value = value_net(state_tensor)

        next_state, reward, done, _ = envs.step(action.numpy())

        states.append(state_tensor)
        actions.append(action)
        rewards.append(reward)
        log_probs.append(log_prob)
        values.append(value)
        dones.append(done)

        state = next_state

    return states, actions, rewards, log_probs, values, dones


# Function to compute advantages and returns
def compute_advantages_and_returns(rewards, values, dones, gamma):
    advantages, returns = [], []
    advantage = 0
    for i in reversed(range(len(rewards))):
        if dones[i]:
            advantage = 0
        td_error = rewards[i] + (gamma * values[i + 1] * (1 - dones[i])) - values[i]
        advantage = td_error + (gamma * advantage * (1 - dones[i]))
        returns.insert(0, advantage + values[i])
        advantages.insert(0, advantage)
    return advantages, returns


# Training loop
for iteration in range(1000):  # Number of training iterations
    states, actions, rewards, log_probs, values, dones = collect_rollouts(policy_net, value_net, envs, rollout_steps)

    values.append(torch.tensor([0.0] * num_envs))  # Add a zero for the last value
    advantages, returns = compute_advantages_and_returns(rewards, values, dones, gamma)

    states = torch.cat(states)
    actions = torch.cat(actions)
    log_probs = torch.cat(log_probs)
    returns = torch.tensor(returns, dtype=torch.float32).view(-1, 1)
    advantages = torch.tensor(advantages, dtype=torch.float32).view(-1, 1)

    for _ in range(update_epochs):
        for start in range(0, len(states), batch_size):
            end = start + batch_size
            batch_states = states[start:end]
            batch_actions = actions[start:end]
            batch_old_log_probs = log_probs[start:end]
            batch_returns = returns[start:end]
            batch_advantages = advantages[start:end]

            dist = policy_net(batch_states)
            new_log_probs = dist.log_prob(batch_actions)
            entropy = dist.entropy().mean()

            ratio = (new_log_probs - batch_old_log_probs).exp()
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * batch_advantages
            policy_loss = -torch.min(surr1, surr2).mean() - entropy_coef * entropy

            value_loss = (value_net(batch_states).squeeze() - batch_returns).pow(2).mean()

            policy_optimizer.zero_grad()
            policy_loss.backward()
            policy_optimizer.step()

            value_optimizer.zero_grad()
            value_loss.backward()
            value_optimizer.step()

    if iteration % 10 == 0:
        print(f'Iteration {iteration}, Loss: {policy_loss.item()}, Value Loss: {value_loss.item()}')

envs.close()
