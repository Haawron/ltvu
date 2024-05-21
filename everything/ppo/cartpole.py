import gym

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


# Hyperparameters
gamma = 0.99
lr = 0.032
clip_epsilon = 0.2
update_epochs = 10
rollout_steps = 2048
batch_size = 1024
entropy_coef = 0.01  # Coefficient for the entropy bonus


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


# Initialize environment, policy network, and value network
env = gym.make('CartPole-v1')
# vr = VideoRecorder(env, 'videos/cartpole-before.mp4')
# env = gym.wrappers.Monitor(env, '.video/', force=True)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

policy_net = PolicyNetwork(state_dim, action_dim).cuda()
value_net = ValueNetwork(state_dim).cuda()
policy_optimizer = optim.Adam(policy_net.parameters(), lr=lr)
value_optimizer = optim.Adam(value_net.parameters(), lr=lr)
# torch.autograd.set_detect_anomaly(True)


@torch.no_grad()
def collect_rollouts(policy_net: PolicyNetwork, value_net: ValueNetwork, env: gym.Env, rollout_steps: int):
    states, actions, rewards, log_probs, values, dones = [], [], [], [], [], []
    state, info = env.reset()
    for _ in range(rollout_steps):
        state = torch.tensor(state, dtype=torch.float32, device='cuda')
        dist = policy_net.forward(state)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        value = value_net.forward(state)

        # vr.capture_frame()
        # env.render()
        next_state, reward, done, truncated, info = env.step(action.item())

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        log_probs.append(log_prob)
        values.append(value)
        dones.append(done)

        state = next_state
        if done:
            state, info = env.reset()

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
    states, actions, rewards, log_probs, values, dones = collect_rollouts(policy_net, value_net, env, rollout_steps)

    values.append(torch.tensor(0.0))  # Add a zero for the last value
    advantages, returns = compute_advantages_and_returns(rewards, values, dones, gamma)

    states = torch.stack(states)
    actions = torch.stack(actions)
    log_probs = torch.stack(log_probs)
    returns = torch.tensor(returns, dtype=torch.float32, device='cuda')
    advantages = torch.tensor(advantages, dtype=torch.float32, device='cuda')

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

env.close()
