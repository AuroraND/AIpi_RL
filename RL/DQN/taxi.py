import random
import numpy as np
import collections
import gym
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm

# 超参数定义
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
HIDDEN_DIM = 128
GAMMA = 0.98
EPSILON_START = 0.99
EPSILON_END = 0.01
EPSILON_DECAY = 20000
TARGET_UPDATE = 20
BUFFER_CAPCITY = 5000

Transition = collections.namedtuple(
    "transition", ("state", "action", "new_state", "reward", "done")
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Replay_Buffer:
    def __init__(self, capcity):
        self.buffer = collections.deque(maxlen=capcity)

    def push(self, args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def size(self):
        return len(self.buffer)


class NN(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(NN, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x):
        x = self.features(x)
        return x


class DQN:
    def __init__(self,state_dim,hidden_dim,action_dim,learning_rate,gamma,target_update,device,):
        self.action_dim = action_dim
        self.gamma = gamma
        self.lr = learning_rate
        self.target_update = target_update
        self.count = 0
        self.stepdone = 0
        # 估计网络
        self.E_nn = NN(state_dim, hidden_dim, self.action_dim).to(device)
        # 目标网络
        self.T_nn = NN(state_dim, hidden_dim, self.action_dim).to(device)
        # 优化器
        self.optimizer = torch.optim.Adam(self.E_nn.parameters(), lr=self.lr)
        # 损失函数
        self.criterion = nn.MSELoss()
        self.device = device

    def select_action(self, state):
        self.stepdone += 1
        epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * (
            np.exp(-(self.stepdone / EPSILON_DECAY))
        )
        rand = random.uniform(0, 1)
        if rand < epsilon:
            action = np.random.choice(self.action_dim)
        else:
            action = self.E_nn(state).detach().max(1)[1].item()
        return action

    def learn(self, transitions):
        # 取样解压
        batch = Transition(*zip(*transitions))
        batch_states = torch.cat(batch.state).to(self.device)
        # [32,1]
        batch_actions = torch.tensor(batch.action).view(-1, 1).to(self.device)
        batch_newstates = torch.cat(batch.new_state).to(self.device)
        # [32,1]
        batch_rewards = (
            torch.tensor(batch.reward, dtype=torch.float).view(-1, 1).to(self.device)
        )
        # [32,1]
        batch_dones = (
            torch.tensor(batch.done, dtype=torch.float).view(-1, 1).to(self.device)
        )
        # 估计Q价值
        E_Q = self.E_nn(batch_states).gather(1, batch_actions).view(-1, 1)
        # 下一状态最大目标Q值
        T_max = self.T_nn(batch_newstates).max(1)[0].view(-1, 1)
        # 目标Q值
        T_Q = batch_rewards + self.gamma * T_max * (1 - batch_dones)
        # 误差分析，反向传播
        loss = torch.mean(self.criterion(E_Q, T_Q))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.T_nn.load_state_dict(self.E_nn.state_dict())

        self.count += 1


env = gym.make("Taxi-v3", render_mode="ansi")
replay_buffer = Replay_Buffer(BUFFER_CAPCITY)
state_dim = env.observation_space.n
action_dim = env.action_space.n
agent = DQN(
    state_dim=state_dim,
    hidden_dim=HIDDEN_DIM,
    action_dim=action_dim,
    learning_rate=LEARNING_RATE,
    gamma=GAMMA,
    device=device,
    target_update=TARGET_UPDATE,
)

# 迭代次数
episodes = 2000

x = []
y = []
step = 500
max_step = 500

with tqdm(total=episodes) as pbar:
    for i in range(episodes):
        x.append(i)
        y.append(step)

        step = 0
        observation = env.reset()[0]
        done = False

        while not done:

            state = np.reshape(np.eye(state_dim)[observation], (1, state_dim))
            state = torch.tensor(state, dtype=torch.float).to(device)

            action = agent.select_action(state)

            observation_, reward, done, info, _ = env.step(action)

            step += 1

            if observation == observation_:
                reward = -10

            if step == max_step:
                break

            state_ = np.reshape(np.eye(state_dim)[observation_], (1, state_dim))
            state_ = torch.tensor(state_, dtype=torch.float).to(device)

            replay_buffer.push((state, action, state_, reward, done))

            if replay_buffer.size() > BATCH_SIZE:
                transitions = replay_buffer.sample(BATCH_SIZE)
                agent.learn(transitions)

            observation = observation_

        pbar.update(1)

plt.figure(figsize=(10, 6))
plt.plot(x, y, color="r")
plt.xlabel("episode")
plt.ylabel("step")
plt.show()

env = gym.make("Taxi-v3", render_mode="human")
observation = env.reset()[0]
env.render()
done = False
step = 0

while not done:

    state = np.reshape(np.eye(state_dim)[observation], (1, state_dim))
    state = torch.tensor(state, dtype=torch.float).to(device)

    action = agent.select_action(state)

    observation_, reward, done, info, _ = env.step(action)

    step += 1
    if step > max_step:
        break

    observation = observation_

print(step)
