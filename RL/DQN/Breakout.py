import numpy as np
from collections import namedtuple
import collections
import torch
import torch.nn as nn
import torch.optim as optim
import gym
import random
import cv2 as cv
import matplotlib.pyplot as plt
from tqdm import tqdm

class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.real_done = True

    def step(self, action):
        obs, reward, done, info, _ = self.env.step(action)
        self.real_done = done
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            done = True
        self.lives = lives
        return obs, reward, done, info, _

    def reset(self, **kwargs):
        if self.real_done:
            obs, info = self.env.reset(**kwargs)
        else:
            obs, _, _, _, info = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs, info


# 超参数定义

# 训练每批样本数
BATCH_SIZE = 32
# 学习率
LEARNING_RATE = 1e-4
# reward discount
GAMMA = 0.98
# greedy policy
EPSILON_START = 1
EPSILON_END = 0.01
EPSILON_DECAY = 30000
# 目标网络更新频率
TARGET_UPDATE = 1000
# sample capacity
BUFFER_CAPCITY = 100000

# 记忆库样本形式
Transition = namedtuple(
    "Transition", ("state", "action", "new_state", "reward", "done")
)
# 设备选择
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 经验回放
class Replay_Buffer:
    def __init__(self, capcity):
        self.buffer = collections.deque(maxlen=capcity)

    def push(self, args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def size(self):
        return len(self.buffer)


class CNN(nn.Module):
    def __init__(self, in_channels, action_size):
        super(CNN, self).__init__()
        # 特征提取
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        # 不同动作价值计算
        self.classify = nn.Sequential(
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU(),
            nn.Linear(512, action_size),
        )

    # 前向传播
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classify(x)
        return x


class DQN:
    def __init__(
        self,
        state_dim,
        action_dim,
        learning_rate,
        gamma,
        target_update,
        device,
    ):
        self.action_dim = action_dim
        self.gamma = gamma
        self.lr = learning_rate
        self.target_update = target_update
        self.count = 0
        self.stepdone = 0
        # 估计网络
        self.E_cnn = CNN(state_dim, self.action_dim).to(device)
        # 目标网络
        self.T_cnn = CNN(state_dim, action_dim).to(device)
        # 优化器
        self.optimizer = torch.optim.Adam(self.E_cnn.parameters(), lr=self.lr)
        # 损失函数
        self.criterion = nn.MSELoss()
        self.device = device

    def select_action(self, state):
        self.stepdone += 1
        epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * (
            np.exp(-(self.stepdone / EPSILON_DECAY))
        )
        print(self.stepdone,epsilon)
        rand = random.uniform(0, 1)
        if rand < epsilon:
            action = np.random.choice(self.action_dim)
        else:
            action = self.E_cnn(state).detach().max(1)[1].item()
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
        E_Q = self.E_cnn(batch_states).gather(1, batch_actions).view(-1, 1)
        # 下一状态最大目标Q值
        T_max = self.T_cnn(batch_newstates).max(1)[0].view(-1, 1)
        # 目标Q值
        T_Q = batch_rewards + self.gamma * T_max * (1 - batch_dones)
        # 误差分析，反向传播
        loss = torch.mean(self.criterion(E_Q, T_Q))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.T_cnn.load_state_dict(self.E_cnn.state_dict())

        self.count += 1


def pre_process(observation):
    # 裁剪、转化为灰度图
    img = cv.cvtColor(cv.resize(observation, (84, 84)), cv.COLOR_BGR2GRAY)
    # 二值化阈值处理
    ret, img = cv.threshold(img, 1, 255, cv.THRESH_BINARY)
    # 堆叠四帧为一个状态
    state = np.stack((img, img, img, img), axis=0)
    state = np.reshape(state, (1, 4, 84, 84))
    state = torch.tensor(state, dtype=torch.float32).to(device)
    return state


# 创建环境
env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")
env.metadata["render_fps"] = 280
env = EpisodicLifeEnv(env)

replay_buffer = Replay_Buffer(BUFFER_CAPCITY)
action_dim = env.action_space.n
agent = DQN(
    state_dim=4,
    action_dim=action_dim,
    learning_rate=LEARNING_RATE,
    gamma=GAMMA,
    device=device,
    target_update=TARGET_UPDATE,
)


# 迭代训练次数
episodes = 5000
x = []
y = []
r = 0
with tqdm(total=episodes) as pbar:
    for i in range(episodes):
        x.append(i)
        y.append(r)
        r = 0

        # 初始化
        observation = env.reset()[0]

        while True:
            # 状态处理
            state = pre_process(observation)

            # 动作选择
            action = agent.select_action(state)

            # 环境转换
            observation_, reward, done, info, _ = env.step(action)
            r += reward

            # 新状态处理
            state_ = pre_process(observation_)

            # 训练学习
            values = (state, action, state_, reward, done)
            replay_buffer.push(values)

            if replay_buffer.size() > BATCH_SIZE:
                transitions = replay_buffer.sample(BATCH_SIZE)
                agent.learn(transitions)

            # 状态转换
            observation = observation_

            if done:
                break

        pbar.update(1)

plt.figure(figsize=(10, 6))
plt.plot(x, y, color="r")
plt.show()

env = gym.make("ALE/Breakout-v5", render_mode="human")
env.metadata["render_fps"] = 280
# 初始化
observation = env.reset()[0]
# 渲染
env.render()
r = 0

while True:

    # 渲染
    env.render()

    # 状态处理
    state = pre_process(observation)

    # 动作选择
    action = agent.select_action(state)

    # 环境转换
    observation_, reward, done, info, _ = env.step(action)

    r += reward

    if done:
        break
    # 状态转换
    observation = observation_

print(r)


