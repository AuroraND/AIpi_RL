import numpy as np
from collections import namedtuple
import torch
import torch.nn as nn
import torch.optim as optim
import gym
import random
import cv2 as cv
import matplotlib.pyplot as plt

# 创建环境
env = gym.make("ALE/Breakout-v5", render_mode="human")
env.metadata["render_fps"] = 60
# 训练每批样本数
batch_size = 32
# 动作维度
action_size = env.action_space.n
# 状态维度
observation_space = env.observation_space
# 学习率
learning_rate = 1e-4
# reward discount
gamma = 0.99
# greedy policy
epsilon_start = 1
epsilon_end = 0.01
epsilon_decay = 1000000
epsilon_random_count = 50000
# 目标网络更新频率
target_update = 1000
# sample capacity
memory_capacity = 100000
# 记忆库样本形式
Transition = namedtuple("Transition", ("state", "action", "new_state", "reward"))
# 设备选择
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 经验回放
class Replay_Memory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    # 向记忆库中存入样本数据
    def push(self, args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


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
        self, in_channels, action_space, gamma, learning_rate, memory_capacity
    ):
        self.in_channels = in_channels
        self.action_space = action_space
        self.action_size = action_space.n
        self.stepdone = 0
        self.gamma = gamma
        self.lr = learning_rate
        # 记忆库
        self.memory_buffer = Replay_Memory(memory_capacity)
        # 价值估计网络
        self.estimate_cnn = CNN(in_channels, action_size)
        # 目标网络
        self.target_cnn = CNN(in_channels, action_size)
        # 优化器
        self.optimizer = optim.Adam(self.estimate_cnn.parameters(), lr=self.lr)
        # 损失函数
        self.criterion = nn.SmoothL1Loss()

    def choose_action(self, state):
        self.stepdone += 1
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * np.exp(
            -(self.stepdone / epsilon_decay)
        )
        rand = np.random.uniform(0, 1)
        if self.stepdone < epsilon_random_count or rand < epsilon:
            action = torch.tensor(
                [[np.random.choice(self.action_size)]], dtype=torch.long
            )
        else:
            action = self.estimate_cnn(state).detach().max(1)[1].view(1, 1)
        return action

    def learn(self, args):
        # 存入记忆库
        self.memory_buffer.push(args)
        # 记忆库中样本数量小于batch_size则跳过学习
        if self.memory_buffer.__len__() < batch_size:
            return
        # 取样
        transitions = self.memory_buffer.sample(batch_size)
        # 分类解压
        batch = Transition(*zip(*transitions))
        # 转化数据为元组
        actions = batch.action
        rewards = tuple(map(lambda r: torch.tensor([r]), batch.reward))
        # 下一状态为非终止状态的掩模 [32]
        no_terminal_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.new_state)), dtype=torch.uint8
        ).bool()
        # 当前批次样本数据
        # [?,4,84,84]
        batch_new_state = torch.cat([s for s in batch.new_state if s is not None])
        # [32,4,84,84]
        batch_state = torch.cat(batch.state)
        # [32,1]
        batch_action = torch.cat(actions)
        # [32]
        batch_reward = torch.cat(rewards)
        # 估计Q值(采取batch_action行动的预测Q值) [32,1]
        estimate_Q = self.estimate_cnn(batch_state).gather(1, batch_action).view(32)
        # 目标Q值(采取batch_action后达到新状态的目标Q值) [32]
        target_Q = torch.zeros(batch_size)
        target_Q[no_terminal_mask] = (
            self.gamma * self.target_cnn(batch_new_state).max(1)[0]
            + batch_reward[no_terminal_mask]
        )
        # 反向传播
        loss = self.criterion(estimate_Q, target_Q)
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.estimate_cnn.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()


dqn = DQN(4, env.action_space, gamma, learning_rate, memory_capacity)


def pre_process(observation):
    # 裁剪、转化为灰度图
    img = cv.cvtColor(cv.resize(observation, (84, 84)), cv.COLOR_BGR2GRAY)
    # 二值化阈值处理
    ret, img = cv.threshold(img, 1, 255, cv.THRESH_BINARY)
    # 堆叠四帧为一个状态
    state = np.stack((img, img, img, img), axis=0)
    return np.reshape(state, (1, 4, 84, 84))


# 迭代训练次数
episodes = 10000

for i in range(episodes):
    print(i)
    # 初始化
    observation = env.reset()[0]
    # 渲染
    env.render()
    while True:
        # 渲染
        env.render()
        # 状态处理
        state = pre_process(observation)
        # 动作选择
        action = dqn.choose_action(state)
        # 环境转换
        observation_, reward, done, info, _ = env.step(action)
        # 新状态处理
        state_ = pre_process(observation_)
        # 训练学习
        values = (
            torch.tensor(state, dtype=torch.float32),
            action,
            torch.tensor(state_, dtype=torch.float32),
            reward,
        )
        dqn.learn(values)
        # 状态转换
        state = state_

        if done:
            break

    if i % 50 == 0:
        dqn.target_cnn = dqn.estimate_cnn

env = gym.make("ALE/Breakout-v5", render_mode="human")
env.metadata["render_fps"] = 60
# 初始化
observation = env.reset()[0]
# 渲染
env.render()
while True:
    # 渲染
    env.render()
    # 状态处理
    state = pre_process(observation)
    # 动作选择
    action = dqn.choose_action(state)
    # 环境转换
    observation_, reward, done, info, _ = env.step(action)
    # 状态转换
    state = state_

    if done:
        break
        # plt.imshow(img,cmap='gray')
        # plt.show()
        # cv.imshow('Breakout',img)
        # cv.waitKey(0)
