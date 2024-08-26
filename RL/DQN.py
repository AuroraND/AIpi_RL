import numpy as np
import torch
import torch.nn as nn
import gym
from gymnasium.wrappers.atari_preprocessing import AtariPreprocessing
from gymnasium.wrappers.frame_stack import FrameStack

# 创建环境
env = gym.make("ALE/Breakout-v5", render_mode="human")

# 动作维度
action_size = env.action_space.n
# 状态维度
observation_space = env.observation_space
# 学习率
learning_rate = 0.001
# reward discount
gamma = 0.9
# greedy policy
epsilon_max = 1
epsilon_min = 0.01
decay_rate = 0.97
# sample capacity
memory_capacity = 5000


class CNN(nn.Module):
    def __init__(self, class_num):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classify = nn.Sequential(
            nn.Linear(32 * 50 * 40, 128),
            nn.ReLU(),
            nn.Linear(128, class_num),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classify(x)
        return x


# class DQN():
#     def __init__(self):


# estimate network
estimate_cnn = CNN(action_size)
# target network
target_cnn = CNN(action_size)


