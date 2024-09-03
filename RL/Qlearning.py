import gym
import numpy as np
import pandas as pd
import time

# 创建环境
env = gym.make("Taxi-v3", render_mode="ansi")

# 动作
actions = [0, 1, 2, 3, 4, 5]
# 动作维度
action_size = env.action_space.n
# 状态维度
state_size = env.observation_space.n
# 创建Q表
q_table = np.zeros((state_size, action_size))
# 学习率
learning_rate = 0.9
# 贪心策略选择初始概率
epsilon_start = 1
# 最终贪心策略选择概率
epsilon_end = 0.02
# 衰减系数
decay_rate = 0.95
# 对未来奖励的折扣
gamma = 0.9
# 训练迭代次数
total_episode = 5000
# 每回合训练最大步数
step_max = 50


def choose_action(state, epsilon, q_table):
    # 生成0-1随机数
    rand = np.random.uniform(0, 1)
    # 以epsilon概率选择q值最高的行动
    if rand > epsilon:
        state_actions = q_table[state, :]
        max_indexs = np.argwhere(state_actions == np.max(state_actions)).flatten()
        action = np.random.choice(max_indexs)
    # 随机采取行动
    else:
        action = np.random.choice(actions)
    # 返回动作
    return action


start_time = time.time()

# 测试训练模型q表
for episode in range(total_episode):
    # 初始状态
    state = env.reset()[0]
    # 初始化步数
    step = 0
    # 渲染画面
    env.render()

    while True:
        # 根据当前状态选择动作
        action = choose_action(state, epsilon_start, q_table)

        # 根据动作得到下一步状态、奖励等
        new_state, reward, done, info, _ = env.step(action)

        # 环境更新
        # env.render()

        if state == new_state:
            reward = -1

        # 根据状态奖励学习并更新q表
        if done:
            q_table[state, action] = reward
            break
        else:
            # 当前q表原状态该动作下估计价值
            q_predict = q_table[state, action]
            # 原状态当前动作下目标价值
            q_target = reward + gamma * np.max(q_table[new_state, :])
            # 以一定学习效率更新原状态当前动作下估计值
            q_table[state, action] += learning_rate * (q_target - q_predict)

        # 更新状态
        state = new_state
        step = step + 1

        if step > step_max:
            break

    if epsilon_start > epsilon_end:
        epsilon_start *= decay_rate

end_time = time.time()

print(end_time - start_time)

print(q_table)

env.close()

env = gym.make("Taxi-v3", render_mode="human")

# 测试结果
# 初始化状态
state = env.reset()[0]
step = 0
# 渲染环境
env.render()

while True:

    # 根据状态选择动作
    action = choose_action(state, epsilon_end, q_table)

    # 更新环境
    new_state, reward, done, info, _ = env.step(action)
    env.render()

    # 更新状态
    state = new_state
    step += 1

    if done or step > step_max:
        break

print(step)
env.close()
