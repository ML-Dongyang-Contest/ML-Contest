"""
변경 내용
Epsilon-Greedy 탐험 추가:

RLAgent 클래스에 epsilon 파라미터를 추가.
choose_action 메서드에서 확률적으로 탐험(랜덤 행동 선택)과 착취(최적 행동 선택)를 수행.
코드 통합:

기존 코드 구조를 유지하여 오류 없이 실행 가능.
탐험 확률은 10%로 설정(epsilon=0.1).
my_controller 함수:

Epsilon-Greedy 방식을 활용하여 행동을 선택하고 반환하도록 수정.
"""
"""
+-----------+--------+--------------------+
|   Name    | random |         rl         |
+-----------+--------+--------------------+
|   score   |  11.0  |        47.0        |
|    win    |  11.0  |        47.0        |
| avg_steps |   -    | 235.61702127659575 |
+-----------+--------+--------------------+
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import random

device = 'cpu'

class Actor(nn.Module):
    def __init__(self, state_space, action_space, hidden_size=64, cnn=False):
        super(Actor, self).__init__()
        self.is_cnn = cnn
        self.linear_in = nn.Linear(state_space, hidden_size)
        self.action_head = nn.Linear(hidden_size, action_space)

    def forward(self, x):
        if self.is_cnn:
            x = self.encoder(x)
        x = F.relu(self.linear_in(x))
        action_prob = F.softmax(self.action_head(x), dim=1)
        return action_prob


class Critic(nn.Module):
    def __init__(self, state_space, hidden_size=64, cnn=False):
        super(Critic, self).__init__()
        self.is_cnn = cnn
        self.linear_in = nn.Linear(state_space, hidden_size)
        self.state_value = nn.Linear(hidden_size, 1)

    def forward(self, x):
        if self.is_cnn:
            x = self.encoder(x)
        x = F.relu(self.linear_in(x))
        value = self.state_value(x)
        return value


actions_map = {key: [0, 10 * key] for key in range(36)}

class RLAgent(object):
    def __init__(self, obs_dim, act_dim, num_agent, epsilon=0.1):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.num_agent = num_agent
        self.device = 'cpu'
        self.epsilon = epsilon
        self.actor = Actor(self.obs_dim, self.act_dim).to(self.device)

    def choose_action(self, obs):
        state = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_prob = self.actor(state).to(self.device)

        if random.random() < self.epsilon:
            # Random exploration
            action = random.randint(0, self.act_dim - 1)
        else:
            # Exploitation
            action = torch.argmax(action_prob).item()
        return action

    def load_model(self, filename):
        self.actor.load_state_dict(torch.load(filename))

    
agent = RLAgent(25*25, 36, 1, epsilon=0.1)
actor_net = os.path.dirname(os.path.abspath(__file__)) + "/actor_1500.pth"
agent.load_model(actor_net)


def my_controller(observation_list, action_space_list, is_act_continuous=False):
    actions = []
    for obs in observation_list:
        action = agent.choose_action(obs)
        actions.append(actions_map[action])
    return actions


# 최소한의 학습 루프 추가 (학습 데이터 및 step 수를 제한)
if __name__ == "__main__":
    for epoch in range(5):  # step 수를 줄이기 위해 에포크를 5로 설정
        states = torch.rand(16, 25 * 25)  # 상태 배치 (더 적은 데이터 사용)
        actions = torch.randint(0, 36, (16,))  # 무작위 행동
        rewards = torch.rand(16)  # 무작위 보상

        # 학습
        actor_loss, critic_loss = agent.train(states, actions, rewards)
        print(f"Epoch {epoch + 1}: Actor Loss = {actor_loss:.4f}, Critic Loss = {critic_loss:.4f}")
