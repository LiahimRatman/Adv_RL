import numpy as np
import math
import random
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from TicTacToeClass import TicTacToe


device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
# device = torch.device("cpu")
MEM_SIZE = 1000000


class Policy:
    def __init__(self):
        self.Q = {}

    def set_learning_params(self, gamma, alpha, eps):
        self.gamma = gamma
        self.alpha = alpha
        self.eps = eps

    def gready(self, board_hash, n_empty):
        if board_hash in self.Q:
            index = np.argmax(self.Q[board_hash])
        else:
            index = np.random.randint(n_empty)

        return index

    def eps_gready(self, board_hash, n_empty):
        if board_hash in self.Q and np.random.random() > self.eps:
            index = np.argmax(self.Q[board_hash])
        else:
            index = np.random.randint(n_empty)
            if board_hash not in self.Q:
                self.Q[board_hash] = np.zeros(n_empty)

        return index

    def set_q(self, board_hash, index, reward):
        self.Q[board_hash][index] = reward

    def update_q(self, prev_board_hash, current_board_hash, prev_action, reward):
        if prev_board_hash is not None:
            temp = self.gamma * np.max(self.Q[current_board_hash]) if current_board_hash in self.Q else 0
            self.Q[prev_board_hash][prev_action] += self.alpha * (
                        -abs(reward) + temp - self.Q[prev_board_hash][prev_action])


class Memory:
    def __init__(self, capacity):
        self.memory = []
        self.capacity = capacity
        self.position = 0

    def store(self, ex):
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.position] = ex
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class NNQ(nn.Module):
    def __init__(self,
                 channels_in,
                 channels_out,
                 kernels,
                 linear_in,
                 linear_out):
        nn.Module.__init__(self)
        self.linear_in = linear_in
        self.convs = nn.ModuleList(
            [nn.Conv2d(channels_in[i], channels_out[i], kernel_size=kernels[i], stride=1) for i in
             range(len(channels_in))])
        self.fcs = nn.ModuleList([nn.Linear(linear_in[i], linear_out[i]) for i in range(len(linear_in))])
        self.flat = nn.Flatten(1, 3)

    def forward(self, input):
        output = input
        for i, conv in enumerate(self.convs):
            output = conv(output)
            output = F.relu(output)
        output = self.flat(output)
        for i, fc in enumerate(self.fcs):
            output = fc(output)
            if i != (len(self.linear_in) - 1):
                output = F.relu(output)

        return output


class NNQ_Dueling(nn.Module):
    def __init__(self, channels_in, channels_out, kernels, linear_in, linear_out):
        nn.Module.__init__(self)
        n = len(linear_in)
        self.linear_in = linear_in
        self.convs = nn.ModuleList(
            [nn.Conv2d(channels_in[i], channels_out[i], kernel_size=kernels[i], stride=1) for i in
             range(len(channels_in))])
        self.fcs = nn.ModuleList([nn.Linear(linear_in[i], linear_out[i]) for i in range(n)])
        self.fc_v = nn.Linear(linear_in[n - 1], 1)
        self.flat = nn.Flatten(1, 3)

    def forward(self, input):
        output = input
        for i, conv in enumerate(self.convs):
            output = conv(output)
            output = F.relu(output)
        output = self.flat(output)
        for i, fc in enumerate(self.fcs):
            if i != (len(self.linear_in) - 1):
                output = fc(output)
                output = F.relu(output)
            else:
                output_v = self.fc_v(output)
                output_a = fc(output)
        output = output_v + (output_a - output_a.mean(dim=1, keepdim=True))

        return output


class DQN:
    def __init__(self,
                 models,
                 n_rows=4,
                 n_cols=4,
                 n_win=4,
                 gamma=0.8,
                 lr=0.0001,
                 weight_decay=0.001,
                 batch_size=512,
                 device=device):
        self.env = TicTacToe(n_rows, n_cols, n_win)
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.models = {-1: models[0].to(device),
                       1: models[1].to(device)}
        self.device = device
        self.batch_size = batch_size
        self.prev_state = {-1: None, 1: None}
        self.prev_action = {}
        self.gamma = gamma
        self.num_step = 0
        self.steps_done = 0
        self.eps_init, self.eps_final, self.eps_decay = 0.9, 0.05, 100000
        self.memories = {-1: Memory(MEM_SIZE), 1: Memory(MEM_SIZE)}
        self.optimizers = {-1: optim.Adam(self.models[-1].parameters(), lr=lr, weight_decay=weight_decay),
                           1: optim.Adam(self.models[1].parameters(), lr=lr, weight_decay=weight_decay)}

    def state_to_tensor(self, state):
        state_int = np.array([int(char) for char in state])
        n = self.n_rows
        crosses = np.where(state_int == 2, 1, 0).reshape(n, n)
        zeros = np.where(state_int == 0, 1, 0).reshape(n, n)
        empty_spaces = np.where(state_int == 1, 1, 0).reshape(n, n)

        return torch.Tensor(np.stack([crosses, zeros, empty_spaces])).reshape(3, n, n)

    def select_greedy_action(self, state, cur_turn):
        return self.models[cur_turn](state.unsqueeze(0)).data.max(1)[1].view(1, 1)

    def select_action(self, state, cur_turn):
        sample = random.random()
        self.num_step += 1
        eps_threshold = self.eps_final + (self.eps_init -
                                          self.eps_final) * math.exp(-1. * self.num_step / self.eps_decay)
        if sample > eps_threshold:
            return self.select_greedy_action(state, cur_turn)
        else:
            return torch.tensor([[random.randrange(self.n_rows * self.n_cols)]],
                                dtype=torch.int64)

    def run_episode(self,
                    do_learning=True,
                    greedy=False):
        self.env.reset()
        done = False
        self.prev_state = {-1: None, 1: None}
        self.prev_action = {}
        state, self.empty_spaces, cur_turn = self.env.getState()
        while not done:
            state_tensor = self.state_to_tensor(state)
            with torch.no_grad():
                if greedy:
                    index = self.select_greedy_action(state_tensor.to(self.device), cur_turn).cpu()
                else:
                    index = self.select_action(state_tensor.to(self.device), cur_turn).cpu()

            self.prev_state[cur_turn] = state_tensor
            self.prev_action[cur_turn] = index
            action = self.env.action_from_int(index.numpy()[0][0])
            (next_state, empty_spaces, next_cur_turn), reward, done, _ = self.env.step(action)
            next_state_tensor = self.state_to_tensor(next_state)

            if done:
                if reward == -10:
                    transition = (state_tensor, index, next_state_tensor, torch.tensor([reward], dtype=torch.float32))
                    self.memories[cur_turn].store(transition)
                else:
                    transition = (
                    state_tensor, index, next_state_tensor, torch.tensor([abs(reward)], dtype=torch.float32))
                    self.memories[cur_turn].store(transition)
                    transition = (self.prev_state[next_cur_turn], self.prev_action[next_cur_turn], next_state_tensor,
                                  torch.tensor([next_cur_turn * reward], dtype=torch.float32))
                    self.memories[next_cur_turn].store(transition)
            else:
                if self.prev_state[next_cur_turn] is not None:
                    transition = (self.prev_state[next_cur_turn], self.prev_action[next_cur_turn], next_state_tensor,
                                  torch.tensor([reward * next_cur_turn], dtype=torch.float32))
                    self.memories[next_cur_turn].store(transition)

            if do_learning:
                self.learn(next_cur_turn)

            state = next_state
            cur_turn = next_cur_turn

    def learn(self, cur_turn):
        if np.min([len(self.memories[cur_turn]), len(self.memories[-cur_turn])]) < self.batch_size:
            return

        transitions = self.memories[cur_turn].sample(self.batch_size)

        batch_state, batch_action, batch_next_state, batch_reward = zip(*transitions)
        batch_state = Variable(torch.stack(batch_state).to(self.device))
        batch_action = Variable(torch.cat(batch_action).to(self.device))
        batch_reward = Variable(torch.cat(batch_reward).to(self.device))
        batch_next_state = Variable(torch.stack(batch_next_state).to(self.device))

        Q = self.models[cur_turn](batch_state)
        Q = Q.gather(1, batch_action).reshape([self.batch_size])
        Qmax = self.models[cur_turn](batch_next_state).detach()
        Qmax = Qmax.max(1)[0]
        Qnext = batch_reward + (self.gamma * Qmax)

        loss = F.smooth_l1_loss(Q, Qnext)
        self.optimizers[cur_turn].zero_grad()
        loss.backward()
        self.optimizers[cur_turn].step()

    def get_policy_reward(self, player, n_episodes=1000):
        rewards = []
        for _ in range(n_episodes):
            self.env.reset()
            state, empty_spaces, cur_turn = self.env.getState()
            done = False
            while not done:
                if cur_turn == player:
                    idx = self.select_greedy_action(self.state_to_tensor(state).to(device), player)
                    action = self.env.action_from_int(idx)
                else:
                    idx = np.random.randint(len(empty_spaces))
                    action = empty_spaces[idx]
                (state, empty_spaces, cur_turn), reward, done, _ = self.env.step(action)
            if reward != -10:
                rewards.append(reward * player)
            else:
                if cur_turn == player:
                    rewards.append(reward)

        return np.array(rewards)


class DoubleDQN(DQN):
    def __init__(self,
                 models,
                 target_models,
                 n_rows=4,
                 n_cols=4,
                 n_win=4,
                 gamma=0.8,
                 device=device,
                 tau=0.01):
        super().__init__(models,
                         n_rows,
                         n_cols,
                         n_win,
                         gamma=gamma,
                         device=device)
        self.target_models = {-1: target_models[0].to(device),
                              1: target_models[1].to(device)}
        self.tau = tau

    def learn(self, cur_turn):
        if np.min([len(self.memories[cur_turn]), len(self.memories[-cur_turn])]) < self.batch_size:
            return

        transitions = self.memories[cur_turn].sample(self.batch_size)

        batch_state, batch_action, batch_next_state, batch_reward = zip(*transitions)
        batch_state = (torch.stack(batch_state).to(self.device))
        batch_action = (torch.cat(batch_action).to(self.device))
        batch_reward = (torch.cat(batch_reward).to(self.device))
        batch_next_state = (torch.stack(batch_next_state).to(self.device))

        Q = self.models[cur_turn](batch_state)
        Q = Q.gather(1, batch_action).reshape([self.batch_size])
        Qmax = self.target_models[cur_turn](batch_next_state).detach()
        Qmax = Qmax.max(1)[0]
        Qnext = batch_reward + (self.gamma * Qmax)

        loss = F.smooth_l1_loss(Q, Qnext)
        self.optimizers[cur_turn].zero_grad()
        loss.backward()
        self.optimizers[cur_turn].step()

        for target_param, param in zip(self.target_models[cur_turn].parameters(), self.models[cur_turn].parameters()):
            target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)
