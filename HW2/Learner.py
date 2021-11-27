import numpy as np
from tqdm import tqdm


class Learner:
    def __init__(self, env, pi1, pi2):
        self.pi = (pi1, pi2)
        self.env = env

    def step(self, pi1, pi2, prev_board_hash_2, prev_action_2, board_hash, empty_spaces):
        n_empty = len(empty_spaces)
        if pi1 is not None:
            # greedy
            index = pi1.eps_gready(board_hash, n_empty)
        else: 
            # random
            index = np.random.randint(n_empty)  
        action = empty_spaces[index]
        state, reward, done, _ = self.env.step(action)
        board_hash_new, empty_spaces_new, cur_turn_new = state
        pi2.update_q(prev_board_hash_2, board_hash_new, prev_action_2, reward) 
        
        if (reward == 1) or (reward == -1):
            pi1.set_q(board_hash, index, abs(reward))
        prev_board_hash_1 = board_hash
        prev_action_1 = index
        board_hash, empty_spaces, cur_turn = board_hash_new, empty_spaces_new, cur_turn_new
        
        return prev_board_hash_1, prev_action_1, board_hash, empty_spaces, cur_turn, done

    def get_policy_reward(self, pi1=None, pi2=None, n_episodes=100):
        total_reward = 0
        for _ in range(n_episodes):
            total_reward += self.game_inference(pi1, pi2)
        total_reward = total_reward / n_episodes
        
        return total_reward

    def game_train(self):
        self.env.reset()
        board_hash, empty_spaces, cur_turn = self.env.getState()

        done = False
        prev_action = [None, None]
        prev_board_hash = [None, None]

        while not done:
            if cur_turn == 1:
                index1 = 0
                index2 = 1
            else:
                index1 = 1
                index2 = 0
            prev_board_hash[index1], prev_action[index1], board_hash, empty_spaces, cur_turn, done = \
                self.step(self.pi[index1], self.pi[index2], prev_board_hash[index2], prev_action[index2], board_hash, empty_spaces)
          
    def game_inference(self, pi1, pi2):
        self.env.reset()
        
        board_hash, empty_spaces, cur_turn = self.env.getState()
        reward = 0
        done = False
       
        while not done:
            if cur_turn == 1:
                pi = pi1
            else:
                pi = pi2
            n_empty = len(empty_spaces)    
            if pi is not None:
                # greedy
                index = pi.gready(board_hash, n_empty)
            else:
                # random
                index = np.random.randint(n_empty)

            action = empty_spaces[index]
            state, reward, done, _ = self.env.step(action)
            board_hash, empty_spaces, cur_turn = state

        return reward
    

def learn(learner, epochs, n_episodes, period):
    crosses_win = []
    zeros_win = [] 

    for i in tqdm(range(1, epochs + 1)):
        learner.run_episode(do_learning=True)
        if i % period == 0:
            crosses_win.append(np.sum(learner.get_policy_reward(1, n_episodes) == 1) / n_episodes)
            zeros_win.append(np.sum(learner.get_policy_reward(-1, n_episodes) == 1) / n_episodes)

    return (crosses_win, zeros_win)
