import numpy as np


def obs_to_state(obs):
    obs = (obs[0] - 4, obs[1] - 1) + obs[2:]
    
    return tuple(map(int, obs))


def calculate_reward(env, pi, episodes):
    env.seed(10)
    total_reward = 0
    for _ in range(episodes):
        obs = env.reset()
        state = obs_to_state(obs)
        done = False
        while not done:
            action = pi[state]
            obs, reward, done, _ = env.step(action)
            state = obs_to_state(obs)
            if done:
                total_reward += reward
    
    return total_reward


def run_episode_actions(env, pi, eps, nA):
    obs = env.reset()
    state = obs_to_state(obs)
    next_action = pi[state] if np.random.rand() > eps else np.random.randint(nA)
    states, actions, rewards = [state], [next_action], [0]
    done = False
    while not done:
        obs, reward, done, _ = env.step(next_action)
        state = obs_to_state(obs)
        states.append(state)
        next_action = pi[state] if np.random.rand() > eps else np.random.randint(nA)
        actions.append(next_action)
        rewards.append(reward)
    
    return states, actions, rewards


def get_random_Q(nS, nA, final_states):
    Q = np.random.random(size=nS + (nA, ))
    Q[final_states, :, :, :] = 0.0
    
    return Q


def init_C(nS, nA):
    
    return np.zeros(nS + (nA, ), dtype=np.float32)


def compute_policy_by_Q(Q, nS):
    
    return np.argmax(Q, axis=len(nS))


def update_C_and_Q(s, a, g, w, Q, C):
    C[s + (a, )] = C[s + (a, )] + w
    Q[s + (a, )] = Q[s + (a, )] + (g - Q[s][a]) * w / C[s][a]
    
    return Q, C


def update_returns_actions_offpolicy_MC(Q, C, pi, states, actions, rewards, nA, epsilon=0.1, gamma=1.0):
    g, w, prob_best_action = 0., 1., 1 - (nA - 1) * epsilon / nA
    Q, C = update_C_and_Q(states[-1], actions[-1], g, w, Q, C)
    for t in range(len(states) - 2, -1, -1):
        if actions[t + 1] != pi[states[t + 1]]:
            break
        w = w / prob_best_action
        g = g * gamma + rewards[t + 1]
        Q, C = update_C_and_Q(states[t], actions[t], g, w, Q, C)

    return Q, C
