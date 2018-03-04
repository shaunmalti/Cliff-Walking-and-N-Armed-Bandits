import numpy as np
import matplotlib.pyplot as plt


class Mouse():
    def __init__(self):
        self.pos = [3,0]
        self.actionvalues = np.zeros(shape=(4,12,4),dtype = float)


def all_possible_moves():
    actionDestination = []
    for i in range(0, 4):
        actionDestination.append([])
        for j in range(0, 12):
            destination = dict()
            destination[0] = [max(i - 1, 0), j]
            destination[2] = [i, max(j - 1, 0)]
            destination[3] = [i, min(j + 1, 11)]
            if i == 2 and 1 <= j <= 10:
                destination[1] = [3,0]
            else:
                destination[1] = [min(i + 1, 3), j]
            actionDestination[-1].append(destination)
    actionDestination[3][0][3] = [3,0]
    return actionDestination


def Qlearning(mouse, epsilon, alpha, gamma, destination, ending_pos, rewards):
    tot_reward = 0.0
    while mouse.pos != ending_pos:
        action = choose_action_greedy(epsilon, mouse, mouse.pos)
        reward = rewards[mouse.pos[0]][mouse.pos[1]][action]
        tot_reward += reward
        new_pos = destination[mouse.pos[0]][mouse.pos[1]][action]
        mouse.actionvalues[mouse.pos[0], mouse.pos[1], action] += alpha * (
            reward + gamma * np.max(mouse.actionvalues[new_pos[0], new_pos[1], :]) -
            mouse.actionvalues[mouse.pos[0], mouse.pos[1], action])
        mouse.pos = new_pos
    return tot_reward


def SARSAlearning(mouse, epsilon, alpha, gamma, destination, ending_pos, rewards):
    action_curr = choose_action_greedy(epsilon, mouse, mouse.pos)
    tot_reward = 0
    while mouse.pos != ending_pos:
        state_new = destination[mouse.pos[0]][mouse.pos[1]][action_curr]
        action_new = choose_action_greedy(epsilon, mouse, state_new)
        reward = rewards[mouse.pos[0]][mouse.pos[1]][action_curr]
        tot_reward += reward
        mouse.actionvalues[mouse.pos[0]][mouse.pos[1]][action_curr] += alpha * (reward +
                (gamma * mouse.actionvalues[state_new[0]][state_new[1]][action_new])
                - mouse.actionvalues[mouse.pos[0]][mouse.pos[1]][action_curr])
        action_curr = action_new
        mouse.pos = state_new
    return tot_reward


def choose_action_greedy(epsilon, mouse, state):
    rand_num = np.random.random()
    if rand_num > 1 - epsilon:
        return np.random.randint(4)
    else:
        action_vals = mouse.actionvalues[state[0]][state[1]][:]
        action = np.random.choice(np.argwhere(action_vals == np.amax(action_vals)).flatten().tolist())
        return action

def main():
    actionRewards = np.zeros(shape=(4, 12, 4),dtype=float)
    actionRewards[:, :, :] = -1.0
    actionRewards[2, 1:11, 1] = -100.0
    actionRewards[3, 0, 3] = -100.0

    ending_pos = [3,11]
    alpha = 0.1
    gamma = 1
    numSteps = 500
    numRuns = 10

    small_epsi = 0.01
    smallest_epsi = 0.0001

    destination = all_possible_moves()

    mouse_q = Mouse()
    mouse_sarsa = Mouse()
    mouse_s = Mouse()
    mouse_ss = Mouse()

    reward_q = np.zeros(numSteps)
    reward_sarsa = np.zeros(numSteps)
    reward_sarsa_s = np.zeros(numSteps)
    reward_sarsa_ss = np.zeros(numSteps)

    for z in range(0,numRuns):
        for i in range(0, numSteps):
            tot_reward_q = Qlearning(mouse_q, 0.1, alpha, gamma, destination, ending_pos, actionRewards)
            reward_q[i] += tot_reward_q
            mouse_q.pos = [3,0]

            tot_reward_sarsa = SARSAlearning(mouse_sarsa, 0.1, alpha, gamma, destination, ending_pos, actionRewards)
            reward_sarsa[i] += tot_reward_sarsa
            mouse_sarsa.pos = [3,0]

            tot_reward_sarsa_s = SARSAlearning(mouse_s, small_epsi, alpha, gamma, destination, ending_pos,
                                             actionRewards)
            reward_sarsa_s[i] += tot_reward_sarsa_s
            mouse_s.pos = [3, 0]

            tot_reward_sarsa_ss = SARSAlearning(mouse_ss, smallest_epsi, alpha, gamma, destination, ending_pos,
                                               actionRewards)
            reward_sarsa_ss[i] += tot_reward_sarsa_ss
            mouse_ss.pos = [3, 0]

    reward_q /= numRuns
    reward_sarsa /= numRuns
    reward_sarsa_s /= numRuns
    reward_sarsa_ss /= numRuns

    smooth_rq = np.copy(reward_q)
    smooth_rs = np.copy(reward_sarsa)
    smooth_rss = np.copy(reward_sarsa_s)
    smooth_rsss = np.copy(reward_sarsa_ss)

    for i in range(numRuns, numSteps):
        smooth_rs[i] = np.mean(reward_sarsa[i - numRuns: i + 1])
        smooth_rq[i] = np.mean(reward_q[i - numRuns: i + 1])
        smooth_rss[i] = np.mean(reward_sarsa_s[i - numRuns: i + 1])
        smooth_rsss[i] = np.mean(reward_sarsa_ss[i - numRuns: i + 1])

    plt.plot(smooth_rq, label='qlearning')
    plt.plot(smooth_rs, label='sarsa 0.1 epsilon')
    plt.plot(smooth_rss, label='sarsa 0.01 epsilon') # best at 0.01 epsilon
    plt.plot(smooth_rsss, label='sarsa 0.0001 epsilon')
    plt.ylim(-100,0)
    plt.ylabel("Cumulive Reward Averaged Over 10 Runs")
    plt.xlabel("Steps")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()