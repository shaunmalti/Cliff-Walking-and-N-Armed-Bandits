import numpy as np
import matplotlib.pyplot as plt


class Mouse():
    def __init__(self):
        # self.estimation = np.zeros(shape=(4,12,4), dtype= float)
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
    i = 0
    while mouse.pos != ending_pos:
        # TODO: IN OTHER METHOD - HE DEFINES ALL POSSIBLE MOVES FROM BEFORE
        i+=1
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
# TODO CHECK THIS OUT


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

    destination = all_possible_moves()

    mouse_q = Mouse()
    mouse_sarsa = Mouse()
    reward_q = np.zeros(500)
    reward_sarsa = np.zeros(500)
    for z in range(0,10):
        for i in range(0, numSteps):
            tot_reward_q = Qlearning(mouse_q, 0.1, alpha, gamma, destination, ending_pos, actionRewards)
            reward_q[i] += tot_reward_q
            mouse_q.pos = [3,0]

            tot_reward_sarsa = SARSAlearning(mouse_q, 0.1, alpha, gamma, destination, ending_pos, actionRewards)
            reward_sarsa[i] += tot_reward_sarsa
            mouse_sarsa.pos = [3,0]

    avg_reward_q = reward_q/10
    # avg_reward_sarsa = reward_sarsa/10
    #
    plt.plot(avg_reward_q)
    # plt.plot(avg_reward_sarsa)
    plt.show()



if __name__ == '__main__':
    main()