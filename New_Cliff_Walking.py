import numpy as np
import matplotlib.pyplot as plt


class Mouse():
    def __init__(self):
        # initialise agent with variables
        self.pos = [3,0]
        self.actionvalues = np.zeros(shape=(4,12,4),dtype = float)


def all_possible_moves():
    # define all possible movements
    actionDestination = []
    # iterate through each possible row
    for i in range(0, 4):
        actionDestination.append([])
        # iterate through each possible column
        for j in range(0, 12):
            # create dictionary for each position that contains
            # all possible moves with respect to that position
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
    # while mouse has not reached the terminal position
    while mouse.pos != ending_pos:
        # choose action
        action = choose_action_greedy(epsilon, mouse, mouse.pos)
        # obtain reward
        reward = rewards[mouse.pos[0]][mouse.pos[1]][action]
        # update total reward
        tot_reward += reward
        # obtain new destination
        new_pos = destination[mouse.pos[0]][mouse.pos[1]][action]
        # update using Qlearning update rule
        mouse.actionvalues[mouse.pos[0], mouse.pos[1], action] += alpha * (
            reward + gamma * np.max(mouse.actionvalues[new_pos[0], new_pos[1], :]) -
            mouse.actionvalues[mouse.pos[0], mouse.pos[1], action])
        # set new position
        mouse.pos = new_pos
    return tot_reward


def SARSAlearning(mouse, epsilon, alpha, gamma, destination, ending_pos, rewards):
    # choose an action
    action_curr = choose_action_greedy(epsilon, mouse, mouse.pos)
    tot_reward = 0
    # while mouse has not reached the terminal position
    while mouse.pos != ending_pos:
        # define to 'would be' new state
        state_new = destination[mouse.pos[0]][mouse.pos[1]][action_curr]
        # find out the new action from this new state
        action_new = choose_action_greedy(epsilon, mouse, state_new)
        # get the reward for the old action
        reward = rewards[mouse.pos[0]][mouse.pos[1]][action_curr]
        # update total reward value
        tot_reward += reward
        # update using SARSA update rule
        mouse.actionvalues[mouse.pos[0]][mouse.pos[1]][action_curr] += alpha * (reward +
                (gamma * mouse.actionvalues[state_new[0]][state_new[1]][action_new])
                - mouse.actionvalues[mouse.pos[0]][mouse.pos[1]][action_curr])
        # set the current action as the new action
        action_curr = action_new
        # set the agent position as the new state
        mouse.pos = state_new
    return tot_reward


def choose_action_greedy(epsilon, mouse, state):
    rand_num = np.random.random()
    if rand_num > 1 - epsilon:
        # if random number is greater than 1-epsilon pick random move
        # example 1-0.1 = 0.9 random must be greater than 0.9
        return np.random.randint(4)
    else:
        # obtain all action values from estimation matrix
        action_vals = mouse.actionvalues[state[0]][state[1]][:]
        # choose the action based on maximum action value
        # if more than one have the same action value then pick random
        action = np.random.choice(np.argwhere(action_vals == np.amax(action_vals)).flatten().tolist())
        return action

def main():
    # define environment with rewards
    actionRewards = np.zeros(shape=(4, 12, 4),dtype=float)
    actionRewards[:, :, :] = -1.0
    actionRewards[2, 1:11, 1] = -100.0
    actionRewards[3, 0, 3] = -100.0

    # define all other constants
    ending_pos = [3,11]
    alpha = 0.1
    gamma = 1
    numSteps = 500
    numRuns = 10
    small_epsi = 0.01
    smallest_epsi = 0.0001

    # produce all possible moves and save in array of dictionaries
    destination = all_possible_moves()

    # create agents
    mouse_q = Mouse()
    mouse_sarsa = Mouse()
    mouse_s = Mouse()
    mouse_ss = Mouse()

    # create arrays
    reward_q = np.zeros(numSteps)
    reward_sarsa = np.zeros(numSteps)
    reward_sarsa_s = np.zeros(numSteps)
    reward_sarsa_ss = np.zeros(numSteps)

    # for number of runs
    for z in range(0,numRuns):
        # for number of steps
        for i in range(0, numSteps):
            # do learning algorithms, each being carried out until terminal state has been reached
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

    # average over 10 runs
    reward_q /= numRuns
    reward_sarsa /= numRuns
    reward_sarsa_s /= numRuns
    reward_sarsa_ss /= numRuns

    smooth_rq = np.copy(reward_q)
    smooth_rs = np.copy(reward_sarsa)
    smooth_rss = np.copy(reward_sarsa_s)
    smooth_rsss = np.copy(reward_sarsa_ss)

    # perform moving average over last 19 episodes
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