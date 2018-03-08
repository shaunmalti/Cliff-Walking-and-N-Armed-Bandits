import numpy as np
import matplotlib.pyplot as plt


def get_reward(action, arm_values):
    # retrieve reward from reward array
    val1 = np.random.normal(0,1)
    reward = arm_values[0][action] + val1
    best_reward = np.argmax(arm_values[0])
    return reward, best_reward


def update_est(action,reward, K, est_values):
    # update all values, together with estimation and pick count
    K[0][action] += 1
    alpha = 1. / K[0][action]
    est_values[0][action] += alpha * (reward - est_values[0][action])


def choose_eps_greedy(epsilon, est_values):
    rand_num = np.random.random()
    if rand_num > 1-epsilon:
        # if random number is greater than 1-epsilon pick random
        # i.e. > 0.9 or 10%
        return np.random.randint(10)
    else:
        # choose the best action
        return np.argmax(est_values)


def choose_greedy(est_values):
    return np.argmax(est_values)


def experiment(arm_vals, K, est_values, Npulls,epsilon, option, avg_outcome_eps0p01):
    history = []
    reward = []
    # for the number of defined pulls
    for i in range(Npulls):
        # choose action method depending on function call
        if option == 1:
            action = choose_eps_greedy(epsilon, est_values)
        else:
            action = choose_greedy(est_values)

        # get reward from carried out action
        R, best_reward = get_reward(action, arm_vals)

        # this is done to check if action picked is optimal and then chart % optimal pick
        if action==best_reward:
            reward.append(1)
        else:
            reward.append(0)

        # update agent estimations
        update_est(action, R, K, est_values)

        # update reward gained array
        history.append(R)
    # add history array to carry out average at the end
    avg_outcome_eps0p01 += np.array(history)
    return np.array(reward)


def main():
    # initialisation of number of episodes and number of pulls per episode
    Nexp = 2000
    Npulls = 1000

    # initialisation of arrays
    avg_outcome_eps0p01 = np.zeros(Npulls)
    avg_outcome_eps0p1 = np.zeros(Npulls)
    avg_outcome_eps_greedy = np.zeros(Npulls)
    avg_perc_eps0p01 = np.zeros(Npulls)
    avg_perc_eps0p1 = np.zeros(Npulls)
    avg_perc_eps_greedy = np.zeros(Npulls)

    # for number of episodes
    for i in range(Nexp):
        # set array values and start experiment for different algorithms
        # and different values of epsilon
        arm_vals = np.array([[np.random.normal(0, 1, 10)]])
        K = np.array([[np.zeros(10)]])
        est_values = np.array([[np.zeros(10)]])
        avg_perc_eps0p01 += experiment(arm_vals[0], K[0], est_values[0], Npulls, 0.01, 1, avg_outcome_eps0p01)

        arm_vals = np.array([[np.random.normal(0, 1, 10)]])
        K = np.array([[np.zeros(10)]])
        est_values = np.array([[np.zeros(10)]])
        avg_perc_eps0p1 += experiment(arm_vals[0], K[0], est_values[0], Npulls, 0.1, 1, avg_outcome_eps0p1)

        arm_vals = np.array([[np.random.normal(0, 1, 10)]])
        K = np.array([[np.zeros(10)]])
        est_values = np.array([[np.zeros(10)]])
        avg_perc_eps_greedy += experiment(arm_vals[0], K[0], est_values[0], Npulls, 0, 2, avg_outcome_eps_greedy)

    # take average for rewards
    avg_outcome_eps0p01 /= np.float(Nexp)
    avg_outcome_eps0p1 /= np.float(Nexp)
    avg_outcome_eps_greedy /= np.float(Nexp)

    # take average for % best pick
    avg_perc_eps0p01 /= np.float(Nexp)
    avg_perc_eps0p1 /= np.float(Nexp)
    avg_perc_eps_greedy /= np.float(Nexp)

    # plots
    plt.plot(avg_outcome_eps0p01, label="eps = 0.01")
    plt.plot(avg_outcome_eps0p1, label="eps = 0.1")
    plt.plot(avg_outcome_eps_greedy, label="greedy")

    plt.ylim(0, 2.2)
    plt.xlabel("Steps")
    plt.ylabel("Average Reward")
    plt.legend()
    plt.show()

    plt.plot(avg_perc_eps0p01, label="eps = 0.01")
    plt.plot(avg_perc_eps0p1, label="eps = 0.1")
    plt.plot(avg_perc_eps_greedy, label="greedy")
    plt.ylim(0,1)
    plt.xlabel("Steps")
    plt.ylabel("Percentage Optimal Action")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()