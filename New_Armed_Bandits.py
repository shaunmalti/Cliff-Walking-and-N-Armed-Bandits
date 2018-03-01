import numpy as np
import matplotlib.pyplot as plt


def epsilon_greedy(epsilon,est_values):
    rand_num = np.random.random()
    if epsilon > rand_num:
        return np.random.randint(10)
    else:
        return np.argmax(est_values)


def get_reward(action, arm_values):
    val1 = np.random.normal(0,1)
    reward = arm_values[0][action] + val1
    best_reward = np.argmax(arm_values[0])
    return reward, best_reward


def update_est(action,reward, K, est_values):
    K[0][action] += 1
    alpha = 1. / K[0][action]
    est_values[0][action] += alpha * (reward - est_values[0][action])  # keeps running average of rewards


def choose_eps_greedy(epsilon, est_values):
    rand_num = np.random.random()
    if rand_num > 1-epsilon:
        return np.random.randint(10)
    else:
        return np.argmax(est_values)


def choose_greedy(est_values):
    return np.argmax(est_values)


def experiment(arm_vals, K, est_values, Npulls,epsilon, option, avg_outcome_eps0p01):
    history = []
    reward = []
    for i in range(Npulls):
        if option == 1:
            action = choose_eps_greedy(epsilon, est_values)
        else:
            action = choose_greedy(est_values)
        R, best_reward = get_reward(action, arm_vals)
        if action==best_reward:
            reward.append(1)
        else:
            reward.append(0)
        update_est(action, R, K, est_values)
        history.append(R)
    avg_outcome_eps0p01 += np.array(history)
    # return np.array(history)
    return np.array(reward)


def main():
    Nexp = 200
    Npulls = 1000
    avg_outcome_eps0p01 = np.zeros(Npulls)
    avg_outcome_eps0p1 = np.zeros(Npulls)
    avg_outcome_eps_greedy = np.zeros(Npulls)
    avg_perc_eps0p01 = np.zeros(Npulls)
    avg_perc_eps0p1 = np.zeros(Npulls)
    avg_perc_eps_greedy = np.zeros(Npulls)
    for i in range(Nexp):
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

    avg_outcome_eps0p01 /= np.float(Nexp)
    avg_outcome_eps0p1 /= np.float(Nexp)
    avg_outcome_eps_greedy /= np.float(Nexp)

    avg_perc_eps0p01 /= np.float(Nexp)
    avg_perc_eps0p1 /= np.float(Nexp)
    avg_perc_eps_greedy /= np.float(Nexp)

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