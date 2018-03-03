import numpy as np
import matplotlib.pyplot as plt
import operator

# environment

class Mouse():
    def __init__(self):
        # self.estimation = np.zeros(shape=(4,12,4), dtype= float)
        self.pos = [3,0]
        self.actionvalues = np.zeros(shape=(4,4,12),dtype = float)

def initialise_rewards(environment):
    for i in range(0,environment.shape[0]):
        for j in range(0,environment.shape[1]):
            if j == 0 and i == 3 or i == 3 and j == 11:
                environment[i][j] = -1
            elif j > 0 and i == 3 or i == 3 and j < 11:
                environment[i][j] = -100
            else:
                environment[i][j] = -1

def initialise_moves():
    moves_dict = {}
    moves_dict["move_up"] = 0
    moves_dict["move_right"] = 1
    moves_dict["move_down"] = 2
    moves_dict["move_left"] = 3
    print(moves_dict)
    return moves_dict

# def SARSA():

# TODO: CURRENTLY WHEN ON RIGHTMOST COLUMN SECOND ROW - GOES UP INSTEAD OF CHOOSING DOWN
# TODO: DOWN IS THE BEST CHOICE BUT IT PICKS UP
def Qlearning(mouse, ending_pos, epsilon, alpha, gamma, environment):
    tot_reward = 0.0
    i = 0
    while mouse.pos != ending_pos:
        i+= 1
        action = choose_action_greedy(epsilon, mouse)
        reward,newstate = get_reward_and_newstate(mouse,action, environment)
        tot_reward += reward
        # TODO: UPDATE ESTIMATION
        mouse.actionvalues[action][mouse.pos[0]][mouse.pos[1]]+= alpha * (
            reward + gamma * mouse.actionvalues[action][newstate[0]][newstate[1]] -
            mouse.actionvalues[action][mouse.pos[0]][mouse.pos[1]])
        # TODO: CHECK IF ON CLIFF
        if 1 <= newstate[1] <= 11 and newstate[0] == 3:
            mouse.pos = [3,0]
            continue
        else:
            mouse.pos = newstate
        # TODO: ELSE CHANGE POSITION
        print("________________________________________________________________________________________________")
        print(mouse.actionvalues)
        print("________________________________________________________________________________________________")
        print(i)
    return tot_reward


def get_reward_and_newstate(mouse, action, environment):
    reward = 0
    if action == 0:  # move up
        reward = environment[mouse.pos[0]-1][mouse.pos[1]]
        newstate = [mouse.pos[0]-1,mouse.pos[1]]
    elif action == 1:
        reward = environment[mouse.pos[0]][mouse.pos[1]+1]
        newstate = [mouse.pos[0], mouse.pos[1]+1]
    elif action == 2:
        reward = environment[mouse.pos[0]+1][mouse.pos[1]]
        newstate = [mouse.pos[0]+1, mouse.pos[1]]
    elif action == 3:
        reward = environment[mouse.pos[0]][mouse.pos[1]-1]
        newstate = [mouse.pos[0], mouse.pos[1]-1]
    return reward, newstate


def choose_action_greedy(epsilon, mouse):
    # TODO: CURRENTLY IGNORING RANDOM
    # random_num = np.random.random()
    # if random_num > 1-epsilon:
    #     poss_moves = find_possible_moves(mouse.pos)
    #     return np.argmax(np.random.random(len(poss_moves)))  # returns max prob of argument, i.e. this returns a number which \
    #     # corresponds to a move, up right down left
    # else:
    poss_moves = find_possible_moves(mouse.pos)
    # TODO: RETURN ACTION VALUES FOR PREVIOUS POSITIONS
    # action_vals = return_col_vals(mouse.actionvalues,mouse.pos[0],mouse.pos[1])
    action_vals, new_dict = return_prev_position_vals(mouse)

    # for i in range(0,len(poss_moves)):
    #     for j in range(0, len(action_vals)):
    #     # best_move = np.argsort(action_vals)
    #         if poss_moves[i] == np.argmax(action_vals):
    #             return poss_moves[i]
    #         else:
    #             action_vals[np.argmax(action_vals)] = -1000
    #             continue
    # dictionary = dict(zip(poss_moves,action_vals))
    return max(new_dict.items(), key=operator.itemgetter(1))[0]

    # TODO CHECK THIS OUT MAKE SURE IT WORKS

        # if best_move == poss_moves



def find_possible_moves(curr_pos):
    if curr_pos[0] == 3 and curr_pos[1] == 0:  # mouse pos = [0,3] i.e. starting pos poss moves are up or right
        return [0,1]
    elif curr_pos[0] == 0 and curr_pos[1] == 0:
        return [1,2]
    elif curr_pos[0] == 0 and curr_pos[1] == 11:
        return [2,3]
    elif curr_pos[0] == 2 and 1 <= curr_pos[1] <= 0:
        return [0,1,2]
    elif curr_pos[1] == 11 and 1 <= curr_pos[0] <= 2:
        return [0,2,3]
    elif curr_pos[0] == 0 and 1 <= curr_pos[1] <= 11:
        return [1,2,3]
    else:
        return [0,1,2,3]

# def choose_best():

def return_col_vals(array,i,j):
    return [row[i][j] for row in array]


def return_prev_position_vals(mouse):
    # TODO THIS HAS TO RETURN AN ARRAY
    pos_holder = [mouse.pos[0],mouse.pos[1]]

    array_vals = np.array([[pos_holder[0]-1,pos_holder[1],0]])
    array_vals = np.append(array_vals, [[pos_holder[0],pos_holder[1]+1,1]], axis=0)
    array_vals = np.append(array_vals, [[pos_holder[0]+1, pos_holder[1],2]], axis=0)
    array_vals = np.append(array_vals, [[pos_holder[0], pos_holder[1] - 1,3]], axis=0)

    for x in range(0,np.shape(array_vals)[0]):
        for z in range(0,np.shape(array_vals)[1]):
            try:
                if array_vals[x][z] < 0:
                    array_vals = np.delete(array_vals,(x),axis=0)
            except IndexError:
                continue

    move_val_dict = dict()
    prev_pos_vals = []
    for i in range(0,len(array_vals)):
        try:
            prev_pos_vals = np.append(prev_pos_vals, mouse.actionvalues[array_vals[i][2]][array_vals[i][0]][array_vals[i][1]])
            move_val_dict.update({array_vals[i][2]:mouse.actionvalues[array_vals[i][2]][array_vals[i][0]][array_vals[i][1]]})
        except IndexError:
            continue



    return prev_pos_vals, move_val_dict



def main():
    #  TODO: define environment
    environment = np.zeros(shape=(4,12), dtype= float)
    ending_pos = [3,11]
    alpha = 0.1
    gamma = 1
    initialise_rewards(environment)
    moves = initialise_moves()
    numEpi = 500

    # test_array = np.ones(shape=(4,4))
    # test = 123
    # test = np.ones(shape=(4,4,12),dtype = float)
    # test[0][2][0] = 0
    # answer = return_col_vals(test,0,2)


    # for i in range(0, numEpi):
    mouse = Mouse()
    Qlearning(mouse, ending_pos, 0.1, alpha, gamma, environment)


    test = 123
#  TODO: define learning algorithms
#  TODO: define run

# TODO: CREATE VALUES FOR EVERY POSITION THROUGH A 12 BY 3 BY 4 ARRAY


if __name__ == '__main__':
    main()