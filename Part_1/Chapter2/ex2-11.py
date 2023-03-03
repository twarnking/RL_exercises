import k_armed_testbed as kat
import action_val_method as avm
import numpy as np
import matplotlib.pyplot as plt

# Methods
# 1 -> e-greedy with sample average
# 2 -> e-greedy with constant-step-size
# 3 -> greedy with optimistic init values


#Parameters
alp = 0.1
runs = 200000


# Rewards functions
def sam_avg(est, alpha, reward):
    return est + (1 / alpha) * (reward - est)

def con_s(est, alpha, reward):
    return est + alp * (reward - est)


# parameters for e-greedy
eps = np.array([1/128, 1/64, 1/32, 1/16, 1/8, 1/4])
method1_rewards = np.zeros(6)



# Calculating six data points for each method
for i in range(6):
    # Creating testbeds
    testbed_1 = kat.k_armed_test(10, 0, 1, 0, 0.01)


    # Initializing methods
    method_1 = avm.action_val(10, eps[i], sam_avg)


    for j in range(runs):
        # Method 1
        action_1 = method_1.select_action()
        reward_1 = testbed_1.give_reward(action_1)
        method_1.update_estimates(reward_1, action_1)
        if(j >= runs/2):
            method1_rewards[i] += reward_1


method1_rewards /= float(runs/2)



# Plotting

plt.plot(eps, method1_rewards, 'b-', label="e-greedy with sample average method")
plt.legend()
plt.xlabel("Parameters")
plt.ylabel("Average rewards per 100000 steps")
plt.title("Testing methods for k-armed-bandits")

plt.show()
