import k_armed_testbed as kat
import action_val_method as avm
import numpy as np
import matplotlib.pyplot as plt

size = 10000
reps = 100

def sam_avg(est, alpha, reward):
    return est + (1 / alpha) * (reward - est)

def con_s(est, alpha, reward):
    return est + 0.1 * (reward - est)

# Initialization

avg_rewards = np.zeros(size, dtype=float)
const_rewards = np.zeros(size, dtype=float)
steps = np.arange(1, size + 1, 1, dtype=int)

# Simulation

for i in range(reps):
    if((i % 500) == 0):
        print(i, "steps completed")

    avg_testbed = kat.k_armed_test(10, 0, 1, 0, 0.01)
    const_testbed = kat.k_armed_test(10, 0, 1, 0, 0.01)

    sample_avg = avm.action_val(10, 0.1, sam_avg)
    const_steps = avm.action_val(10, 0.1, con_s)

    for j in range(size):
        avg_action = sample_avg.select_action()
        const_action = const_steps.select_action()

        reward_1 = avg_testbed.give_reward(avg_action)
        reward_2 = const_testbed.give_reward(const_action)

        #print(reward_1)
        #print(reward_2, "\n")

        sample_avg.update_estimates(reward_1, avg_action)
        const_steps.update_estimates(reward_2, const_action)

        avg_rewards[j] += reward_1
        const_rewards[j] += reward_2


avg_rewards /= float(reps)
const_rewards /= float(reps)

#print(avg_rewards)
#print(const_rewards)


# Plotting

plt.plot(steps, avg_rewards, 'b-', label="Sample average method")
plt.plot(steps, const_rewards, 'r-', label="Const stepsize method")
plt.legend()
plt.xlabel("Steps")
plt.ylabel("Reward per step")
plt.title("Testing action-value methods")

plt.show()





