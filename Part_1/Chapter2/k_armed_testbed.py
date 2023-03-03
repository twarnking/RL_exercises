import numpy as np

class k_armed_test:
    def __init__(self, k, init_m, init_sd, c_m, c_sd):
        self.q_star_ = np.random.normal(init_m, init_sd, k)
        self.arms = k
        self.change_mean = c_m
        self.change_sd = c_sd
    
    def one_step(self):
        temp = np.random.normal(self.change_mean, self.change_sd, self.arms)
        self.q_star_ += temp
    
    def give_reward(self, A_t):
        self.one_step()
        return float(np.random.normal(self.q_star_[A_t], 1))

    def print_a_v(self):
        """Prints the real action values of the testbed"""
        print(self.q_star_)