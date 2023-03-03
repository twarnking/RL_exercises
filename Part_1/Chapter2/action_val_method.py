import numpy as np
import random as rn


class action_val:
    def __init__(self, k, eps, val_func):
        self.size = k
        self.epsilon = eps
        self.value_function = val_func
        self.estimates = np.zeros(k)
        self.steps = np.zeros(k)
    
    def update_estimates(self, R_a, i):
        self.steps[i] += 1
        self.estimates[i] = self.value_function(self.estimates[i], self.steps[i], R_a)
    

    def select_action(self):
        if (self.epsilon >= rn.random()):
            return int(rn.uniform(0, self.size))
        else:
            return np.argmax(self.estimates)
    
    def optimistic_init(self, value):
        self.estimates = np.full(self.size, value)



    