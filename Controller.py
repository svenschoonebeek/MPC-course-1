import numpy as np
from Agent import Agent
import matplotlib.pyplot as plt

class Controller:

    def __init__(self):
        self.x_0 = np.array([10,0])
        self.A = np.vstack(([1, 0.1], [0, 1]))
        self.B = np.array([0, 0.1])
        self.C = np.eye(2, 2)
        self.D = np.array([0, 0])
        self.Q = np.eye(2, 2)
        self.R = 1
        self.Np = 10
        self.agent = None
        self.max_iter = 30

    def run(self):
        self.agent = Agent(self.x_0, self.A, self.B, self.C, self.D, self.Q, self.R, self.Np)
        for i in range(self.max_iter):
            print("Running at iteration " + str(i + 1))
            self.agent.minimize_objective_function()
        plt.plot(range(self.max_iter), self.agent.y_k)
        plt.show()

controller = Controller()
controller.run()










