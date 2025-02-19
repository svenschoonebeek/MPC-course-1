import numpy as np

from Agent import Agent
import matplotlib.pyplot as plt

class Controller:

    def __init__(self):
        self.x_0 = np.array([2, -2])
        self.A = np.vstack(([2, 1], [0, 2]))
        self.B = np.eye(2, 2)
        self.alpha = 0.1
        self.Q = np.vstack(([self.alpha, 0], [0, self.alpha]))
        self.R = np.eye(2, 2)
        self.P = np.vstack(([2.7, 1.8], [1.8, 4.2]))
        self.Np = 20
        self.agent = None
        self.max_iter = 5
        self.bounds = [(-1, 1)] * 2 * self.Np
        #self.state_constraint = [-5, 5]
        self.state_constraint = None
        self.penalty_weight = 10

    def plot(self):
        fig = plt.figure(figsize=(10, 6))
        figs = fig.subfigures(1, 3)
        ax2 = figs[1].add_subplot(1, 1, 1)
        print(self.agent.u_k)
        ax2.plot(range(self.max_iter), self.agent.u_k)
        ax2.set_title("Input over time")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Input")
        ax3 = figs[2].add_subplot(1, 1, 1)
        ax3.plot(range(self.max_iter), self.agent.x_seq[:-1])
        ax3.set_title("State over time")
        ax3.set_xlabel("Time")
        ax3.set_ylabel("State")
        plt.show()

    def run(self):
        self.agent = Agent(self.x_0, self.A, self.B, self.Q, self.R, self.Np,
                           self.state_constraint, self.bounds, self.penalty_weight)
        for i in range(self.max_iter):
            print("Running at iteration " + str(i + 1))
            self.agent.minimize_objective_function()
        self.plot()

controller = Controller()
controller.run()










