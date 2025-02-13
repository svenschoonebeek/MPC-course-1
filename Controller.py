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
        self.bounds = [(-1, 1)] * self.Np
        self.state_constraint = None
        self.terminal_constraint = 1e-4
        self.penalty_weight = 10

    def plot(self):
        fig = plt.figure(figsize=(10, 6))
        figs = fig.subfigures(1, 3)
        ax1 = figs[0].add_subplot(1, 1, 1)
        ax1.plot(range(self.max_iter), self.agent.y_k)
        ax1.set_title("Output over time")
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Output")
        ax2 = figs[1].add_subplot(1, 1, 1)
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
        self.agent = Agent(self.x_0, self.A, self.B, self.C, self.D, self.Q, self.R, self.Np,
                           self.state_constraint, self.bounds, self.terminal_constraint, self.penalty_weight)
        for i in range(self.max_iter):
            print("Running at iteration " + str(i + 1))
            self.agent.minimize_objective_function()
        self.plot()

controller = Controller()
controller.run()










