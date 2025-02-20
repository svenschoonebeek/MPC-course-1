import numpy as np

from Agent import Agent
import matplotlib.pyplot as plt

class Controller:

    def __init__(self):
        self.x_0 = np.array([-5, 5])
        self.A = np.vstack(([0.9, 0.1], [0, 0.95]))
        self.B = np.eye(2, 2)
        self.beta = 5
        self.Q = np.vstack(([self.beta, 0], [0, self.beta]))
        self.R = np.eye(2, 2)
        self.P = np.vstack(([1.48, 0.06], [0.06, 1.56]))
        self.Np = 20
        self.agent = None
        self.max_iter = 20
        self.bounds = [(-1, 1)] * 2 * self.Np
        self.state_constraint = [-5, 5]
        self.penalty_weight = 10

    def plot(self):
        fig = plt.figure(figsize=(10, 6))
        figs = fig.subfigures(1, 2)
        ax2 = figs[0].add_subplot(1, 1, 1)
        ax2.plot(range(self.max_iter), self.agent.u_k)
        ax2.set_title("Input over time")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Input")
        ax3 = figs[1].add_subplot(1, 1, 1)
        ax3.plot(range(self.max_iter), self.agent.x_seq[:-1])
        ax3.set_title("State over time")
        ax3.set_xlabel("Time")
        ax3.set_ylabel("State")
        plt.show()

    def run(self):
        self.agent = Agent(self.x_0, self.A, self.B, self.Q, self.R, self.P, self.Np,
                           self.state_constraint, self.bounds, self.penalty_weight)
        for i in range(self.max_iter):
            print("Running at iteration " + str(i + 1))
            self.agent.minimize_objective_function()
        self.plot()

controller = Controller()
controller.run()










