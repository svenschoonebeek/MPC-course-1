import numpy as np
from scipy.optimize import minimize

class Agent:

    def __init__(self, x_0, A, B, C, D, Q, R, Np, state_constraint, bounds, terminal_constraint, penalty_weight):
        self.x_k = x_0
        self.x_pred = np.tile(x_0, Np)
        self.x_seq = [x_0]
        self.u_k = []
        self.u_pred = np.zeros(Np)
        self.y_k = []
        self.Np = Np
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.Q = Q
        self.R = R
        self.state_constraint = state_constraint
        self.terminal_constraint = terminal_constraint
        self.constraints = None
        self.bounds = bounds
        self.penalty_weight = penalty_weight

    def predict_state(self, u):
        return self.A @ self.x_pred + self.B * u

    def compute_output(self):
        self.y_k.append(self.C @ self.x_k + self.D * self.u_pred[0])

    def objective_function(self, u):
        J, penalty = 0, 0
        self.x_pred = self.x_k
        for i in range(self.Np):
            J += self.x_pred.T @ self.Q @ self.x_pred + self.R * u[i]**2
            self.x_pred = self.predict_state(u[i])
            if self.state_constraint is not None:
                violation = max(0, self.state_constraint - self.x_pred[0])
                penalty += self.penalty_weight * violation ** 2
            if self.terminal_constraint is not None and i == self.Np - 1:
                violation = max(0, np.abs(self.x_pred[1]) - self.terminal_constraint)
                penalty += self.penalty_weight * violation ** 2
        return J + penalty

    def minimize_objective_function(self):
        result = minimize(self.objective_function, self.u_pred, bounds=self.bounds, method='SLSQP', options={'maxiter': 100, 'ftol': 1e-6, 'disp': True})
        self.u_pred = result.x
        self.compute_output()
        self.u_k.append(self.u_pred[0])
        self.x_k = self.predict_state(self.u_pred[0])
        self.x_seq.append(self.x_k)








