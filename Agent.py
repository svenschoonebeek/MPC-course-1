import numpy as np
from scipy.optimize import minimize

class Agent:

    def __init__(self, x_0, A, B, Q, R, P, Np, state_constraint, bounds, penalty_weight):
        self.x_k = x_0
        self.x_pred = np.tile(x_0, Np)
        self.x_seq = [x_0]
        self.u_k = []
        self.u_pred = np.zeros(2 * Np)
        self.Np = Np
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.P = P
        self.state_constraint = state_constraint
        self.bounds = bounds
        self.penalty_weight = penalty_weight

    def predict_state(self, u):
        return self.A @ self.x_pred + self.B @ u

    def objective_function(self, u):
        J, penalty = 0, 0
        self.x_pred = self.x_k
        for i in range(self.Np):
            u_i = np.array(u[2 * i:2 * i + 2])
            J += self.x_pred.T @ self.Q @ self.x_pred + u_i.T @ self.R @ u_i
            self.x_pred = self.predict_state(u_i)
            if self.state_constraint is not None:
                violation = max(0, self.state_constraint[0] - self.x_pred[0]) + max(0, self.x_pred[0] - self.state_constraint[1])
                penalty += self.penalty_weight * violation ** 2
        return 0.5 * J + penalty + 0.5 * self.x_pred.T @ self.P @ self.x_pred

    def minimize_objective_function(self):
        result = minimize(self.objective_function, np.array(self.u_pred).flatten(), bounds=self.bounds, method='SLSQP', options={'maxiter': 100, 'ftol': 1e-8, 'disp': True})
        self.u_pred = result.x
        self.u_k.append(self.u_pred[:2])
        self.x_k = self.predict_state(self.u_pred[:2])
        self.x_seq.append(self.x_k)




