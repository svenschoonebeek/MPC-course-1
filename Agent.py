import numpy as np
from scipy.optimize import minimize

class Agent:

    def __init__(self, x_0, A, B, C, D, Q, R, Np):
        self.x_k = x_0
        self.x_pred = np.tile(x_0, Np)
        self.u_k = 0
        self.u_pred = np.zeros(Np)
        self.y_k = []
        self.Np = Np
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.Q = Q
        self.R = R

    def predict_state(self, u):
        return self.A @ self.x_pred + self.B * u

    def compute_output(self):
        self.y_k.append(self.C @ self.x_k + self.D * self.u_pred[0])

    def objective_function(self, u):
        J = 0
        self.x_pred = self.x_k
        for i in range(self.Np):
            J += self.x_pred.T @ self.Q @ self.x_pred + self.R * u[i]**2
            self.x_pred = self.predict_state(u[i])
        return J

    def minimize_objective_function(self):
        result = minimize(self.objective_function, self.u_pred, method='SLSQP', options={'maxiter': 100, 'ftol': 1e-6, 'disp': True, 'step_size': 0.01})
        self.u_pred = result.x
        self.compute_output()
        self.x_k = self.predict_state(self.u_pred[0])








