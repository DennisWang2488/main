import numpy as np
import torch
from torch.utils.data import Dataset
from optModel import optCvModel

class AlphaFairOptDataset(Dataset):
    def __init__(self, a, b, c, r, x, Q, alpha, epsilon):
        """
        Initialization of the dataset.
        Args:
            a, b, c, r, x: Generated data from genData function.
            Q: Upper bound.
            alpha: Alpha parameter for fairness.
            epsilon: Small constant to ensure feasibility.
        """
        self.a = a
        self.b = b
        self.c = c
        self.r = r
        self.x = x
        self.Q = Q
        self.alpha = alpha
        self.epsilon = epsilon
        self.opt_solutions = []
        self.opt_objective_values = []

        self._solve_optimization_problems()

    def _solve_optimization_problems(self):
        """
        Solve the optimization problem for each data point and store the optimal solutions and values.
        """
        n_data = self.a.shape[0]
        for i in range(n_data):
            model = optCvModel(
                n=self.a.shape[1],
                alpha=self.alpha,
                Q=self.Q,
                epsilon=self.epsilon,
                a=self.a[i],
                r=self.r[i],
                b=self.b[i],
                c=self.c[i]
            )
            u_opt, d_opt, opt_value = model.solveP()
            self.opt_solutions.append(d_opt)
            self.opt_objective_values.append(opt_value)
        self.opt_solutions = np.array(self.opt_solutions)
        self.opt_objective_values = np.array(self.opt_objective_values)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        """
        Get a data point and its corresponding optimal solution and objective value.
        Args:
            idx: Index of the data point.
        Returns:
            Tuple of features, true r, optimal solution, and optimal objective value.
        """
        return (
            torch.FloatTensor(self.x[idx]),
            torch.FloatTensor(self.r[idx]),
            torch.FloatTensor(self.opt_solutions[idx]),
            torch.FloatTensor([self.opt_objective_values[idx]])
        )
