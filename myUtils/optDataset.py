import numpy as np
import torch
from torch.utils.data import Dataset
from optModel import optModelAr, optModelRd

class optDatasetAr(Dataset):
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
            model = optModelAr(
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
    
class optDatasetRd(Dataset):
    def __init__(self, x, r, c, Q, alpha, num_items, num_data):
        """
        Initialization of the dataset.
        Args:
            r, c: Generated data from genData_rd function.
            Q: Upper bound.
            alpha: Alpha parameter for fairness.
        """
        self.x = x
        self.r = r
        self.c = c
        self.Q = Q
        self.alpha = alpha
        self.num_items = num_items
        self.num_data = num_data
        print(f'r shape: {self.r.shape}', f'c shape: {self.c.shape}')

        self.opt_solutions_solver = []
        self.opt_objective_values_solver = []
        self.opt_solutions_closed = []
        self.opt_objective_values_closed = []

        self._solve_optimization_problems()
        self._solve_optimization_problems_closed()

    def _solve_optimization_problems(self):
        """
        Solve the optimization problem for each data point using the solver and store the optimal solutions and values.
        """
        n_data = self.r.shape[0]
        for i in range(n_data):
            model = optModelRd(
                num_items=self.num_items,
                num_data=self.num_data,
                alpha=self.alpha,
                Q=self.Q,
                r=self.r[i],
                c=self.c[i]
            )
            

            d_opt, opt_value = model.solveP()
            self.opt_solutions_solver.append(d_opt)
            self.opt_objective_values_solver.append(opt_value)
        self.opt_solutions_solver = np.array(self.opt_solutions_solver)
        self.opt_objective_values_solver = np.array(self.opt_objective_values_solver)

    def _solve_optimization_problems_closed(self):
        """
        Solve the optimization problem for each data point using the closed-form solution and store the optimal solutions and values.
        """
        n_data = self.r.shape[0]
        for i in range(n_data):
            model = optModelRd(
                num_items=self.num_items,
                num_data=self.num_data,
                alpha=self.alpha,
                Q=self.Q,
                r=self.r[i],
                c=self.c[i]
            )
            d_opt_closed, opt_value_closed = model.solveC()
            self.opt_solutions_closed.append(d_opt_closed)
            self.opt_objective_values_closed.append(opt_value_closed)
        self.opt_solutions_closed = np.array(self.opt_solutions_closed)
        self.opt_objective_values_closed = np.array(self.opt_objective_values_closed)

    def __len__(self):
        return len(self.r)

    def __getitem__(self, idx):
        """
        Get a data point and its corresponding optimal solutions and objective values from both solver and closed form.
        Args:
            idx: Index of the data point.
        Returns:
            Tuple of r, c, optimal solution from solver, optimal objective value from solver, 
            optimal solution from closed form, and optimal objective value from closed form.
        """
        return (
            torch.FloatTensor(self.x[idx]),
            torch.FloatTensor(self.r[idx]),
            torch.FloatTensor(self.c[idx]),
            torch.FloatTensor(self.opt_solutions_solver[idx]),
            torch.FloatTensor([self.opt_objective_values_solver[idx]]),
            torch.FloatTensor(self.opt_solutions_closed[idx]),
            torch.FloatTensor([self.opt_objective_values_closed[idx]])
        )
