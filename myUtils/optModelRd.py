import numpy as np
import cvxpy as cp

class optModel_2:
    """
    Abstract optimization model based on cvxpy
    """

    def __init__(self, num_items, num_data, alpha, Q):
        self.n = num_items
        self.m = num_data
        self.alpha = alpha
        self.Q = Q

    def __repr__(self):
        return "OptModel " + self.__class__.__name__

    @property
    def num_items(self):
        """
        Number of items in the optimization problem
        """
        return self.n


class optModelRd(optModel_2):
    """
    Concrete implementation of OptModel for solving specific optimization problems
    """

    def __init__(self, num_items, num_data, alpha, Q, r, c):
        super().__init__(num_items, num_data, alpha, Q)
        self.r = r
        self.c = c
        self.d = cp.Variable((self.m, self.n))
        self.setObj(r, c)

    def solveP(self):
        """
        Solve the optimization problem using cvxpy
        """
        return self.solve()

    def solveC(self):
        """
        Solve the optimization problem using a closed-form solution
        """
        if self.alpha == 1:
            return print("Work in progress")
        else:
            S = np.sum(self.c ** (1 - 1 / self.alpha) * self.r ** (-1 + 1 / self.alpha))
            d_closed_form = (self.c ** (-1 / self.alpha) * self.r ** (-1 + 1 / self.alpha) * self.Q) / S
            optimal_value_closed_form = np.sum((self.r * d_closed_form) ** (1 - self.alpha)) / (1 - self.alpha)
        
        return d_closed_form, optimal_value_closed_form

    def setObj(self, r, c):
        """
        A method to set the objective function and constraints

        Args:
            r, c: Problem parameters
        """
        if self.alpha == 1:
            self.objective = cp.sum(cp.log(cp.multiply(r, self.d)))
        else:
            self.objective = cp.sum(cp.power(cp.multiply(r, self.d), 1 - self.alpha)) / (1 - self.alpha)

        # Updating constraints as per the problem definition
        self.constraints = [
            self.d >= 0,  # 0 ≤ d_i
            cp.sum(cp.multiply(c, self.d)) <= self.Q  # ∑ c_i d_i ≤ Q
        ]

        self.problem = cp.Problem(cp.Maximize(self.objective), self.constraints)

    def solve(self):
        """
        A method to solve the optimization problem

        Returns:
            tuple: optimal solution (list) and objective value (float)
        """
        self.problem.solve()
        d_opt = self.d.value
        optimal_value = self.problem.value

        return d_opt, optimal_value
