import numpy as np
import cvxpy as cp

class optModel_1:
    """
    Abstract optimization model based on cvxpy
    """

    def __init__(self, n, alpha, Q,ep):
        self.n = n
        self.alpha = alpha
        self.Q = Q
        self.epsilon = ep
        self.u = cp.Variable(n)
        self.constraints = []
        self.objective = None
        self.problem = None

    def __repr__(self):
        return "optModel " + self.__class__.__name__

    @property
    def num_items(self):
        """
        Number of items in the optimization problem
        """
        return self.n

    def setObj(self, a, r, b, c):
        """
        A method to set the objective function and constraints

        Args:
            a, r, b, c: Problem parameters
        """
        if self.alpha == 1:
            self.objective = cp.sum(cp.log(self.u))
        else:
            self.objective = cp.sum(cp.power(self.u, 1 - self.alpha)) / (1 - self.alpha)
        
        self.constraints = [
            self.u >= a * r + self.epsilon,
            cp.sum(c / b * self.u) <= self.Q + np.sum(c * (a * r + self.epsilon) / b)
        ]

        self.problem = cp.Problem(cp.Maximize(self.objective), self.constraints)

    def solve(self):
        """
        A method to solve the optimization problem

        Returns:
            tuple: optimal solution (list) and objective value (float)
        """
        self.problem.solve(warm_start=True, verbose=False)
        u_opt = self.u.value
        optimal_value = self.objective.value
        d_opt = (u_opt - self.a * self.r - self.epsilon) / self.b
        return u_opt, d_opt, optimal_value

    def addConstr(self, new_constraint):
        """
        A method to add a new constraint to the optimization problem

        Args:
            new_constraint: The new constraint to add
        """
        self.constraints.append(new_constraint)
        self.problem = cp.Problem(cp.Maximize(self.objective), self.constraints)

class optModel_2:
    """
    Abstract optimization model based on cvxpy
    """

    def __init__(self, num_items, num_data, alpha, Q):
        self.n = num_items
        self.m = num_data
        self.alpha = alpha
        self.Q = Q
        self.d = cp.Variable(self.n)
        self.constraints = []
        self.objective = None
        self.problem = None

    def __repr__(self):
        return "OptModel " + self.__class__.__name__

    @property
    def num_items(self):
        """
        Number of items in the optimization problem
        """
        return self.n

    def setObj(self, r, c):
        """
        A method to set the objective function and constraints

        Args:
            r, c: Problem parameters
        """
        if self.alpha == 1:
            self.objective = cp.sum(cp.log(r * self.d))
        else:
            self.objective = cp.sum(cp.power(r * self.d, 1 - self.alpha)) / (1 - self.alpha)
        
        self.constraints = [
            self.d >= 0,
            cp.sum(cp.multiply(c, self.d)) <= self.Q
        ]

        self.problem = cp.Problem(cp.Maximize(self.objective), self.constraints)

    def solve(self):
        """
        A method to solve the optimization problem

        Returns:
            tuple: optimal solution (list) and objective value (float)
        """
        self.problem.solve(warm_start=True, verbose=False)
        d_opt = self.d.value
        optimal_value = self.objective.value
        return d_opt, optimal_value


class optModelAr(optModel_1):
    """
    Concrete implementation of optModel for solving specific optimization problems
    """

    def __init__(self, n, alpha, Q, epsilon, a, r, b, c):
        super().__init__(n, alpha, Q, epsilon)
        self.a = a
        self.r = r
        self.b = b
        self.c = c
        self.setObj(a, r, b, c)

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
            Q_term = self.Q + np.sum(self.c * (self.a * self.r + self.epsilon) / self.b)
            u_closed_form = (self.b / self.c) * (Q_term / self.n)
            optimal_value_closed_form = np.sum(np.log(u_closed_form))
        else:
            S = np.sum((self.c / self.b) ** (1 - 1 / self.alpha))
            Q_term = self.Q + np.sum(self.c * (self.a * self.r + self.epsilon) / self.b)
            mu = (S / Q_term) ** self.alpha * (1 - self.alpha)
            u_closed_form = (self.c / self.b) ** (-1 / self.alpha) * Q_term / S
            optimal_value_closed_form = np.sum(np.log(u_closed_form)) if self.alpha == 1 else np.sum(np.power(u_closed_form, 1 - self.alpha)) / (1 - self.alpha)

        d_closed_form = (u_closed_form - self.a * self.r - self.epsilon) / self.b

        return u_closed_form, d_closed_form, optimal_value_closed_form

class optModelRd(optModel_2):
    """
    Concrete implementation of OptModel for solving specific optimization problems
    """

    def __init__(self, num_items, num_data, alpha, Q, r, c):
        super().__init__(num_items, num_data, alpha, Q)
        self.r = r
        self.c = c
        self.n = num_items
        self.m = num_data
        self.d = cp.Variable((self.m, self.n))
        self.constraints = []
        self.objective = None
        self.problem = None
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
            optimal_value_closed_form = np.sum((self.r * d_closed_form)**(1 - self.alpha)) / (1 - self.alpha)
        
        return d_closed_form, optimal_value_closed_form

    def setObj(self, r, c):
        """
        A method to set the objective function and constraints

        Args:
            r, c: Problem parameters
        """
        if self.alpha == 1:
            self.objective = cp.sum(cp.log(r * self.d))
        else:
            self.objective = cp.sum(cp.power(r * self.d, 1 - self.alpha)) / (1 - self.alpha)
        
        self.constraints = [
            self.d >= 0,
            cp.sum(cp.multiply(c, self.d)) <= self.Q
        ]

        self.problem = cp.Problem(cp.Maximize(self.objective), self.constraints)

    def solve(self):
        """
        A method to solve the optimization problem

        Returns:
            tuple: optimal solution (list) and objective value (float)
        """
        self.problem.solve(warm_start=True, verbose=False)
        d_opt = self.d.value
        optimal_value = self.objective.value
        return d_opt, optimal_value