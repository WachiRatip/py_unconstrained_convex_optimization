import numpy as np

class ConvexSolver:
    """
    Class for solving the optimisation problem
    using the gradient descent algorithm
    when f(x) is a convex function and the gradient of f(x) is Lipschitz continuous
    the gradient descent algorithm is guaranteed to converge to the global minimum
    of f(x) when the learning rate is sufficiently small
    """
    def __init__(self, f, grad_f, x0, lr, tol, max_iter):
        """
        Constructor for the ConvexSolver class

        Parameters
        ----------
        f : function
            The function to be minimised
        grad_f : function
            The gradient of the function to be minimised
        x0 : numpy.ndarray
            The initial guess
        lr : float
            The learning rate
        tol : float
            The tolerance
        max_iter : int
            The maximum number of iterations
        """
        self.f = f
        self.grad_f = grad_f
        self.x0 = x0
        self.lr = lr
        self.tol = tol
        self.max_iter = max_iter
        self.x = x0
        self.f_x = f(x0)
        self.grad_f_x = grad_f(x0)
        self.iter = 0
        self.converged = False
        self.converged_reason = None
        self.converged_reasons = {
            "max_iter": "Maximum number of iterations reached",
            "tol": "Tolerance reached"
        }
        self.sequnce_of_x = [self.x]
    
    def solve(self):
        """
        Solve the optimisation problem
        Note: the implementation of gradient descent
                x = x - lr*grad_f(x)
        """
        while not self.converged:
            self.iter += 1
            self.x = self.x - self.lr*self.grad_f_x
            self.f_x = self.f(self.x)
            self.grad_f_x = self.grad_f(self.x)
            self.sequnce_of_x.append(self.x)
            if self.iter >= self.max_iter:
                self.converged = True
                self.converged_reason = self.converged_reasons["max_iter"]
            elif np.linalg.norm(self.grad_f_x) <= self.tol:
                self.converged = True
                self.converged_reason = self.converged_reasons["tol"]


class ConvexSolverWithMomentum:
    """
    Class for solving the optimisation problem
    using the gradient descent algorithm with momentum
    when f(x) is a convex function and the gradient of f(x) is Lipschitz continuous
    the gradient descent algorithm is guaranteed to converge to the global minimum
    of f(x) when the learning rate is sufficiently small
    """
    def __init__(self, f, grad_f, x0, lr, tol, max_iter, beta):
        """
        Constructor for the ConvexSolverWithMomentum class

        Parameters
        ----------
        f : function
            The function to be minimised
        grad_f : function
            The gradient of the function to be minimised
        x0 : numpy.ndarray
            The initial guess
        lr : float
            The learning rate
        tol : float
            The tolerance
        max_iter : int
            The maximum number of iterations
        beta : float
            The momentum parameter
        """
        self.f = f
        self.grad_f = grad_f
        self.x0 = x0
        self.lr = lr
        self.tol = tol
        self.max_iter = max_iter
        self.beta = beta
        self.x = x0
        self.f_x = f(x0)
        self.grad_f_x = grad_f(x0)
        self.iter = 0
        self.converged = False
        self.converged_reason = None
        self.converged_reasons = {
            "max_iter": "Maximum number of iterations reached",
            "tol": "Tolerance reached"
        }
        self.v = np.zeros_like(self.x)
        self.sequnce_of_x = [self.x]

    def solve(self):
        """
        Solve the optimisation problem
        Note: the implementation of gradient descent with momentum
                v = beta*v + lr*grad_f(x)
                x = x - v
        """
        while not self.converged:
            self.iter += 1
            self.v = self.beta*self.v + self.lr*self.grad_f_x
            self.x = self.x - self.v
            self.f_x = self.f(self.x)
            self.grad_f_x = self.grad_f(self.x)
            self.sequnce_of_x.append(self.x)
            if self.iter >= self.max_iter:
                self.converged = True
                self.converged_reason = self.converged_reasons["max_iter"]
            elif np.linalg.norm(self.grad_f_x) <= self.tol:
                self.converged = True
                self.converged_reason = self.converged_reasons["tol"]


class ConvexSolverWithNesterovMomentum:
    """
    Class for solving the optimisation problem
    using the gradient descent algorithm with Nesterov's accelerated gradient descent
    when f(x) is a convex function and the gradient of f(x) is Lipschitz continuous
    the gradient descent algorithm is guaranteed to converge to the global minimum
    of f(x) when the learning rate is sufficiently small
    """
    def __init__(self, f, grad_f, x0, lr, tol, max_iter, beta):
        """
        Constructor for the ConvexSolverWithNesterovMomentum class

        Parameters
        ----------
        f : function
            The function to be minimised
        grad_f : function
            The gradient of the function to be minimised
        x0 : numpy.ndarray
            The initial guess
        lr : float
            The learning rate
        tol : float
            The tolerance
        max_iter : int
            The maximum number of iterations
        beta : float
            The momentum parameter
        """
        self.f = f
        self.grad_f = grad_f
        self.x0 = x0
        self.lr = lr
        self.tol = tol
        self.max_iter = max_iter
        self.beta = beta
        self.x = x0
        self.f_x = f(x0)
        self.grad_f_x = grad_f(x0)
        self.iter = 0
        self.converged = False
        self.converged_reason = None
        self.converged_reasons = {
            "max_iter": "Maximum number of iterations reached",
            "tol": "Tolerance reached"
        }
        self.v = np.zeros_like(self.x)
        self.sequnce_of_x = [self.x]

    def solve(self):
        """
        Solve the optimisation problem
        Note: the implementation of Nesterov's accelerated gradient descent
                v = beta*v + lr*grad_f(x - beta*v)
                x = x - v
        this accelerated gradient descent algorithm is guaranteed to converge to the global minimum
        of f(x) when the learning rate is sufficiently small
        when beta = 0, this algorithm is equivalent to gradient descent
        """
        while not self.converged:
            self.iter += 1
            self.v = self.beta*self.v + self.lr*self.grad_f(self.x - self.beta*self.v)
            self.x = self.x - self.v
            self.f_x = self.f(self.x)
            self.grad_f_x = self.grad_f(self.x)
            self.sequnce_of_x.append(self.x)
            if self.iter >= self.max_iter:
                self.converged = True
                self.converged_reason = self.converged_reasons["max_iter"]
            elif np.linalg.norm(self.grad_f_x) <= self.tol:
                self.converged = True
                self.converged_reason = self.converged_reasons["tol"]
