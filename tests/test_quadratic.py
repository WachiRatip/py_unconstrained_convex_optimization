import numpy as np
from numpy.testing import assert_almost_equal, assert_array_almost_equal

from optimisations.optimisation_algorithms import ConvexSolver, ConvexSolverWithMomentum, ConvexSolverWithNesterovMomentum

# Test the ConvexSolver class
def test_convex_solver():
    # f(x) = x^2 + 2x + 1
    # min(x) = -1
    def f(x):
        return x**2 + 2*x + 1
    def grad_f(x):
        return 2*x + 2
    x0 = 0
    lr = 0.1
    tol = 1e-9
    max_iter = 100
    convex_solver = ConvexSolver(f, grad_f, x0, lr, tol, max_iter)
    convex_solver.solve()
    assert_almost_equal(convex_solver.x, -1, decimal=6)
    assert_almost_equal(convex_solver.f_x, 0, decimal=6)
    assert_almost_equal(convex_solver.grad_f_x, 0, decimal=6)

# Test the ConvexSolverWithMomentum class
def test_convex_solver_with_momentum():
    # f(x) = x^2 + 2x + 1
    # min(x) = -1
    def f(x):
        return x**2 + 2*x + 1
    def grad_f(x):
        return 2*x + 2
    x0 = 0
    lr = 0.1
    tol = 1e-9
    max_iter = 100
    beta = 0.09
    convex_solver_with_momentum = ConvexSolverWithMomentum(f, grad_f, x0, lr, tol, max_iter, beta)
    convex_solver_with_momentum.solve()
    assert_almost_equal(convex_solver_with_momentum.x, -1, decimal=6)
    assert_almost_equal(convex_solver_with_momentum.f_x, 0, decimal=6)
    assert_almost_equal(convex_solver_with_momentum.grad_f_x, 0, decimal=6)

# Test the ConvexSolverWithNesterovMomentum class
def test_convex_solver_with_nesterov_momentum():
    # f(x) = x^2 + 2x + 1
    # min(x) = -1
    def f(x):
        return x**2 + 2*x + 1
    def grad_f(x):
        return 2*x + 2
    x0 = 0
    lr = 0.1
    tol = 1e-9
    max_iter = 100
    beta = 0.09
    convex_solver_with_nesterov_momentum = ConvexSolverWithNesterovMomentum(f, grad_f, x0, lr, tol, max_iter, beta)
    convex_solver_with_nesterov_momentum.solve()
    assert_almost_equal(convex_solver_with_nesterov_momentum.x, -1, decimal=6)
    assert_almost_equal(convex_solver_with_nesterov_momentum.f_x, 0, decimal=6)
    assert_almost_equal(convex_solver_with_nesterov_momentum.grad_f_x, 0, decimal=6)

# Test the ConvexSolver class with a 2D function
def test_convex_solver_2d():
    # f(x) = x^2 + 2x + 1
    # min(x) = -1
    # x_min = [-1, -1]
    def f(x):
        return x[0]**2 + x[1]**2 + 2*x[0] + 2*x[1] + 1
    def grad_f(x):
        return np.array([2*x[0] + 2, 2*x[1] + 2])
    x0 = np.array([0, 0])
    lr = 0.1
    tol = 1e-9
    max_iter = 100
    convex_solver = ConvexSolver(f, grad_f, x0, lr, tol, max_iter)
    convex_solver.solve()
    assert_array_almost_equal(convex_solver.x, np.array([-1, -1]), decimal=6)
    assert_almost_equal(convex_solver.f_x, -1, decimal=6)
    assert_array_almost_equal(convex_solver.grad_f_x, np.array([0, 0]), decimal=6)

# Test the ConvexSolverWithMomentum class with a 2D function
def test_convex_solver_with_momentum_2d():
    # f(x) = x^2 + 2x + 1
    # min(x) = -1
    # x_min = [-1, -1]
    def f(x):
        return x[0]**2 + x[1]**2 + 2*x[0] + 2*x[1] + 1
    def grad_f(x):
        return np.array([2*x[0] + 2, 2*x[1] + 2])
    x0 = np.array([0, 0])
    lr = 0.1
    tol = 1e-9
    max_iter = 100
    beta = 0.09
    convex_solver_with_momentum = ConvexSolverWithMomentum(f, grad_f, x0, lr, tol, max_iter, beta)
    convex_solver_with_momentum.solve()
    assert_array_almost_equal(convex_solver_with_momentum.x, np.array([-1, -1]), decimal=6)
    assert_almost_equal(convex_solver_with_momentum.f_x, -1, decimal=6)
    assert_array_almost_equal(convex_solver_with_momentum.grad_f_x, np.array([0, 0]), decimal=6)

# Test the ConvexSolverWithNesterovMomentum class with a 2D function
def test_convex_solver_with_nesterov_momentum_2d():
    # f(x) = x^2 + 2x + 1
    # min(x) = -1
    # x_min = [-1, -1]
    def f(x):
        return x[0]**2 + x[1]**2 + 2*x[0] + 2*x[1] + 1
    def grad_f(x):
        return np.array([2*x[0] + 2, 2*x[1] + 2])
    x0 = np.array([0, 0])
    lr = 0.1
    tol = 1e-9
    max_iter = 100
    beta = 0.09
    convex_solver_with_nesterov_momentum = ConvexSolverWithNesterovMomentum(f, grad_f, x0, lr, tol, max_iter, beta)
    convex_solver_with_nesterov_momentum.solve()
    assert_array_almost_equal(convex_solver_with_nesterov_momentum.x, np.array([-1, -1]), decimal=6)
    assert_almost_equal(convex_solver_with_nesterov_momentum.f_x, -1, decimal=6)
    assert_array_almost_equal(convex_solver_with_nesterov_momentum.grad_f_x, np.array([0, 0]), decimal=6)