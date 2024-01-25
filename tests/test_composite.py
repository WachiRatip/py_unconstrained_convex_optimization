import numpy as np
from numpy.testing import assert_almost_equal, assert_array_almost_equal

from optimisations.optimisation import ConvexSolver, ConvexSolverWithMomentum, ConvexSolverWithNesterovMomentum

# test for composition of linear and quadratic functions
def test_convex_solver():
    # f(g(x)) = (x^2 + 2x + 1)^2 + 2(x^2 + 2x + 1) + 1
    # min(x) = -1
    def f(x):
        return x**2 + 2*x + 1
    def grad_f(x):
        return 2*x + 2
    def g(x):
        return f(x)**2 + 2*f(x) + 1
    def grad_g(x):
        return 2*f(x)*grad_f(x) + 2*grad_f(x)
    x0 = 0
    lr = 0.1
    tol = 1e-9
    max_iter = 100
    convex_solver = ConvexSolver(g, grad_g, x0, lr, tol, max_iter)
    convex_solver.solve()
    assert_almost_equal(convex_solver.x, -1, decimal=6)
    assert_almost_equal(convex_solver.f_x, 1, decimal=6)
    assert_almost_equal(convex_solver.grad_f_x, 0, decimal=6)

# test for composition of linear and quadratic functions with momentum
def test_convex_solver_with_momentum():
    # f(g(x)) = (x^2 + 2x + 1)^2 + 2(x^2 + 2x + 1) + 1
    # min(x) = -1
    def f(x):
        return x**2 + 2*x + 1
    def grad_f(x):
        return 2*x + 2
    def g(x):
        return f(x)**2 + 2*f(x) + 1
    def grad_g(x):
        return 2*f(x)*grad_f(x) + 2*grad_f(x)
    x0 = 0
    lr = 0.1
    tol = 1e-9
    max_iter = 100
    beta = 0.09
    convex_solver_with_momentum = ConvexSolverWithMomentum(g, grad_g, x0, lr, tol, max_iter, beta)
    convex_solver_with_momentum.solve()
    assert_almost_equal(convex_solver_with_momentum.x, -1, decimal=6)
    assert_almost_equal(convex_solver_with_momentum.f_x, 1, decimal=6)
    assert_almost_equal(convex_solver_with_momentum.grad_f_x, 0, decimal=6)

# test for composition of linear and quadratic functions with nesterov momentum
def test_convex_solver_with_nesterov_momentum():
    # f(g(x)) = (x^2 + 2x + 1)^2 + 2(x^2 + 2x + 1) + 1
    # min(x) = -1
    def f(x):
        return x**2 + 2*x + 1
    def grad_f(x):
        return 2*x + 2
    def g(x):
        return f(x)**2 + 2*f(x) + 1
    def grad_g(x):
        return 2*f(x)*grad_f(x) + 2*grad_f(x)
    x0 = 0
    lr = 0.1
    tol = 1e-9
    max_iter = 100
    beta = 0.09
    convex_solver_with_nesterov_momentum = ConvexSolverWithNesterovMomentum(g, grad_g, x0, lr, tol, max_iter, beta)
    convex_solver_with_nesterov_momentum.solve()
    assert_almost_equal(convex_solver_with_nesterov_momentum.x, -1, decimal=6)
    assert_almost_equal(convex_solver_with_nesterov_momentum.f_x, 1, decimal=6)
    assert_almost_equal(convex_solver_with_nesterov_momentum.grad_f_x, 0, decimal=6)

def test_convex_solver_2d():
    # f(g(x)) = (x^2 + 2x + 1)^2 + 2(x^2 + 2x + 1) + 1
    # min(x) = 0
    # x_min = [-1, -1]
    def f(x):
        return x[0]**2 + x[1]**2 + 2*x[0] + 2*x[1] + 1
    def grad_f(x):
        return np.array([2*x[0] + 2, 2*x[1] + 2])
    def g(x):
        return f(x)**2 + 2*f(x) + 1
    def grad_g(x):
        return 2*f(x)*grad_f(x) + 2*grad_f(x)
    
    x0 = np.array([0, 0])
    lr = 0.125
    tol = 1e-9
    max_iter = 100
    convex_solver = ConvexSolver(g, grad_g, x0, lr, tol, max_iter)
    convex_solver.solve()
    assert_array_almost_equal(convex_solver.x, np.array([-1, -1]), decimal=6)
    assert_almost_equal(convex_solver.f_x, 0, decimal=6)
    assert_array_almost_equal(convex_solver.grad_f_x, np.array([0, 0]), decimal=6)

def test_convex_solver_with_momentum_2d():
    # f(g(x)) = (x^2 + 2x + 1)^2 + 2(x^2 + 2x + 1) + 1
    # min(x) = 0
    # x_min = [-1, -1]
    def f(x):
        return x[0]**2 + x[1]**2 + 2*x[0] + 2*x[1] + 1
    def grad_f(x):
        return np.array([2*x[0] + 2, 2*x[1] + 2])
    def g(x):
        return f(x)**2 + 2*f(x) + 1
    def grad_g(x):
        return 2*f(x)*grad_f(x) + 2*grad_f(x)
    
    x0 = np.array([0, 0])
    lr = 0.125
    tol = 1e-9
    max_iter = 100
    beta = 0.09
    convex_solver_with_momentum = ConvexSolverWithMomentum(g, grad_g, x0, lr, tol, max_iter, beta)
    convex_solver_with_momentum.solve()
    assert_array_almost_equal(convex_solver_with_momentum.x, np.array([-1, -1]), decimal=6)
    assert_almost_equal(convex_solver_with_momentum.f_x, 0, decimal=6)
    assert_array_almost_equal(convex_solver_with_momentum.grad_f_x, np.array([0, 0]), decimal=6)