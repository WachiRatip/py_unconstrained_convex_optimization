import numpy as np
from optimisations.optimisation_algorithms import ConvexSolver, ConvexSolverWithMomentum, ConvexSolverWithNesterovMomentum

def main():
    x0 = 0
    lr = 0.05
    tol = 1e-12
    max_iter = 500

    # Test the ConvexSolver class
    # f(x) = x^2 + 2x + 1
    # min(x) = -1
    def f(x):
        return x**2 + 2*x + 1
    def grad_f(x):
        return 2*x + 2

    convex_solver = ConvexSolver(f, grad_f, x0, lr, tol, max_iter)
    convex_solver.solve()
    print("min(x) = %.4f" %(convex_solver.x))
    print("f(x) = %.4f" %(convex_solver.f_x))
    print("grad(f(x)) = %.4f" %(convex_solver.grad_f_x))
    print("iterations = %d" %(convex_solver.iter))
    print("converged = %s" %(convex_solver.converged))
    print("converged reason = %s" %(convex_solver.converged_reason))
    print("sequence of x:")
    print(convex_solver.sequnce_of_x)
    print("")

    # Test the ConvexSolverWithMomentum class
    # f(x) = x^2 + 2x + 1
    # min(x) = -1
    def f(x):
        return x**2 + 2*x + 1
    def grad_f(x):
        return 2*x + 2
    
    beta = 0.05
    convex_solver_with_momentum = ConvexSolverWithMomentum(f, grad_f, x0, lr, tol, max_iter, beta)
    convex_solver_with_momentum.solve()
    print("min(x) = %.4f" %(convex_solver_with_momentum.x))
    print("f(x) = %.4f" %(convex_solver_with_momentum.f_x))
    print("grad(f(x)) = %.4f" %(convex_solver_with_momentum.grad_f_x))
    print("iterations = %d" %(convex_solver_with_momentum.iter))
    print("converged = %s" %(convex_solver_with_momentum.converged))
    print("converged reason = %s" %(convex_solver_with_momentum.converged_reason))
    print("sequence of x:")
    print(convex_solver_with_momentum.sequnce_of_x)
    print("")

    # Test the ConvexSolverWithNesterovMomentum class
    # f(x) = x^2 + 2x + 1
    # min(x) = -1
    def f(x):
        return x**2 + 2*x + 1
    def grad_f(x):
        return 2*x + 2

    beta = 0.09
    convex_solver_with_nesterov_momentum = ConvexSolverWithNesterovMomentum(f, grad_f, x0, lr, tol, max_iter, beta)
    convex_solver_with_nesterov_momentum.solve()
    print("min(x) = %.4f" %(convex_solver_with_nesterov_momentum.x))
    print("f(x) = %.4f" %(convex_solver_with_nesterov_momentum.f_x))
    print("grad(f(x)) = %.4f" %(convex_solver_with_nesterov_momentum.grad_f_x))
    print("iterations = %d" %(convex_solver_with_nesterov_momentum.iter))
    print("converged = %s" %(convex_solver_with_nesterov_momentum.converged))
    print("converged reason = %s" %(convex_solver_with_nesterov_momentum.converged_reason))
    print("sequence of x:")
    print(convex_solver_with_nesterov_momentum.sequnce_of_x)
    print("")

def main_2d():
    x0 = np.array([0, 0])
    lr = 0.125
    tol = 1e-12
    max_iter = 150
    beta = 0.099

    # Test the ConvexSolver class with a 2D function
    # f(x) = x^2 + 2x + 1
    # min(x) = -1
    def f(x):
        return x[0]**2 + x[1]**2 + 2*x[0] + 2*x[1] + 1
    def grad_f(x):
        return np.array([2*x[0] + 2, 2*x[1] + 2])

    convex_solver = ConvexSolver(f, grad_f, x0, lr, tol, max_iter)
    convex_solver.solve()
    print("min(x) = %.4f" %(convex_solver.x[0]))
    print("f(x) = %.4f" %(convex_solver.f_x))
    print("grad(f(x)) = (%.4f, %.4f)" %(convex_solver.grad_f_x[0], convex_solver.grad_f_x[1]))
    print("iterations = %d" %(convex_solver.iter))
    print("converged = %s" %(convex_solver.converged))
    print("converged reason = %s" %(convex_solver.converged_reason))
    print("sequence of x:")
    print(convex_solver.sequnce_of_x)
    print("")

    # Test the ConvexSolverWithMomentum class with a 2D function
    # f(x) = x^2 + 2x + 1
    # min(x) = -1
    def f(x):
        return x[0]**2 + x[1]**2 + 2*x[0] + 2*x[1] + 1
    def grad_f(x):
        return np.array([2*x[0] + 2, 2*x[1] + 2])

    convex_solver_with_momentum = ConvexSolverWithMomentum(f, grad_f, x0, lr, tol, max_iter, beta)
    convex_solver_with_momentum.solve()
    print("min(x) = %.4f" %(convex_solver_with_momentum.x[0]))
    print("f(x) = %.4f" %(convex_solver_with_momentum.f_x))
    print("grad(f(x)) = (%.4f, %.4f)" %(convex_solver_with_momentum.grad_f_x[0], convex_solver_with_momentum.grad_f_x[1]))
    print("iterations = %d" %(convex_solver_with_momentum.iter))
    print("converged = %s" %(convex_solver_with_momentum.converged))
    print("converged reason = %s" %(convex_solver_with_momentum.converged_reason))
    print("sequence of x:")
    print(convex_solver_with_momentum.sequnce_of_x)
    print("")

    # Test the ConvexSolverWithNesterovMomentum class with a 2D function
    # f(x) = x^2 + 2x + 1
    # min(x) = -1
    def f(x):
        return x[0]**2 + x[1]**2 + 2*x[0] + 2*x[1] + 1
    def grad_f(x):
        return np.array([2*x[0] + 2, 2*x[1] + 2])
    
    convex_solver_with_nesterov_momentum = ConvexSolverWithNesterovMomentum(f, grad_f, x0, lr, tol, max_iter, beta)
    convex_solver_with_nesterov_momentum.solve()
    print("min(x) = %.4f" %(convex_solver_with_nesterov_momentum.x[0]))
    print("f(x) = %.4f" %(convex_solver_with_nesterov_momentum.f_x))
    print("grad(f(x)) = (%.4f, %.4f)" %(convex_solver_with_nesterov_momentum.grad_f_x[0], convex_solver_with_nesterov_momentum.grad_f_x[1]))
    print("iterations = %d" %(convex_solver_with_nesterov_momentum.iter))
    print("converged = %s" %(convex_solver_with_nesterov_momentum.converged))
    print("converged reason = %s" %(convex_solver_with_nesterov_momentum.converged_reason))
    print("sequence of x:")
    print(convex_solver_with_nesterov_momentum.sequnce_of_x)
    print("")


if __name__ == "__main__":
    main()
    #main_2d()