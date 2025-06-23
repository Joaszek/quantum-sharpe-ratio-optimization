import numpy as np
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import Sampler

def optimize_sharpe(log_returns_window, k=2, lambda_risk=1.0):
    mu = log_returns_window.mean().values
    sigma = log_returns_window.cov().values
    n = len(mu)

    qp = QuadraticProgram()
    for i in range(n):
        qp.binary_var(name=f"x{i}")

    linear = {f"x{i}": -mu[i] for i in range(n)}
    quadratic = {}

    for i in range(n):
        for j in range(n):
            if sigma[i][j] != 0.0:
                key = (f"x{i}", f"x{j}")
                coeff = lambda_risk * sigma[i][j]
                if key in quadratic:
                    quadratic[key] += coeff
                else:
                    quadratic[key] = coeff

    qp.minimize(linear=linear, quadratic=quadratic)

    qp.linear_constraint(
        linear={f"x{i}": 1 for i in range(n)},
        sense="==",
        rhs=k,
        name="select_k_assets"
    )

    sampler = Sampler()
    optimizer = COBYLA()
    qaoa = QAOA(sampler=sampler, optimizer=optimizer)
    meo = MinimumEigenOptimizer(qaoa)
    result = meo.solve(qp)

    return result.x, result.fval
