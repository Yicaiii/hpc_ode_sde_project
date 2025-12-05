//抽象基类：所有 ODE/SDE 求解器的共同接口：step(x, t, dt)

#pragma once

class SolverBase {
public:
    virtual ~SolverBase() = default;

    // Pure virtual integration step
    virtual void step(double& x, double t, double dt) = 0;

    // Optional: Local error estimator for adaptive solvers
    virtual double estimate_error(double x, double t, double dt) {
        return 0.0;
    }
};

