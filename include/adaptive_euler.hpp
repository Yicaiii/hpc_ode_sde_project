#pragma once
#include "solver_base.hpp"
#include <cmath>

class ODEAdaptiveEuler : public SolverBase {
public:
    ODEAdaptiveEuler(double lambda, double tol = 1e-4)
        : lambda(lambda), tol(tol) {}

    void step(double& x, double t, double dt) override {
        // One large step
        double x1 = x + (-lambda * x) * dt;

        // Two half steps
        double x_half = x + (-lambda * x) * (0.5 * dt);
        double x2 = x_half + (-lambda * x_half) * (0.5 * dt);

        double error = std::abs(x2 - x1);

        // Adaptive control
        if (error > tol) {
            dt *= 0.5;
        } else if (error < tol / 10) {
            dt *= 1.5;
        }

        x = x2;
    }

    double estimate_error(double x, double t, double dt) override {
        double x1 = x + (-lambda * x) * dt;
        double x_half = x + (-lambda * x) * (0.5 * dt);
        double x2 = x_half + (-lambda * x_half) * (0.5 * dt);

        return std::abs(x2 - x1);
    }

private:
    double lambda, tol;
};
