#pragma once
#include "solver_base.hpp"
#include <random>
#include <cmath>

class Milstein : public SolverBase {
public:
    Milstein(double lambda, double sigma)
        : lambda(lambda), sigma(sigma) {}

    void step(double& x, double t, double dt) override {
        double dW = normal_dist(rng);
        double g = sigma;
        double gprime = 0.0;  // OU g(x)=sigma → constant

        x += -lambda * x * dt
             + g * std::sqrt(dt) * dW
             + 0.5 * g * gprime * (dW*dW - dt);
    }

private:
    double lambda, sigma;
    thread_local static std::mt19937 rng;
    std::normal_distribution<double> normal_dist{0.0, 1.0};
};

thread_local std::mt19937 Milstein::rng{5678};
