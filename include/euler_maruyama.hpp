#pragma once
#include "solver_base.hpp"
#include <random>
#include <cmath>

class EulerMaruyama : public SolverBase {
public:
    EulerMaruyama(double lambda, double sigma)
        : lambda(lambda), sigma(sigma) {}

    void step(double& x, double t, double dt) override {
        double dW = normal_dist(rng) * std::sqrt(dt);
        x += -lambda * x * dt + sigma * dW;
    }

private:
    double lambda, sigma;
    thread_local static std::mt19937 rng;
    std::normal_distribution<double> normal_dist{0.0, 1.0};
};

// Definition of thread_local rng
thread_local std::mt19937 EulerMaruyama::rng{1234};
