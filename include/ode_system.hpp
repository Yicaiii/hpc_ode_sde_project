#pragma once
#include <vector>

struct OdeSystem {
    int dim;

    explicit OdeSystem(int d) : dim(d) {}

    virtual void rhs(double t, const double* y, double* dydt) const = 0;

    virtual ~OdeSystem() = default;
};


// ---------------------
// Exponential ODE
// dx/dt = -lambda x
// ---------------------
struct ExponentialDecay : public OdeSystem {
    double lambda;

    explicit ExponentialDecay(double lambda_)
        : OdeSystem(1), lambda(lambda_) {}

    void rhs(double t, const double* y, double* dydt) const override {
        (void)t;
        dydt[0] = -lambda * y[0];
    }

    // ⭐ euler_step 需要
    double drift(double x, double t) const {
        (void)t;
        return -lambda * x;
    }
};


// ---------------------
// OU SDE
// dX = -lambda X dt + sigma dW
// ---------------------
struct OrnsteinUhlenbeckSDE : public OdeSystem {
    double lambda;
    double sigma;

    OrnsteinUhlenbeckSDE(double lambda_, double sigma_)
        : OdeSystem(1), lambda(lambda_), sigma(sigma_) {}

    void rhs(double t, const double* y, double* dydt) const override {
        (void)t;
        dydt[0] = -lambda * y[0];
    }

    // ⭐ euler_maruyama 需要
    double drift(double x, double t) const {
        (void)t;
        return -lambda * x;
    }

    double diffusion(double x, double t) const {
        (void)x; (void)t;
        return sigma;
    }
};
