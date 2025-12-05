#pragma once
#include "ode_system.hpp"
#include <vector>
#include <random>
#include <cmath>

class EulerIntegrator {
public:
    // 显式 Euler 对 ODE 积分： y_{n+1} = y_n + dt * f(t_n, y_n)
    static void integrate_ode(
        const OdeSystem& system,
        double t0,
        double t_end,
        double dt,
        const std::vector<double>& y0,
        std::vector<double>& y_out
    ) {
        int dim = system.dim;
        int n_steps = static_cast<int>((t_end - t0) / dt);

        y_out = y0;
        std::vector<double> dydt(dim);

        double t = t0;
        for (int step = 0; step < n_steps; ++step) {
            system.rhs(t, y_out.data(), dydt.data());
            for (int i = 0; i < dim; ++i) {
                y_out[i] += dt * dydt[i];
            }
            t += dt;
        }
    }

    // Euler-Maruyama 积分 SDE: dX = f dt + sigma dW
    // 为简单起见，假设是 1 维、扩散系数 sigma 为常数
    static void integrate_sde_1d_euler_maruyama(
        const OrnsteinUhlenbeckSDE& system,
        double t0,
        double t_end,
        double dt,
        double x0,
        double& x_out,
        unsigned int seed = 42
    ) {
        int n_steps = static_cast<int>((t_end - t0) / dt);

        std::mt19937 rng(seed);
        std::normal_distribution<double> normal(0.0, 1.0);

        double x = x0;
        double t = t0;
        double drift[1];

        for (int step = 0; step < n_steps; ++step) {
            system.rhs(t, &x, drift);      // drift = f(t, x)
            double dW = std::sqrt(dt) * normal(rng); // Brownian increment
            x += dt * drift[0] + system.sigma * dW;  // Euler-Maruyama
            t += dt;
        }

        x_out = x;
    }
};

// ----------------------------------------------
// Single-thread Euler integrator for ODE
// ----------------------------------------------
template<typename System>
double euler_step(System& sys, double x0,
                  double t0, double t_end, double dt)
{
    int steps = static_cast<int>((t_end - t0) / dt);
    double x = x0;

    for (int k = 0; k < steps; k++)
    {
        x += dt * sys.drift(x, t0 + k * dt);
    }
    return x;
}


// ----------------------------------------------
// Single-thread Euler–Maruyama for SDE
// (not used yet, but kept for completeness)
// ----------------------------------------------
template<typename System>
double euler_maruyama_step(System& sys, double x0,
                           double t0, double t_end, double dt)
{
    int steps = static_cast<int>((t_end - t0) / dt);
    double x = x0;

    std::mt19937 rng(1234);
    std::normal_distribution<double> norm(0.0, 1.0);

    for (int k = 0; k < steps; k++)
    {
        double drift = sys.drift(x, t0 + k * dt);
        double diff  = sys.diffusion(x, t0 + k * dt);
        double dW    = norm(rng) * std::sqrt(dt);

        x += drift * dt + diff * dW;
    }

    return x;
}
