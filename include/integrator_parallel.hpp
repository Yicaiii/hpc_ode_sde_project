#pragma message(">>> Using integrator_parallel.hpp from THIS PATH")
#pragma once
#include <vector>
#include <cmath>
#include <random>
#include <omp.h>

// ------------------------------------------
// Parallel Euler integrator for ODE
// ------------------------------------------
template<typename System>
void euler_parallel(
    System& sys,
    double x0,
    double t0,
    double t_end,
    double dt,
    int N,
    std::vector<double>& results)
{
    int steps = static_cast<int>((t_end - t0) / dt);

    results.resize(N);

    #pragma omp parallel for
    for (int i = 0; i < N; i++)
    {
        double x = x0;
        for (int k = 0; k < steps; k++)
            x += dt * sys.drift(x, t0 + k * dt);

        results[i] = x;
    }
}


// ---------------------------------------------------
// Parallel Euler–Maruyama integrator for SDE
// ---------------------------------------------------
template<typename System>
void euler_maruyama_parallel(
    System& sys,
    double x0,
    double t0,
    double t_end,
    double dt,
    int N,
    std::vector<double>& results)
{
    int steps = static_cast<int>((t_end - t0) / dt);

    results.resize(N);

    #pragma omp parallel
    {
        std::mt19937 rng(1234 + omp_get_thread_num());
        std::normal_distribution<double> norm(0.0, 1.0);

        #pragma omp for
        for (int i = 0; i < N; i++)
        {
            double x = x0;

            for (int k = 0; k < steps; k++)
            {
                double drift = sys.drift(x, t0 + k * dt);
                double diff = sys.diffusion(x, t0 + k * dt);

                double dW = norm(rng) * std::sqrt(dt);

                x += drift * dt + diff * dW;
            }

            results[i] = x;
        }
    }
}
