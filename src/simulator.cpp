#include <omp.h>
#include <vector>
#include <cmath>
#include <iostream>

#include "../include/solver_base.hpp"
#include "../include/state_soa.hpp"

void simulate(StateSOA& state, SolverBase* solver,
              int steps, double t0, double dt)
{
    size_t N = state.size();

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < N; i++) {
        double xi = state[i];

        #pragma omp simd
        for (int k = 0; k < steps; k++) {
            solver->step(xi, t0 + k * dt, dt);
        }

        state[i] = xi;
    }
}
