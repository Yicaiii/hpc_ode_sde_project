#include <iostream>
#include <chrono>
#include <omp.h>

#include "../include/solver_base.hpp"
#include "../include/euler_maruyama.hpp"
#include "../include/state_soa.hpp"
#include "simulator.cpp"   // 或者直接链接库时删除 include

void print_samples(const StateSOA& state) {
    std::cout << "Example outputs: "
              << state.x[0] << ", "
              << state.x[1] << ", "
              << state.x[2] << "\n";
}

int main() {
    size_t N = 100000;     // 10 万条轨迹
    int steps = 5000;      // 每个轨迹的步数
    double dt = 1e-3;
    double t0 = 0.0;

    EulerMaruyama solver(1.0, 0.3);

    // =========================
    // Baseline (Single Thread)
    // =========================
    StateSOA state_single(N);

    auto t1 = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < N; i++) {
        double xi = 1.0;
        for (int k = 0; k < steps; k++)
            solver.step(xi, t0 + k * dt, dt);
        state_single.x[i] = xi;
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    double baseline_time =
        std::chrono::duration<double>(t2 - t1).count();

    std::cout << "==== Baseline (Single-thread) ====\n";
    print_samples(state_single);
    std::cout << "Time = " << baseline_time << " s\n\n";

    // =========================
    // Parallel Version (OpenMP)
    // =========================
    StateSOA state_parallel(N);

    auto t3 = std::chrono::high_resolution_clock::now();

    simulate(state_parallel, &solver, steps, t0, dt);

    auto t4 = std::chrono::high_resolution_clock::now();
    double parallel_time =
        std::chrono::duration<double>(t4 - t3).count();

    std::cout << "==== OpenMP Parallel ====\n";
    print_samples(state_parallel);
    std::cout << "Time = " << parallel_time << " s\n\n";

    // =========================
    // Speedup
    // =========================
    std::cout << "==== Speedup ====\n";
    std::cout << "Speedup = "
              << baseline_time / parallel_time << "x\n";

    return 0;
}
