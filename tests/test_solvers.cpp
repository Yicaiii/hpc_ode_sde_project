#include <iostream>
#include <cmath>
#include <cassert>

#include "../include/euler_maruyama.hpp"
#include "../include/milstein.hpp"

int main() {
    double lambda = 1.0;
    double sigma = 0.1;
    double dt = 1e-3;
    int steps = 1000;
    double t_final = dt * steps;

    EulerMaruyama em(lambda, sigma);
    Milstein mil(lambda, sigma);

    double x1 = 1.0;
    double x2 = 1.0;

    for (int k = 0; k < steps; k++) {
        em.step(x1, k*dt, dt);
        mil.step(x2, k*dt, dt);
    }

    double exact_mean = std::exp(-lambda * t_final);

    std::cout << "Euler–Maruyama final x = " << x1 << "\n";
    std::cout << "Milstein final x = " << x2 << "\n";
    std::cout << "Exact mean = " << exact_mean << "\n";

    assert(std::abs(x1 - exact_mean) < 0.2);
    assert(std::abs(x2 - exact_mean) < 0.2);

    std::cout << "Unit tests passed.\n";
    return 0;
}
