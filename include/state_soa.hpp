#pragma once
#include <vector>

struct StateSOA {
    std::vector<double> x;

    StateSOA(size_t N) : x(N) {}

    inline size_t size() const { return x.size(); }
    inline double& operator[](size_t i) { return x[i]; }
};
