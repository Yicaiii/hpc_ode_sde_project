cmake_minimum_required(VERSION 3.10)
project(HPC_ODE_SDE LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(hpc_ode_sde
    src/main.cpp
)

target_include_directories(hpc_ode_sde PRIVATE include)

# -------------------------
# ENABLE OPENMP
# -------------------------
find_package(OpenMP)

if (OpenMP_CXX_FOUND)
    message(STATUS "OpenMP FOUND, enabling...")
    target_compile_options(hpc_ode_sde PRIVATE ${OpenMP_CXX_FLAGS})
    target_link_libraries(hpc_ode_sde PRIVATE ${OpenMP_CXX_FLAGS})
else()
    message(WARNING "OpenMP NOT FOUND by CMake, adding -fopenmp manually for MinGW")
    target_compile_options(hpc_ode_sde PRIVATE -fopenmp)
    target_link_libraries(hpc_ode_sde PRIVATE -fopenmp)
endif()

# --- MINIMUM REQUIRED FOR MSYS2/MINGW -----
# 手动链接 gomp，解决 undefined reference to omp_get_wtime
target_link_libraries(hpc_ode_sde PRIVATE gomp)
