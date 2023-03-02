//---------------------------------------------------------------------------
// kernel.cu
// Author: Frank Sossi
//
// This program implements vector addition using CUDA. The program will
// use a template function as a reference and then compare the results
// to the CUDA implementation.
//
// To compile: nvcc -arch=sm_86 -lcublas kernel.cu -o lab3
// To run: ./lab3
// Note: -arch=sm_86 is your compute capability mine is 8.6
//---------------------------------------------------------------------------
#include <iostream>
#include <sstream>
#include <string>
#include <limits>
#include <vector>
#include <random>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "device_launch_parameters.h"
#include "gemv.h"

int main()
{

    int n = {};
    int THREAD_PER_BLOCK = {};
    std::string want_reference;
    int part_to_run;
    bool quit = false;

    while (!quit)
    {
        std::cout << "Enter the value of n (less than " << MAX_NUM << "), or 'q' to quit: ";
        std::string input;
        std::cin >> input;
        if (input == "q")
        {
            quit = true;
            continue;
        }

        while (!(std::stringstream(input) >> n) || n >= MAX_NUM)
        {
            std::cout << "Invalid input. Please enter the value of n (less than " << MAX_NUM << "), or 'q' to quit: ";
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            std::cin >> input;

            if (input == "q")
            {
                quit = true;
                break;
            }
        }
        if (quit)
        {
            break;
        }

        std::cout << "Enter the number of threads per block, or 'q' to quit: ";
        std::cin >> input;
        
        if (input == "q")
        {
            quit = true;
            continue;
        }

        while (!(std::stringstream(input) >> THREAD_PER_BLOCK))
        {
            std::cout << "Invalid input. Please enter the number of threads per block less than, or 'q' to quit: ";
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            std::cin >> input;
            if (input == "q")
            {
                quit = true;
                break;
            }
        }

        if (quit)
        {
            break;
        }

        std::cout << "Do you want to check reference? (y/n), or 'q' to quit: ";
        std::cin >> input;

        if (input == "q")
        {
            quit = true;
            continue;
        }

        while (input != "y" && input != "n")
        {
            std::cout << "Invalid input. Please enter 'y' or 'n' for whether you want to check reference, or 'q' to quit: ";
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            std::cin >> input;

            if (input == "q")
            {
                quit = true;
                break;
            }
        }

        if (quit)
        {
            break;
        }

        want_reference = input;

        std::cout << "Which part do you want to run? (1/2/3), or 'q' to quit: ";
        std::cin >> input;

        if (input == "q")
        {
            quit = true;
            continue;
        }

        while (!(std::stringstream(input) >> part_to_run) || part_to_run < 1 || part_to_run > 3)
        {
            std::cout << "Invalid input. Please enter 1, 2, or 3 for which part to run, or 'q' to quit: ";
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            std::cin >> input;

            if (input == "q")
            {
                quit = true;
                break;
            }
        }

        if (quit)
        {
            break;
        }

        // Vectors to hold timing data
        std::vector<long long> execution_w_memory;
        std::vector<long long> execution_wo_memory;

        // Allocate memory for each vector on host
        double* vector = (double*)malloc(n * sizeof(double));
        double* matrix = (double*)malloc(n * n * sizeof(double));
        double* ref_result = (double*)malloc(n * sizeof(double));
        double* calc_result = (double*)malloc(n * sizeof(double));

        // Random number generator
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0, 2);

        // Fill the vector with random numbers
        for (int i = 0; i < n; i++)
        {
            vector[i] = dis(gen);
        }

        // Fill the matrix with random numbers
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                matrix[i * n + j] = dis(gen);
            }
        }

        // Allocate pointers to GPU memory
        double* device_vector2 = nullptr;
        double* device_matrix2 = nullptr;
        double* device_result2 = nullptr;

        // Get time before allocating memory on GPU
        auto w_memory_start = get_time();

        cudaMalloc((void**)&device_vector2, n * sizeof(double));
        cudaMalloc((void**)&device_matrix2, n * n * sizeof(double));
        cudaMalloc((void**)&device_result2, n * sizeof(double));

        // Copy input data to GPU memory
        cudaMemcpy(device_vector2, vector, n * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(device_matrix2, matrix, n * n * sizeof(double), cudaMemcpyHostToDevice);

        const int NUM_BLOCKS = (n + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;
        const int BLOCKS = std::min(NUM_BLOCKS, MAX_BLOCKS);
        dim3 grid(BLOCKS,            1, 1);
        dim3 block(THREAD_PER_BLOCK, 1, 1);

        auto wo_memory_start = get_time();

        switch (part_to_run)
        {
        case 1:
            gemv_kernel_part1<<<grid, block>>>(device_matrix2, device_vector2, device_result2, n, n);
            break;
        case 2:
            gemv_part2_ver1_1<<<grid, block>>>(device_matrix2, device_vector2,
                                               device_result2, n, n, THREAD_PER_BLOCK);
            break;
        case 3:
            gemv_kernel_part3<<<grid, block>>>(device_matrix2, device_vector2, device_result2, n, n);
            break;
        default:
            std::cerr << "Invalid part to run" << std::endl;
            return 1;
        }

        auto wo_memory_end = get_time();

        cudaError_t err = cudaDeviceSynchronize();
        
        if (err != cudaSuccess)
        {
            printf("Kernel launch failed with error code %d: %s\n", err, cudaGetErrorString(err));
        }

        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("Kernel encountered an error: %d: %s\n", err, cudaGetErrorString(err));
        }

        // Copy the result from GPU memory to host memory
        cudaMemcpy(calc_result, device_result2, n * sizeof(double), cudaMemcpyDeviceToHost);

        cudaFree(device_vector2);
        cudaFree(device_matrix2);
        cudaFree(device_result2);

        auto w_memory_end = get_time();

        // Calculate time for memory allocation
        auto w_memory_time = std::chrono::duration_cast<std::chrono::nanoseconds>(w_memory_end - w_memory_start).count();

        // Print time for memory allocation
        std::cout << "Time with memory allocation: " << w_memory_time << std::endl;

        // Calculate time without memory allocation
        auto wo_memory_time = std::chrono::duration_cast<std::chrono::nanoseconds>(wo_memory_end - wo_memory_start).count();

        // Print time without memory allocation
        std::cout << "Time without memory allocation: " << wo_memory_time << std::endl;

        if (want_reference == "y")
        {
            // Code to perform reference check

            // Allocate pointers to GPU memory
            double* device_vector = nullptr;
            double* device_matrix = nullptr;
            double* device_result = nullptr;

            cudaMalloc((void**)&device_vector, n * sizeof(double));
            cudaMalloc((void**)&device_matrix, n * n * sizeof(double));
            cudaMalloc((void**)&device_result, n * sizeof(double));

            // Copy input data to GPU memory
            cudaMemcpy(device_vector, vector, n * sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(device_matrix, matrix, n * n * sizeof(double), cudaMemcpyHostToDevice);

            // Create a handle for cuBLAS
            cublasHandle_t handle;
            cublasCreate(&handle);

            // Perform the vector-matrix multiplication using cuBLAS
            // Perform the matrix-vector multiplication using cuBLAS
            double alpha = 1.0;
            double beta = 0.0;
            cublasDgemv(handle, CUBLAS_OP_N, n, n, &alpha, device_matrix, n, device_vector, 1, &beta, device_result, 1);


            // Copy the result from GPU memory to host memory
            cudaMemcpy(ref_result, device_result, n * sizeof(double), cudaMemcpyDeviceToHost);

            // Destroy the cuBLAS handle
            cublasDestroy(handle);

            cudaFree(device_vector);
            cudaFree(device_matrix);
            cudaFree(device_result);

            // verify the result
            double error = 0.0;
            for (int i = 0; i < n; i++)
            {
                error += abs((ref_result[i] - calc_result[i]));
            }
            // find average error
            error /= n;
            std::cout << "Average Error: " << error << std::endl;
        }

        free(vector);
        free(matrix);
        free(ref_result);
        free(calc_result);
    }

    return 0;
}
