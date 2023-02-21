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
#include <vector>
#include <chrono>
#include <random>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>
#include "device_launch_parameters.h"
#include "gemv.h"

// Block and thread count to run different configurations when testing
//   different block sizes use add_bloc() when testing different 
//   thread counts use add_thread() add this to check added

// Threads 8 64 128 512 1024 
//constexpr auto THREAD_PER_BLOCK = 128;



constexpr int MEMORY_STRIDE       = 33;
// constexpr int THREAD_PER_BLOCK = 32;


void part_1(double* vector, double* matrix, double* result, unsigned int N)
{
	cudaEvent_t start;
	cudaEvent_t stop;
	float milliseconds = 0;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
	
	// Allocate pointers to GPU memory
	double* device_vector2 = nullptr;
	double* device_matrix2 = nullptr;
	double* device_result2 = nullptr;

	cudaMalloc((void**)&device_vector2, N * MEMORY_STRIDE     * sizeof(double));
	cudaMalloc((void**)&device_matrix2, N * N * MEMORY_STRIDE * sizeof(double));
	cudaMalloc((void**)&device_result2, N *  MEMORY_STRIDE    * sizeof(double));

	// Copy input data to GPU memory
	cudaMemcpy(device_vector2, vector, N *     sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(device_matrix2, matrix, N * N * sizeof(double), cudaMemcpyHostToDevice);

	// Call gemv_kernel
	// dim3 num_blocks(N);
	// dim3 threads_per_block(THREAD_PER_BLOCK);
	// <<<grid_size, block_size>>
	// <<number_of_blocks, number_of_threads>>
	// The maximum number of threads in the block is limited to 1024.

	cudaEventRecord(start, 0);

	// Call the matrix vector - addition
	gemv_kernel2<<<(N + 31)/N, N>>>(device_matrix2, device_vector2, device_result2, N, N);

	// Copy the result from GPU memory to host memory
	cudaMemcpy(result, device_result2, N * sizeof(double), cudaMemcpyDeviceToHost);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Time elapsed: %f ms\n", milliseconds);

	cudaFree(device_vector2);
	cudaFree(device_matrix2);
	cudaFree(device_result2);
}


int main() 
{
	// Vectors to hold timing data
	// std::vector<long long> execution_w_memory;
	// std::vector<long long> execution_wo_memory;

	unsigned int N = 32;

	// Allocate memory for each vector on host
	double* vector      = (double*)malloc(N *     sizeof(double));
	double* matrix      = (double*)malloc(N * N * sizeof(double));
	double* result = (double*)malloc(N *     sizeof(double));

	// Random number generator
	std::random_device rd;
	std::mt19937 gen(rd()); // C++ standard 32 bit Mersenne Twister 19937 generator
	std::uniform_real_distribution<> dis(0, 10);

	// Fill vector with random numbers
	for (int i = 0; i < N; i++) 
	{
		vector[i] = dis(gen);
	}

	// Fill matrix with random numbers
	for (int i = 0; i < N; i++) 
	{
		for (int j = 0; j < N; j++) 
		{
			matrix[i * N + j] = dis(gen);
		}
	}

	// Print vector
	std::cout << "Vector" << std::endl;
	for (int i = 0; i < N; i++) 
	{
		std::cout << vector[i] << " ";
	}
	std::cout << std::endl;

	// Print matrix
	std::cout << "Matrix" << std::endl;
	for (int i = 0; i < N; i++) 
	{
		for (int j = 0; j < N; j++) 
		{
			std::cout << matrix[i * N + j] << " ";
		}
		std::cout << std::endl;
	}

	part_1(vector, matrix, result, N);

	// Print calculated result
	std::cout << "Calculated Result" << std::endl;
	for (int i = 0; i < N; i++)
	{
		std::cout << result[i] << " ";
	}

	std::cout << std::endl;

	free(vector);
	free(matrix);
	free(result); 
}
