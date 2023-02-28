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
#include <random>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "device_launch_parameters.h"
#include "gemv.h"

int main() {
	
	// vectors to hold timing data
	std::vector<long long> execution_w_memory;
	std::vector<long long> execution_wo_memory;


	// Allocate memory for each vector on host
	double* vector = (double*)malloc(n * sizeof(double));
	double* matrix = (double*)malloc(n * n * sizeof(double));
	double* ref_result = (double*)malloc(n * sizeof(double));		
	double* calc_result = (double*)malloc(n * sizeof(double));

#ifdef TESTPARAM

	// fill vector with 2's
	for (int i = 0; i < n; i++) {
		vector[i] = 2;
	}

	// fill matrix with 2's
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			matrix[i * n + j] = 2;
		}
	}

#endif

#ifdef REALDATA

	// random number generator
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> dis(0, 2);

	// fill vector with random numbers
	for (int i = 0; i < n; i++) {
		vector[i] = dis(gen);
	}

	// fill matrix with random numbers
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			matrix[i * n + j] = dis(gen);
		}
	}

#endif

#ifdef DEBUGINPUT

	// print vector
	std::cout << "Vector" << std::endl;
	for (int i = 0; i < n; i++) {
		std::cout << vector[i] << " ";
	}
	std::cout << std::endl;

	// print matrix
	std::cout << "Matrix" << std::endl;
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			std::cout << matrix[i * n + j] << " ";
		}
		std::cout << std::endl;
	}

#endif

#ifdef REFERENCE

	// allocate pointers to GPU memory
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
		
#endif


	// allocate pointers to GPU memory
	double* device_vector2 = nullptr;
	double* device_matrix2 = nullptr;
	double* device_result2 = nullptr;

	// get time before allocating memory on GPU
	auto w_memory_start = get_time();

	cudaMalloc((void**)&device_vector2, n * sizeof(double));
	cudaMalloc((void**)&device_matrix2, n * n * sizeof(double));
	cudaMalloc((void**)&device_result2, n * sizeof(double));

	// Copy input data to GPU memory
	cudaMemcpy(device_vector2, vector, n * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(device_matrix2, matrix, n * n * sizeof(double), cudaMemcpyHostToDevice);


	const int num_blocks = (n + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;
	//const int max_blocks = 32767;
	const int blocks = std::min(num_blocks, max_blocks);
	dim3 grid(blocks, 1, 1);
	dim3 block(THREAD_PER_BLOCK, 1, 1);

	auto wo_memory_start = get_time();
    
	// run part 1 Kernel
	#ifdef PART1
	gemv_kernel_part1<<<grid, block>>>(device_matrix2, device_vector2, device_result2, n, n);
	#endif
	// run part 2 Kernel
	#ifdef PART2
	gemv_part2_ver1_1<<<grid, block>>>(device_matrix2, device_vector2, device_result2, n, n);
	#endif
	// run part 3 Kernel
	#ifdef PART3
	gemv_kernel_part3<<<grid, block>>>(device_matrix2, device_vector2, device_result2, n, n);
	#endif

	auto wo_memory_end = get_time();

	cudaError_t err = cudaDeviceSynchronize();
	if (err != cudaSuccess) {
		printf("Kernel launch failed with error code %d: %s\n", err, cudaGetErrorString(err));
	}

	err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("Kernel encountered an error: %d: %s\n", err, cudaGetErrorString(err));
	}
	
	// Copy the result from GPU memory to host memory
	cudaMemcpy(calc_result, device_result2, n * sizeof(double), cudaMemcpyDeviceToHost);

	cudaFree(device_vector2);
	cudaFree(device_matrix2);
	cudaFree(device_result2);

	auto w_memory_end = get_time();

	// calculate time for memory allocation
	auto w_memory_time = std::chrono::duration_cast<std::chrono::nanoseconds>(w_memory_end - w_memory_start).count();

	// print time for memory allocation
	std::cout << "Time with memory allocation: " << w_memory_time << std::endl;

	// calculate time without memory allocation
	auto wo_memory_time = std::chrono::duration_cast<std::chrono::nanoseconds>(wo_memory_end - wo_memory_start).count();

	// print time without memory allocation
	std::cout << "Time without memory allocation: " << wo_memory_time << std::endl;



#ifdef DEBUG

	// print reference result
	std::cout << "Reference Result" << std::endl;
	for (int i = 0; i < n; i++) {
		std::cout << ref_result[i] << " ";
	}
	std::cout << std::endl;

	// print calculated result
	std::cout << "Calculated Result" << std::endl;
	for (int i = 0; i < n; i++) {
		std::cout << calc_result[i] << " ";
	}
	std::cout << std::endl;

#endif

#ifdef REFERENCE

	// verify the result
	double error = 0.0;
	for (int i = 0; i < n; i++) {
		error += (ref_result[i] - calc_result[i]);
	}
	// find average error
	error /= n;
	std::cout << "Average Error: " << error << std::endl;

#endif

	free(vector);
	free(matrix);
	free(ref_result);
	free(calc_result);
    
}
