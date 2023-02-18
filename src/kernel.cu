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
#include "vector_add.h"

// Block and thread count to run different configurations when testing
//   different block sizes use add_bloc() when testing different 
//   thread counts use add_thread() add this to check added

#define REFERENCE
#define DEBUG

// Threads 8 64 128 512 1024 
//constexpr auto THREAD_PER_BLOCK = 128;

// Size of the vector
constexpr int n = 10;


int main() {
	
	// vectors to hold timing data
	std::vector<long long> execution_w_memory;
	std::vector<long long> execution_wo_memory;

	// allocate pointers to GPU memory
	double* device_vector = nullptr;
	double* device_matrix = nullptr;
	double* device_result = nullptr;

	cudaMalloc((void**)&device_vector, n * sizeof(double));
	cudaMalloc((void**)&device_matrix, n * n * sizeof(double));
	cudaMalloc((void**)&device_result, n * n * sizeof(double));

	// Allocate memory for each vector on host
	double* vector = (double*)malloc(n * sizeof(double));
	double* matrix = (double*)malloc(n * n * sizeof(double));
	double* ref_result = (double*)malloc(n * sizeof(double));		
	double* calc_result = (double*)malloc(n * sizeof(double));		

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

#ifdef DEBUG

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

#endif



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


	cudaFree(device_vector);
	cudaFree(device_matrix);
	cudaFree(device_result);
		
    
}
