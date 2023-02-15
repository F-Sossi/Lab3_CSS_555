//---------------------------------------------------------------------------
// kernel.cu
// Author: Frank Sossi
// 
// This program implements vector addition using CUDA. The program will
// use a template function as a reference and then compare the results
// to the CUDA implementation.
// 
// Program compiled with Visual Studio 2022 Community.
// Program requires:
//
// CUDA Developer toolkit will detect and integrate with Visual Studio 2022 Community when installed.
// System: requirements: CUDA capable GPU compute 6.1 or higher [1].
// 1. Install CUDA Developer Toolkit
//     CUDA Toolkit 12.0
// 2. Open with Visual Studio 2022
//	Visual Studio
//	1. Open Existing project
//	2. Select folder {Lab2_CSS_535}
//	3. Open file {2Lab2_CSS_535.sln}
//	4. Set Platform x64
//	5. Build -> Build Solution
//---------------------------------------------------------------------------
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cuda_runtime.h>
#include <cmath>
#include "device_launch_parameters.h"
#include "vector_add.h"

// Block and thread count to run different configurations when testing
//   different block sizes use add_bloc() when testing different 
//   thread counts use add_thread() add this to check

// Threads 8 64 128 512 1024 
constexpr auto THREAD_PER_BLOCK = 128;

// Blocks 1 2 4 8 16 32 64 128
constexpr auto BLOCK_PER_GRID = 1;



int main() {

	// vectors to hold timing data
	std::vector<long long> execution_w_memory;
	std::vector<long long> execution_wo_memory;
	std::vector<int> data_size;

    // for loop that goes from 2^5 to 2^20
	for (int i = 5; i <= 20; i++) {
		
		// calculate data size
		int n = pow(2, i);
		
		// store it for output
		data_size.push_back(n);
		
		// allocate pointers to GPU memory
		int* device_a = nullptr;
		int* device_b = nullptr;
	    int* device_c = nullptr;

		// Allocate memory for each vector on host
		int* vector_a = (int*)malloc(n * sizeof(int));
		int* vector_b = (int*)malloc(n * sizeof(int));
		int* vector_c = (int*)malloc(n * sizeof(int));
		int* vector_d = (int*)malloc(n * sizeof(int));

		// Allocate memory for each vector on GPU
		cudaMalloc((void**)&device_a, n * sizeof(int));
		cudaMalloc((void**)&device_b, n * sizeof(int));
		cudaMalloc((void**)&device_c, n * sizeof(int));

		// Fill vectors with random integers
		for (int i = 0; i < n; i++) {
			vector_a[i] = random_int(-10, 10);
			vector_b[i] = random_int(-10, 10);
		}
		
		// run reference function 
		vector_add(vector_a, vector_b, vector_d, n);
		
		
		// Begin CUDA function
		auto w_memory = get_time();
		
		// Copy host vectors to device
		cudaMemcpy(device_a, vector_a, n * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(device_b, vector_b, n * sizeof(int), cudaMemcpyHostToDevice);

		// variables to work with balanced threads and blocks add_grid()
		int thread_per_block = THREAD_PER_BLOCK;
		int block_num = (n + thread_per_block - 1) / thread_per_block;
		dim3 block_dim(thread_per_block, 1, 1);
		dim3 grid_dim(block_num, 1, 1);

		
		// Call CUDA kernel
		auto wo_memory = get_time();
		//add_grid<<<grid_dim, block_dim>>> (device_a, device_b, device_c, n);
		//add_thread <<< BLOCK_PER_GRID, n >>> (device_a, device_b, device_c);
		add_block <<< n, THREAD_PER_BLOCK >>> (device_a, device_b, device_c);
		auto wo_end = get_time();
		
		//Synchronize threads
		cudaDeviceSynchronize();

		// Copy array c back to host
		cudaMemcpy(vector_c, device_c, n * sizeof(int), cudaMemcpyDeviceToHost);
		auto w_end = get_time();

		// Calculate execution time
		auto w_memory_time = std::chrono::duration_cast<std::chrono::nanoseconds>(w_end - w_memory).count();
		auto wo_memory_time = std::chrono::duration_cast<std::chrono::nanoseconds>(wo_end - wo_memory).count();
		
		// push back execution time to vector
		execution_w_memory.push_back(w_memory_time);
		execution_wo_memory.push_back(wo_memory_time);

		// print line and data size
		std::cout << "Run number: " << i << std::endl;
		std::cout << "Data size: " << n << std::endl;
		std::cout << "Execution time with memory transfer: " << w_memory_time << " ns" << std::endl;
		std::cout << "Execution time without memory transfer: " << wo_memory_time << " ns" << std::endl;
		std::cout << "Speedup: " << (double)w_memory_time / (double)wo_memory_time << std::endl;
		std::cout << "________________________________________________________________\n" << std::endl;

		//Check the returned vector print error if calculations are incorrect
		for (int i = 0; i < n; i++) {
			if (vector_c[i] != vector_d[i]) {
				std::cout << "Error: vector_c and vector_d do not match" << std::endl;
				
				// print the values that do not match
				std::cout << "Vector_c: " << vector_c[i] << " Vector_d(ref): " << vector_d[i] << std::endl;

				break;
			}
		}

		// Free memory on device
		cudaFree(device_a);
		cudaFree(device_b);
		cudaFree(device_c);

		// Free memory on host
		free(vector_a);
		free(vector_b);
		free(vector_c);
		free(vector_d);
		
		write_data(execution_w_memory, execution_wo_memory, data_size, block_num, THREAD_PER_BLOCK);
	}
    
    
}
