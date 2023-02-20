//---------------------------------------------------------------------------
// gemv.h
// Author: Frank Sossi
// 
// File contains: 
//		- gemv_kernel_part1_ver1() - Function for Naive Matrix Vector Multiplication
//		- gemv_part2_ver1() - Function for Shared memeory Matrix Vector Multiplication
//		- get_time() - Function to return time
//		- add_grid() - Kernel function to add two vectors
//		- add_block() - Kernel function to add two vectors for use with specified blocks
//		- add_thread() - Kernel function to add two vectors for use with specified blocks
//		- random_int() - Function to generate random integers
//		- Function for Shared memeory Matrix Vector Multiplication
//		- Function to return time
//		- Kernel function to add two vectors
// 
//---------------------------------------------------------------------------
#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>
#include <stdio.h>
#include <fstream>
#include <chrono>
#include <random>

// NOTE: one but not both of these should be defined
// Test parameters all 2's to check 
#define TESTPARAM
// Random values for vector and matrix
//#define REALDATA


#define REFERENCE
//#define PART1
#define PART2
#define DEBUG

// Threads 8 64 128 512 1024 
//constexpr auto THREAD_PER_BLOCK = 128;

// Size of the vector
constexpr int n = 32;
constexpr int THREAD_PER_BLOCK = 32;
constexpr int BLOCK_SIZE = 32;

//---------------------------------------------------------------------------
// Function for Naive Matrix Vector Multiplication
// Input: pointers to matrix, vector, and result vector, matrix dimensions
// Output: none
//---------------------------------------------------------------------------
template<typename T>
__global__ void gemv_kernel_part1_ver1(const T* matrix, const T* vector, T* result, const int row, const int col) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < row) {
        
        T sum = 0.0;
        
        for (int j = 0; j < col; j++) {
           // use fused multiply-add to improve performance 
           sum = fma(matrix[i * col + j], vector[j], sum); 
        }
        result[i] = sum;
    }

}

///---------------------------------------------------------------------------
// Function for Shared memeory Matrix Vector Multiplication
// Input: pointers to matrix, vector, and result vector, matrix dimensions
// Output: none
//---------------------------------------------------------------------------
template<typename T>
__global__ void gemv_part2_ver1(const T * matrix, const T * vector, T * result, const unsigned int rows, const unsigned int col)
{
    const unsigned int thread_index = threadIdx.x + blockIdx.x * blockDim.x;

    __shared__ T vector_shared[BLOCK_SIZE];

    T temp = 0.0;

    
    for (unsigned int i = 0; i < ((col + BLOCK_SIZE - 1)/ BLOCK_SIZE); ++i)
    {
        if ((i * BLOCK_SIZE + threadIdx.x) <  col) { 
            vector_shared[threadIdx.x] = vector[threadIdx.x + i * BLOCK_SIZE];
        }
        else{
            vector_shared[threadIdx.x] = 0.0;
        }
            
        __syncthreads();


        for (unsigned int j = 0; j < BLOCK_SIZE; ++j) {
            // Col ordering
            temp += matrix[thread_index + (j + BLOCK_SIZE * i) * rows] * vector_shared[j];

        }

        __syncthreads();
    }

    if (thread_index < rows){

        result[thread_index] = temp;
    }

}

//---------------------------------------------------------------------------
// Function to return time
// Input: none
// Output: returns time in nanoseconds
//---------------------------------------------------------------------------
std::chrono::high_resolution_clock::time_point get_time() {
  return std::chrono::high_resolution_clock::now();
}

//---------------------------------------------------------------------------
// Kernel function to add two vectors
// Input: pointers to two vectors and a pointer to the result vector
// Output: none
//---------------------------------------------------------------------------
__global__ void add_grid(int *a, int *b, int *c, int n) {
  
	// Calculate global thread ID
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	
	// Vector addition
	if (i < n) {
		c[i] = a[i] + b[i];
	}
}

//---------------------------------------------------------------------------
// Kernel function to add two vectors for use with specified blocks 
//	
// Input: pointers to two vectors and a pointer to the result vector
// Output: none
//---------------------------------------------------------------------------
__global__ void add_block(int *a, int *b, int *c) {
	
	c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];

}

//---------------------------------------------------------------------------
// Kernel function to add two vectors for use with specified blocks 
//	
// Input: pointers to two vectors and a pointer to the result vector
// Output: none
//---------------------------------------------------------------------------
__global__ void add_thread(int *a, int *b, int *c) {
	
	c[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];

}


//---------------------------------------------------------------------------
// Function to generate random integers
// Input: min and max values
// Output: returns random integer
//---------------------------------------------------------------------------
int random_int(int lower, int upper) {
  static std::mt19937 gen(std::chrono::high_resolution_clock::now().time_since_epoch().count());
  std::uniform_int_distribution<int> dist(lower, upper);
  return dist(gen);
}

//---------------------------------------------------------------------------
// Function write timing data to a file
// Input: vectors to hold timing data
// Output: results.csv
//---------------------------------------------------------------------------
template <typename T>
void write_data(std::vector<T> w_memory, std::vector<T> wo_memory, std::vector<int> size, int blocks, int threads) {
    std::ofstream file("results.csv", std::ios::app);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for writing." << std::endl;
        return;
    }


    for (int i = 0; i < w_memory.size(); i++) {
        file << blocks << "," << threads << "," << size[i] << "," 
			<< w_memory[i] << "," << wo_memory[i] << std::endl;
    }
}


//---------------------------------------------------------------------------
// Function to add two vectors
// Input: pointers to two vectors and a pointer to the result vector and size
// Output: none
//---------------------------------------------------------------------------
template <typename T>
void vector_add(T* a, T* b, T* c, int n) {
	for (int i = 0; i < n; i++) {
		c[i] = a[i] + b[i];
	}
}
