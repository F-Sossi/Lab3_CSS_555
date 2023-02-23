//---------------------------------------------------------------------------
// gemv.h
// Author: Frank Sossi
// 
// File contains: 
//		- gemv_kernel_part1_ver1() - Function for Naive Matrix Vector Multiplication
//		- gemv_part2_ver1() - Function for Shared memeory Matrix Vector Multiplication
//		- gemv_kernel_part3_ver1() - Function for Registers Matrix Vector Multiplication
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
//#define PART3
#define DEBUG
//#define DEBUGINPUT

// Size of the vector 8000 max for some reason
// Part 3 is dependent on this value so if this is changed, make sure to inspect and possibly update part 3 kernel
constexpr int n = 8000;
// NOTE For further inquiry part 2 over 128 threads per block is not working
constexpr int THREAD_PER_BLOCK = 128;
// this is the size of the block 
constexpr int TILE_SIZE = 32;

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

    __shared__ T vector_shared[TILE_SIZE];

    T temp = 0.0;

    
    for (unsigned int i = 0; i < ((col + TILE_SIZE - 1)/ TILE_SIZE); ++i)
    {
        if ((i * TILE_SIZE + threadIdx.x) <  col) { 
            vector_shared[threadIdx.x] = vector[threadIdx.x + i * TILE_SIZE];
        }
        else{
            vector_shared[threadIdx.x] = 0.0;
        }
            
        __syncthreads();


        for (unsigned int j = 0; j < TILE_SIZE; ++j) {
            // Col ordering
            temp += matrix[thread_index + (j + TILE_SIZE * i) * rows] * vector_shared[j];

        }

        __syncthreads();
    }

    if (thread_index < rows){

        result[thread_index] = temp;
    }

}

///---------------------------------------------------------------------------
// Function for Shared memeory Matrix Vector Multiplication incorporates grid 
//      stride loop
// Input: pointers to matrix, vector, and result vector, matrix dimensions
// Output: none
//---------------------------------------------------------------------------
template<typename T>
__global__ void gemv_part2_ver2(const T * matrix, const T * vector, T * result, const unsigned int rows, const unsigned int col)
{
    // Calculate the thread index
    const unsigned int thread_index = threadIdx.x + blockIdx.x * blockDim.x;
    // Calculate the stride between threads
    const unsigned int stride = gridDim.x * blockDim.x;

    // Allocate shared memory for the vector
    __shared__ T vector_shared[TILE_SIZE];

    // Initialize the temporary variable to zero
    T temp = 0.0;

    // Iterate over the rows of the matrix using a grid-stride loop
    for (unsigned int row = thread_index; row < rows; row += stride) {

        // Iterate over the columns of the matrix using the existing loop code
        for (unsigned int i = 0; i < ((col + TILE_SIZE - 1)/ TILE_SIZE); ++i)
        {
            // Load a subset of the vector into shared memory
            if ((i * TILE_SIZE + threadIdx.x) <  col) { 
                vector_shared[threadIdx.x] = vector[threadIdx.x + i * TILE_SIZE];
            }
            else{
                vector_shared[threadIdx.x] = 0.0;
            }

            // Synchronize threads to ensure all data is loaded into shared memory
            __syncthreads();

            // Compute the dot product of the matrix and the vector subset
            for (unsigned int j = 0; j < TILE_SIZE; ++j) {
                // Col ordering
                temp += matrix[row + (j + TILE_SIZE * i) * rows] * vector_shared[j];
            }

            // Synchronize threads to ensure all data is used before modifying shared memory
            __syncthreads();
        }

        // Store the result for the current row in global memory
        if (thread_index < rows){
            result[row] = temp;
            // Reset the temporary variable to zero for the next row
            temp = 0.0;
        }
    }
}

///---------------------------------------------------------------------------
// Function for Shared memeory Matrix Vector Multiplication incorporates grid 
//      stride loop and more efficient memory access caching both matrix and
//      vector in shared memory
// Input: pointers to matrix, vector, and result vector, matrix dimensions
// Output: none
//---------------------------------------------------------------------------

template<typename T>
__global__ void gemv_part2_ver3(const T * matrix, const T * vector, T * result, const unsigned int rows, const unsigned int col)
{
    const unsigned int thread_index = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int stride = gridDim.x * blockDim.x;

    __shared__ T matrix_shared[TILE_SIZE * TILE_SIZE];
    __shared__ T vector_shared[TILE_SIZE];

    T temp = 0.0;

    for (unsigned int row = thread_index; row < rows; row += stride) {

        for (unsigned int i = 0; i < ((col + TILE_SIZE - 1)/ TILE_SIZE); ++i)
        {
            // Load a block of the matrix into shared memory
            for (unsigned int j = threadIdx.x; j < TILE_SIZE; j += blockDim.x) {
                matrix_shared[threadIdx.x * TILE_SIZE + j] = matrix[(i * TILE_SIZE + j) * rows + row];
            }

            // Load a block of the vector into shared memory
            if (threadIdx.x == 0) {
                for (unsigned int j = 0; j < TILE_SIZE; j++) {
                    if (i * TILE_SIZE + j < col) {
                        vector_shared[j] = vector[i * TILE_SIZE + j];
                    } else {
                        vector_shared[j] = 0.0;
                    }
                }
            }

            // Synchronize threads to ensure all data is loaded into shared memory
            __syncthreads();

            // Compute the dot product of the matrix and the vector block
            for (unsigned int j = 0; j < TILE_SIZE; ++j) {
                // Row ordering
                temp += matrix_shared[threadIdx.x * TILE_SIZE + j] * vector_shared[j];
            }

            // Synchronize threads to ensure all data is used before modifying shared memory
            __syncthreads();
        }

        // Store the result for the current row in global memory
        if (thread_index < rows){
            result[row] = temp;
            temp = 0.0;
        }
    }
}

//---------------------------------------------------------------------------
// Function for Registers Matrix Vector Multiplication
// Input: pointers to matrix, vector, and result vector, matrix dimensions
// Output: none
//---------------------------------------------------------------------------
template<typename T>
__global__ void gemv_kernel_part3_ver1(const T* matrix, const T* vector, T* result, const int row, const int col) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < row) {
        
        T sum = 0.0;
        
        for (int j = 0; j < col; j+=10) {
           // use fused multiply-add to improve performance 
           sum = fma(matrix[i * col + j], vector[j], sum);
	   sum = fma(matrix[i * col + (j + 1)], vector[(j + 1)], sum);
           sum = fma(matrix[i * col + (j + 2)], vector[(j + 2)], sum);
	   sum = fma(matrix[i * col + (j + 3)], vector[(j + 3)], sum);
           sum = fma(matrix[i * col + (j + 4)], vector[(j + 4)], sum);
	   sum = fma(matrix[i * col + (j + 5)], vector[(j + 5)], sum);
           sum = fma(matrix[i * col + (j + 6)], vector[(j + 6)], sum);
	   sum = fma(matrix[i * col + (j + 7)], vector[(j + 7)], sum);
           sum = fma(matrix[i * col + (j + 8)], vector[(j + 8)], sum);
	   sum = fma(matrix[i * col + (j + 9)], vector[(j + 9)], sum);
        }
        result[i] = sum;
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
