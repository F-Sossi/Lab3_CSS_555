//---------------------------------------------------------------------------
// gemv.h
// Author: Frank Sossi
//
// File contains:
//		- gemv_kernel_part1_ver1() - Function for Naive Matrix Vector
// Multiplication
//		- gemv_part2_ver1() - Function for Shared memeory Matrix Vector
// Multiplication
//		- gemv_kernel_part3_ver1() - Function for Registers Matrix
// Vector Multiplication
//		- get_time() - Function to return time
//		- add_grid() - Kernel function to add two vectors
//		- add_block() - Kernel function to add two vectors for use with
// Specified blocks
//		- add_thread() - Kernel function to add two vectors for use
//with
// Specified blocks
//		- Random_int() - Function to generate random integers
//		- Function for Shared memeory Matrix Vector Multiplication
//		- Function to return time
//		- Kernel function to add two vectors
//
//---------------------------------------------------------------------------
#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <chrono>
#include <fstream>
#include <math.h>
#include <random>
#include <stdio.h>

// NOTE: one but not botmakeh of these should be defined
// Test parameters all 2's to check
// #define TESTPARAM
// Random values for vector and matrix
#define REALDATA

// #define REFERENCE
// #define PART1
// #define PART2
// #define PART3
// #define DEBUG
// #define DEBUGINPUT
// #define DEBUG_KERNEL
// #define VERIFY

// Size of the vector 8000 max for some reason
// Part 3 is dependent on this value so if this is changed, make sure to
// inspect and possibly update part 3 kernel constexpr int n = 10000;

const int n = 16384;

const int MAX_NUM = 20000;

// This is the size of the block
constexpr int TILE_SIZE = 1024;

// Max number of blocks as per spec
constexpr int max_blocks = 32767;

//---------------------------------------------------------------------------
// Function for Naive Matrix Vector Multiplication #WORKs
// Input: pointers to matrix, vector, and result vector, matrix dimensions
//        without fma
// Output: none
//---------------------------------------------------------------------------
template <typename T>
__global__ void
gemv_kernel_part1(
    const T *matrix, const T *vector, T *result, const int row, const int col)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < row)
    {
        T sum = 0.0;

        for (int j = 0; j < col; j++)
        {
            sum += matrix[j * row + i] * vector[j]; // multiply and accumulate
                                                    // with row-major ordering
        }
        result[i] = sum;
    }
}

///---------------------------------------------------------------------------
// Function for Shared memeory Matrix Vector Multiplication
// Input: pointers to matrix, vector, and result vector, matrix dimensions
// Output: none
//---------------------------------------------------------------------------
template <typename T>
__global__ void
gemv_part2_ver1(const T *matrix,
                const T *vector,
                T *result,
                const unsigned int rows,
                const unsigned int col)
{
    const unsigned int thread_index = threadIdx.x + blockIdx.x * blockDim.x;

    //__shared__ T vector_shared[TILE_SIZE];
    extern __shared__ T vector_shared[];

    T temp = 0.0;

#ifdef DEBUGKERNEL
    printf("thread_index = %u\n", thread_index);
#endif

    for (unsigned int i = 0; i < ((col + TILE_SIZE - 1) / TILE_SIZE); ++i)
    {
        if ((i * TILE_SIZE + threadIdx.x) < col)
        {
            vector_shared[threadIdx.x] = vector[threadIdx.x + i * TILE_SIZE];
        }
        else
        {
            vector_shared[threadIdx.x] = 0.0;
        }

        __syncthreads();

        for (unsigned int j = 0; j < TILE_SIZE; ++j)
        {
            // Col ordering
            temp += matrix[thread_index + (j + TILE_SIZE * i) * rows]
                    * vector_shared[j];
        }

        __syncthreads();
    }

    if (thread_index < rows)
    {
        result[thread_index] = temp;
    }

#ifdef DEBUGKERNEL
    printf("thread_index = %u, result[%u] = %f\n", thread_index, thread_index,
           result[thread_index]);
#endif
}

///---------------------------------------------------------------------------
// Function for Shared memeory Matrix Vector Multiplication
// Input: pointers to matrix, vector, and result vector, matrix dimensions
// Output: debug info
//---------------------------------------------------------------------------
template <typename T>
__global__ void
gemv_part2_ver1_1(const T *matrix,
                  const T *vector,
                  T *result,
                  const unsigned int rows,
                  const unsigned int col, const unsigned int threads)
{
    // Compute the thread index
    const unsigned int thread_index = threadIdx.x + blockIdx.x * blockDim.x;

    // Allocate shared memory for the vector
    __shared__ T vector_shared[TILE_SIZE];

    // Initialize the temporary sum to zero
    T temp = 0.0;

#ifdef DEBUG_KERNEL
    // Print a debug message when a thread starts
    printf("Thread %u starting\n", thread_index);
#endif

    // Loop over the columns of the matrix, processing TILE_SIZE columns at a
    // time
    for (unsigned int i = 0; i < ((col + threads - 1) / threads); ++i)
    {
        // Load a block of the vector into shared memory
        if ((i * threads + threadIdx.x) < col)
        {
            vector_shared[threadIdx.x] = vector[threadIdx.x + i * threads];
        }
        else
        {
            vector_shared[threadIdx.x] = 0.0;
        }

        // Wait for all threads to finish loading the vector
        __syncthreads();

#ifdef DEBUG_KERNEL
        // Print a debug message showing the contents of the shared vector
        printf("Thread %u vector_shared[%u] = %f\n", thread_index, threadIdx.x,
               vector_shared[threadIdx.x]);
#endif

        // Compute the dot product of the matrix block and the vector block
        for (unsigned int j = 0; j < threads; ++j)
        {
            // Check that the column is within the bounds of the matrix
            if ((j + threads * i) < col)
            {
                // Col ordering
                temp += matrix[thread_index + (j + threads * i) * rows]
                        * vector_shared[j];

#ifdef DEBUG_KERNEL
                // Print a debug message showing the computation being
                // performed
                printf("Thread %u computing temp = %f + %f * %f\n",
                       thread_index, temp,
                       matrix[thread_index + (j + TILE_SIZE * i) * rows],
                       vector_shared[j]);
#endif
            }
        }

        // Wait for all threads to finish the computation
        __syncthreads();

#ifdef DEBUG_KERNEL
        // Print a debug message when a loop is completed
        printf("Thread %u completed loop %u\n", thread_index, i);
#endif
    }

    // Store the result in global memory
    if (thread_index < rows)
    {
        result[thread_index] = temp;

#ifdef DEBUG_KERNEL
        // Print a debug message showing the result being stored
        printf("Thread %u completed. result[%u] = %f\n", thread_index,
               thread_index, result[thread_index]);
#endif
    }
}

///---------------------------------------------------------------------------
// Function for Shared memeory Matrix Vector Multiplication incorporates grid
// stride loop
// Input: pointers to matrix, vector, and result vector, matrix dimensions
// Output: none
//---------------------------------------------------------------------------
template <typename T>
__global__ void
gemv_part2_ver2(const T *matrix,
                const T *vector,
                T *result,
                const unsigned int rows,
                const unsigned int col)
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
    for (unsigned int row = thread_index; row < rows; row += stride)
    {
        // Iterate over the columns of the matrix using the existing loop code
        for (unsigned int i = 0; i < ((col + TILE_SIZE - 1) / TILE_SIZE); ++i)
        {
            // Load a subset of the vector into shared memory
            if ((i * TILE_SIZE + threadIdx.x) < col)
            {
                vector_shared[threadIdx.x]
                    = vector[threadIdx.x + i * TILE_SIZE];
            }
            else
            {
                vector_shared[threadIdx.x] = 0.0;
            }

            // Synchronize threads to ensure all data is loaded into shared
            // memory
            __syncthreads();

            // Compute the dot product of the matrix and the vector subset
            for (unsigned int j = 0; j < TILE_SIZE; ++j)
            {
                // Col ordering
                temp += matrix[row + (j + TILE_SIZE * i) * rows]
                        * vector_shared[j];
            }

            // Synchronize threads to ensure all data is used before modifying
            // shared memory
            __syncthreads();
        }

        // Store the result for the current row in global memory
        if (thread_index < rows)
        {
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

template <typename T>
__global__ void
gemv_part2_ver3(const T *matrix,
                const T *vector,
                T *result,
                const unsigned int rows,
                const unsigned int col)
{
    const unsigned int thread_index = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int stride       = gridDim.x * blockDim.x;

    __shared__ T matrix_shared[TILE_SIZE * TILE_SIZE];
    __shared__ T vector_shared[TILE_SIZE];

    T temp = 0.0;

    for (unsigned int row = thread_index; row < rows; row += stride)
    {
        for (unsigned int i = 0; i < ((col + TILE_SIZE - 1) / TILE_SIZE); ++i)
        {
            // Load a block of the matrix into shared memory
            for (unsigned int j = threadIdx.x; j < TILE_SIZE; j += blockDim.x)
            {
                matrix_shared[threadIdx.x * TILE_SIZE + j]
                    = matrix[(i * TILE_SIZE + j) * rows + row];
            }

            // Load a block of the vector into shared memory
            if (threadIdx.x == 0)
            {
                for (unsigned int j = 0; j < TILE_SIZE; j++)
                {
                    if (i * TILE_SIZE + j < col)
                    {
                        vector_shared[j] = vector[i * TILE_SIZE + j];
                    }
                    else
                    {
                        vector_shared[j] = 0.0;
                    }
                }
            }

            // Synchronize threads to ensure all data is loaded into shared
            // memory
            __syncthreads();

            // Compute the dot product of the matrix and the vector block
            for (unsigned int j = 0; j < TILE_SIZE; ++j)
            {
                // Row ordering
                temp += matrix_shared[threadIdx.x * TILE_SIZE + j]
                        * vector_shared[j];
            }

            // Synchronize threads to ensure all data is used before modifying
            // shared memory
            __syncthreads();
        }

        // Store the result for the current row in global memory
        if (thread_index < rows)
        {
            result[row] = temp;
            temp        = 0.0;
        }
    }
}

//---------------------------------------------------------------------------
// Function for Registers Matrix Vector Multiplication #WORKS
// Input: pointers to matrix, vector, and result vector, matrix dimensions
// Output: none
//---------------------------------------------------------------------------

template <typename T>
__global__ void
gemv_kernel_part3(
    const T *matrix, const T *vector, T *result, const int row, const int col)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < row)
    {
        T sum = 0.0;

        int j = 0;
        for (; j < col - 7; j += 8)
        {
            sum += matrix[j * row + i] * vector[j];
            sum += matrix[(j + 1) * row + i] * vector[(j + 1)];
            sum += matrix[(j + 2) * row + i] * vector[(j + 2)];
            sum += matrix[(j + 3) * row + i] * vector[(j + 3)];
            sum += matrix[(j + 4) * row + i] * vector[(j + 4)];
            sum += matrix[(j + 5) * row + i] * vector[(j + 5)];
            sum += matrix[(j + 6) * row + i] * vector[(j + 6)];
            sum += matrix[(j + 7) * row + i] * vector[(j + 7)];
        }
        for (; j < col; j++)
        {
            sum += matrix[j * row + i] * vector[j];
        }
        result[i] = sum;
    }
}

//---------------------------------------------------------------------------
// Function to return time
// Input: none
// Output: returns time in nanoseconds
//---------------------------------------------------------------------------
std::chrono::high_resolution_clock::time_point
get_time()
{
    return std::chrono::high_resolution_clock::now();
}

//---------------------------------------------------------------------------
// Function write timing data to a file
// Input: vectors to hold timing data
// Output: results.csv
//---------------------------------------------------------------------------
template <typename T>
void
write_data(std::vector<T> w_memory,
           std::vector<T> wo_memory,
           std::vector<int> size,
           int blocks,
           int threads)
{
    std::ofstream file("results.csv", std::ios::app);
    if (!file.is_open())
    {
        std::cerr << "Failed to open file for writing." << std::endl;
        return;
    }

    for (int i = 0; i < w_memory.size(); i++)
    {
        file << blocks << "," << threads << "," << size[i] << ","
             << w_memory[i] << "," << wo_memory[i] << std::endl;
    }
}

template<typename T>
__global__ void gemv_part2_verML_overflow16(const T* matrix, const T* vector, T* result, const int row, const int col)
{   
    //int n = row;

    //T matrix_local[n / 4];
    __shared__ T vector_shared[n / 16];

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    T sum2 = 0.0;

    T sum = 0.0;
    // sum += matrix[j * row + i] * vector[j];

    if (index < row) {

        for (int i = 0; i < n / 16; i++) {
            //matrix_local[i] = matrix[i * row + index];
            vector_shared[i] = vector[i];
            //x++;
        }

        for (int j = 0; j < n / 16; j++) {
            sum += matrix[j * row + index] * vector_shared[j]; // multiply and accumulate with row-major ordering
        }

        int x = 0;
        for (int i = n / 16; i < n / 8; i++) {
            //matrix_local[x] = matrix[(i)*row + index];
            vector_shared[x] = vector[i];
            x++;
        }

        for (int j = 0; j < n / 16; j++) {
            sum += matrix[(j + (n / 16)) * row + index] * vector_shared[j]; // multiply and accumulate with row-major ordering
        }

        x = 0;
        for (int i = n / 8; i < (3 * n) / 16; i++) {
            //matrix_local[x] = matrix[(i)*row + index];
            vector_shared[x] = vector[i];
            x++;
        }

        for (int j = 0; j < n / 16; j++) {
            sum += matrix[(j + (n / 8)) * row + index] * vector_shared[j]; // multiply and accumulate with row-major ordering
        }

        x = 0;
        for (int i = (3 * n) / 16; i < n / 4; i++) {
            //matrix_local[x] = matrix[(i)*row + index];
            vector_shared[x] = vector[i];
            x++;
        }

        for (int j = 0; j < n / 16; j++) {
            sum += matrix[(j + ((3 * n) / 16)) * row + index] * vector_shared[j]; // multiply and accumulate with row-major ordering
        }

        x = 0;
        for (int i = n / 4; i < (5 * n) / 16; i++) {
            //matrix_local[i] = matrix[i * row + index];
            vector_shared[x] = vector[i];
            x++;
        }

        for (int j = 0; j < n / 16; j++) {
            sum += matrix[(j + (n / 4)) * row + index] * vector_shared[j]; // multiply and accumulate with row-major ordering
        }

        x = 0;
        for (int i = (5 * n) / 16; i < (3 * n) / 8; i++) {
            //matrix_local[x] = matrix[(i)*row + index];
            vector_shared[x] = vector[i];
            x++;
        }

        for (int j = 0; j < n / 16; j++) {
            sum += matrix[(j + ((5 * n) / 16)) * row + index] * vector_shared[j]; // multiply and accumulate with row-major ordering
        }

        x = 0;
        for (int i = (3 * n) / 8; i < (7 * n) / 16; i++) {
            //matrix_local[x] = matrix[(i)*row + index];
            vector_shared[x] = vector[i];
            x++;
        }

        for (int j = 0; j < n / 16; j++) {
            sum += matrix[(j + ((3 * n) / 8)) * row + index] * vector_shared[j]; // multiply and accumulate with row-major ordering
        }

        x = 0;
        for (int i = (7 * n) / 16; i < n / 2; i++) {
            //matrix_local[x] = matrix[(i)*row + index];
            vector_shared[x] = vector[i];
            x++;
        }

        for (int j = 0; j < n / 16; j++) {
            sum += matrix[(j + ((7 * n) / 16)) * row + index] * vector_shared[j]; // multiply and accumulate with row-major ordering
        }

        // halfway

        x = 0;
        for (int i = n / 2; i < (9 * n) / 16; i++) {
            //matrix_local[i] = matrix[i * row + index];
            vector_shared[x] = vector[i];
            x++;
        }

        for (int j = 0; j < n / 16; j++) {
            sum += matrix[(j + (n / 2)) * row + index] * vector_shared[j]; // multiply and accumulate with row-major ordering
        }

        x = 0;
        for (int i = (9 * n) / 16; i < (5 * n) / 8; i++) {
            //matrix_local[x] = matrix[(i)*row + index];
            vector_shared[x] = vector[i];
            x++;
        }

        for (int j = 0; j < n / 16; j++) {
            sum += matrix[(j + (9 * n) / 16) * row + index] * vector_shared[j]; // multiply and accumulate with row-major ordering
        }

        x = 0;
        for (int i = (5 * n) / 8; i < (11 * n) / 16; i++) {
            //matrix_local[x] = matrix[(i)*row + index];
            vector_shared[x] = vector[i];
            x++;
        }

        for (int j = 0; j < n / 16; j++) {
            sum += matrix[(j + (5 * n) / 8) * row + index] * vector_shared[j]; // multiply and accumulate with row-major ordering
        }

        x = 0;
        for (int i = (11 * n) / 16; i < (3 * n) / 4; i++) {
            //matrix_local[x] = matrix[(i)*row + index];
            vector_shared[x] = vector[i];
            x++;
        }

        for (int j = 0; j < n / 16; j++) {
            sum += matrix[(j + ((11 * n) / 16)) * row + index] * vector_shared[j]; // multiply and accumulate with row-major ordering
        }

        x = 0;
        for (int i = (3 * n) / 4; i < (13 * n) / 16; i++) {
            //matrix_local[i] = matrix[i * row + index];
            vector_shared[x] = vector[i];
            x++;
        }

        for (int j = 0; j < n / 16; j++) {
            sum += matrix[(j + (3 * n / 4)) * row + index] * vector_shared[j]; // multiply and accumulate with row-major ordering
        }

        x = 0;
        for (int i = (13 * n) / 16; i < (7 * n) / 8; i++) {
            //matrix_local[x] = matrix[(i)*row + index];
            vector_shared[x] = vector[i];
            x++;
        }

        for (int j = 0; j < n / 16; j++) {
            sum += matrix[(j + ((13 * n) / 16)) * row + index] * vector_shared[j]; // multiply and accumulate with row-major ordering
        }



        x = 0;
        for (int i = (7 * n) / 8; i < (15 * n) / 16; i++) {
            //matrix_local[x] = matrix[(i)*row + index];
            vector_shared[x] = vector[i];
            x++;
        }

        for (int j = 0; j < n / 16; j++) {
            sum += matrix[(j + ((7 * n) / 8)) * row + index] * vector_shared[j]; // multiply and accumulate with row-major ordering
        }

        x = 0;
        for (int i = (15 * n) / 16; i < n; i++) {
            //matrix_local[x] = matrix[(i)*row + index];
            vector_shared[x] = vector[i];
            x++;
        }



        for (int j = 0; j < n / 16; j++) {
            sum += matrix[(j + ((15 * n) / 16)) * row + index] * vector_shared[j]; // multiply and accumulate with row-major ordering
        }


        // fake work below this line 


        x = 0;
        for (int i = (7 * n) / 8; i < (15 * n) / 16; i++) {
            //matrix_local[x] = matrix[(i)*row + index];
            vector_shared[x] = vector[i];
            x++;
        }

        for (int j = 0; j < n / 16; j++) {
            sum2 = matrix[(j + ((7 * n) / 8)) * row + index] * vector_shared[j]; // multiply and accumulate with row-major ordering
        }

        x = 0;
        for (int i = (15 * n) / 16; i < n; i++) {
            //matrix_local[x] = matrix[(i)*row + index];
            vector_shared[x] = vector[i];
            x++;
        }

        for (int j = 0; j < n / 16; j++) {
            sum2 = matrix[(j + ((15 * n) / 16)) * row + index] * vector_shared[j]; // multiply and accumulate with row-major ordering
        }

        x = 0;
        for (int i = (7 * n) / 8; i < (15 * n) / 16; i++) {
            //matrix_local[x] = matrix[(i)*row + index];
            vector_shared[x] = vector[i];
            x++;
        }

        for (int j = 0; j < n / 16; j++) {
            sum2 = matrix[(j + ((7 * n) / 8)) * row + index] * vector_shared[j]; // multiply and accumulate with row-major ordering
        }

        x = 0;
        for (int i = (15 * n) / 16; i < n; i++) {
            //matrix_local[x] = matrix[(i)*row + index];
            vector_shared[x] = vector[i];
            x++;
        }

        for (int j = 0; j < n / 16; j++) {
            sum2 = matrix[(j + ((15 * n) / 16)) * row + index] * vector_shared[j]; // multiply and accumulate with row-major ordering
        }

        /*
        x = 0;
        for (int i = (15 * n) / 16; i < n; i++) {
            //matrix_local[x] = matrix[(i)*row + index];
            vector_shared[x] = vector[i];
            x++;
        }

        for (int j = 0; j < n / 16; j++) {
            sum2 = matrix[(j + ((15 * n) / 16)) * row + index] * vector_shared[j]; // multiply and accumulate with row-major ordering
        }

        x = 0;
        for (int i = (7 * n) / 8; i < (15 * n) / 16; i++) {
            //matrix_local[x] = matrix[(i)*row + index];
            vector_shared[x] = vector[i];
            x++;
        }

        for (int j = 0; j < n / 16; j++) {
            sum2 = matrix[(j + ((7 * n) / 8)) * row + index] * vector_shared[j]; // multiply and accumulate with row-major ordering
        }

        x = 0;
        for (int i = (15 * n) / 16; i < n; i++) {
            //matrix_local[x] = matrix[(i)*row + index];
            vector_shared[x] = vector[i];
            x++;
        }

        for (int j = 0; j < n / 16; j++) {
            sum2 = matrix[(j + ((15 * n) / 16)) * row + index] * vector_shared[j]; // multiply and accumulate with row-major ordering
        }
        */


        //result[index] = sum;
    }
    result[index] = sum;
}
<<<<<<< HEAD
=======

>>>>>>> 703ddcc94b6e9261f375913e88ead759d74a3321
