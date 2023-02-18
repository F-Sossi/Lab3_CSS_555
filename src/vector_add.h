//---------------------------------------------------------------------------
// CUDA Vector add.h
// Author: Frank Sossi
// 
// File contains: 
// get_time() - returns the current time in milliseconds
// add_grid() - CUDA kernel that adds two vectors
// random_int() - returns a random integer between min and max
// write_data() - writes the execution time to a CSV file
// vector_add() - reference function that adds two vectors
// 
//---------------------------------------------------------------------------
#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <fstream>
#include <chrono>
#include <random>

//---------------------------------------------------------------------------
// Function for Naive Matrix Vector Multiplication
// Input: pointers to matrix, vector, and result vector, matrix dimensions
// Output: none
//---------------------------------------------------------------------------
template<typename T>
__global__
void gemv_kernel(T* A, T* x, T* y, int m, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m) {
        T sum = 0.0;
        for (int j = 0; j < n; j++) {
            sum += A[i * n + j] * x[j];
        }
        y[i] = sum;
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
