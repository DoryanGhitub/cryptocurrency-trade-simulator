#include "CudaTransformerKernels.h"
#include <iostream>
#include <stdexcept>

// CudaMemoryManager implementation
float* CudaMemoryManager::allocate_device_memory(size_t size) {
    float* ptr;
    cudaError_t error = cudaMalloc(&ptr, size * sizeof(float));
    if (error != cudaSuccess) {
        throw std::runtime_error("Failed to allocate GPU memory: " + 
                                std::string(cudaGetErrorString(error)));
    }
    return ptr;
}

void CudaMemoryManager::free_device_memory(float* ptr) {
    if (ptr) {
        cudaFree(ptr);
    }
}

void CudaMemoryManager::copy_host_to_device(float* d_dst, const float* h_src, size_t size, cudaStream_t stream) {
    cudaError_t error;
    if (stream) {
        error = cudaMemcpyAsync(d_dst, h_src, size * sizeof(float), cudaMemcpyHostToDevice, stream);
    } else {
        error = cudaMemcpy(d_dst, h_src, size * sizeof(float), cudaMemcpyHostToDevice);
    }
    
    if (error != cudaSuccess) {
        throw std::runtime_error("Failed to copy from host to device: " + 
                                std::string(cudaGetErrorString(error)));
    }
}

void CudaMemoryManager::copy_device_to_host(float* h_dst, const float* d_src, size_t size, cudaStream_t stream) {
    cudaError_t error;
    if (stream) {
        error = cudaMemcpyAsync(h_dst, d_src, size * sizeof(float), cudaMemcpyDeviceToHost, stream);
    } else {
        error = cudaMemcpy(h_dst, d_src, size * sizeof(float), cudaMemcpyDeviceToHost);
    }
    
    if (error != cudaSuccess) {
        throw std::runtime_error("Failed to copy from device to host: " + 
                                std::string(cudaGetErrorString(error)));
    }
}

void CudaMemoryManager::copy_device_to_device(float* d_dst, const float* d_src, size_t size, cudaStream_t stream) {
    cudaError_t error;
    if (stream) {
        error = cudaMemcpyAsync(d_dst, d_src, size * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    } else {
        error = cudaMemcpy(d_dst, d_src, size * sizeof(float), cudaMemcpyDeviceToDevice);
    }
    
    if (error != cudaSuccess) {
        throw std::runtime_error("Failed to copy from device to device: " + 
                                std::string(cudaGetErrorString(error)));
    }
}

cudaError_t CudaMemoryManager::check_cuda_error(const char* operation) {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error after " << operation << ": " 
                  << cudaGetErrorString(error) << std::endl;
    }
    return error;
}

// CudaStreamManager implementation
CudaStreamManager::CudaStreamManager() {
    CUDA_CHECK(cudaStreamCreate(&stream_));
}

CudaStreamManager::~CudaStreamManager() {
    if (stream_) {
        cudaStreamDestroy(stream_);
    }
}

void CudaStreamManager::synchronize() {
    CUDA_CHECK(cudaStreamSynchronize(stream_));
} 