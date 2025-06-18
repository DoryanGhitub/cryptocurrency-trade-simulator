#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>

// CUDA kernel launch functions
void launch_softmax_kernel(float* d_input, float* d_output, int rows, int cols, cudaStream_t stream = 0);

void launch_layer_norm_kernel(float* d_input, float* d_output, float* d_gamma, float* d_beta,
                              int batch_size, int seq_len, int hidden_dim, float eps = 1e-6, 
                              cudaStream_t stream = 0);

void launch_relu_kernel(float* d_input, float* d_output, int size, cudaStream_t stream = 0);

void launch_gelu_kernel(float* d_input, float* d_output, int size, cudaStream_t stream = 0);

void launch_attention_kernel(float* d_query, float* d_key, float* d_value,
                             float* d_attention_weights, float* d_output,
                             int batch_size, int seq_len, int head_dim, float scale,
                             cudaStream_t stream = 0);

// GPU memory management utilities
class CudaMemoryManager {
public:
    static float* allocate_device_memory(size_t size);
    static void free_device_memory(float* ptr);
    static void copy_host_to_device(float* d_dst, const float* h_src, size_t size, cudaStream_t stream = 0);
    static void copy_device_to_host(float* h_dst, const float* d_src, size_t size, cudaStream_t stream = 0);
    static void copy_device_to_device(float* d_dst, const float* d_src, size_t size, cudaStream_t stream = 0);
    static cudaError_t check_cuda_error(const char* operation);
};

// CUDA stream management
class CudaStreamManager {
public:
    CudaStreamManager();
    ~CudaStreamManager();
    
    cudaStream_t get_stream() const { return stream_; }
    void synchronize();
    
private:
    cudaStream_t stream_;
};

// Error checking macros
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(error) << std::endl; \
            exit(1); \
        } \
    } while(0)

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            std::cerr << "CUBLAS error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << status << std::endl; \
            exit(1); \
        } \
    } while(0) 