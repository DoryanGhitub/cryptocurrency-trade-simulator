#include "CudaTransformerKernels.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>
#include <iostream>

// CUDA kernel for softmax operation
__global__ void softmax_kernel(float* input, float* output, int rows, int cols) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int row = tid / cols;
    int col = tid % cols;
    
    if (row < rows) {
        // Shared memory for reduction
        extern __shared__ float sdata[];
        
        // Find max value in the row for numerical stability
        float max_val = -INFINITY;
        for (int i = 0; i < cols; i++) {
            max_val = fmaxf(max_val, input[row * cols + i]);
        }
        
        // Compute exp(x - max) and sum
        float sum = 0.0f;
        for (int i = 0; i < cols; i++) {
            float exp_val = expf(input[row * cols + i] - max_val);
            if (i == col) {
                sdata[threadIdx.x] = exp_val;
            }
            sum += exp_val;
        }
        
        // Normalize
        if (col < cols) {
            output[row * cols + col] = sdata[threadIdx.x] / sum;
        }
    }
}

// CUDA kernel for layer normalization
__global__ void layer_norm_kernel(float* input, float* output, float* gamma, float* beta, 
                                  int batch_size, int seq_len, int hidden_dim, float eps) {
    int batch_idx = blockIdx.x;
    int seq_idx = blockIdx.y;
    int hidden_idx = threadIdx.x;
    
    if (batch_idx < batch_size && seq_idx < seq_len && hidden_idx < hidden_dim) {
        int base_idx = batch_idx * seq_len * hidden_dim + seq_idx * hidden_dim;
        
        // Calculate mean
        float mean = 0.0f;
        for (int i = 0; i < hidden_dim; i++) {
            mean += input[base_idx + i];
        }
        mean /= hidden_dim;
        
        // Calculate variance
        float var = 0.0f;
        for (int i = 0; i < hidden_dim; i++) {
            float diff = input[base_idx + i] - mean;
            var += diff * diff;
        }
        var /= hidden_dim;
        
        // Normalize
        float normalized = (input[base_idx + hidden_idx] - mean) / sqrtf(var + eps);
        output[base_idx + hidden_idx] = gamma[hidden_idx] * normalized + beta[hidden_idx];
    }
}

// CUDA kernel for ReLU activation
__global__ void relu_kernel(float* input, float* output, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        output[tid] = fmaxf(0.0f, input[tid]);
    }
}

// CUDA kernel for GELU activation (used in transformers)
__global__ void gelu_kernel(float* input, float* output, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        float x = input[tid];
        float cdf = 0.5f * (1.0f + tanhf(sqrtf(2.0f / M_PI) * (x + 0.044715f * x * x * x)));
        output[tid] = x * cdf;
    }
}

// CUDA kernel for scaled dot-product attention
__global__ void scaled_dot_product_attention_kernel(float* query, float* key, float* value,
                                                    float* attention_weights, float* output,
                                                    int batch_size, int seq_len, int head_dim,
                                                    float scale) {
    int batch_idx = blockIdx.x;
    int seq_i = blockIdx.y;
    int seq_j = threadIdx.x;
    
    if (batch_idx < batch_size && seq_i < seq_len && seq_j < seq_len) {
        // Compute Q * K^T
        float score = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            int q_idx = batch_idx * seq_len * head_dim + seq_i * head_dim + d;
            int k_idx = batch_idx * seq_len * head_dim + seq_j * head_dim + d;
            score += query[q_idx] * key[k_idx];
        }
        score *= scale;
        
        // Store attention weights
        int attn_idx = batch_idx * seq_len * seq_len + seq_i * seq_len + seq_j;
        attention_weights[attn_idx] = score;
        
        // Apply softmax (simplified version)
        __syncthreads();
        
        // Find max for numerical stability
        float max_score = -INFINITY;
        for (int k = 0; k < seq_len; k++) {
            int max_idx = batch_idx * seq_len * seq_len + seq_i * seq_len + k;
            max_score = fmaxf(max_score, attention_weights[max_idx]);
        }
        
        // Compute exp and sum
        float exp_score = expf(score - max_score);
        attention_weights[attn_idx] = exp_score;
        
        __syncthreads();
        
        float sum = 0.0f;
        for (int k = 0; k < seq_len; k++) {
            int sum_idx = batch_idx * seq_len * seq_len + seq_i * seq_len + k;
            sum += attention_weights[sum_idx];
        }
        
        // Normalize
        attention_weights[attn_idx] = exp_score / sum;
        
        // Compute attention * V
        __syncthreads();
        
        for (int d = 0; d < head_dim; d++) {
            float weighted_value = 0.0f;
            for (int k = 0; k < seq_len; k++) {
                int attn_w_idx = batch_idx * seq_len * seq_len + seq_i * seq_len + k;
                int v_idx = batch_idx * seq_len * head_dim + k * head_dim + d;
                weighted_value += attention_weights[attn_w_idx] * value[v_idx];
            }
            int out_idx = batch_idx * seq_len * head_dim + seq_i * head_dim + d;
            output[out_idx] = weighted_value;
        }
    }
}

// Host function to launch softmax kernel
void launch_softmax_kernel(float* d_input, float* d_output, int rows, int cols, cudaStream_t stream) {
    int threads_per_block = 256;
    int blocks = (rows * cols + threads_per_block - 1) / threads_per_block;
    
    size_t shared_mem_size = threads_per_block * sizeof(float);
    softmax_kernel<<<blocks, threads_per_block, shared_mem_size, stream>>>(
        d_input, d_output, rows, cols);
}

// Host function to launch layer normalization kernel
void launch_layer_norm_kernel(float* d_input, float* d_output, float* d_gamma, float* d_beta,
                              int batch_size, int seq_len, int hidden_dim, float eps, 
                              cudaStream_t stream) {
    dim3 blocks(batch_size, seq_len);
    dim3 threads(hidden_dim);
    
    layer_norm_kernel<<<blocks, threads, 0, stream>>>(
        d_input, d_output, d_gamma, d_beta, batch_size, seq_len, hidden_dim, eps);
}

// Host function to launch ReLU kernel
void launch_relu_kernel(float* d_input, float* d_output, int size, cudaStream_t stream) {
    int threads_per_block = 256;
    int blocks = (size + threads_per_block - 1) / threads_per_block;
    
    relu_kernel<<<blocks, threads_per_block, 0, stream>>>(d_input, d_output, size);
}

// Host function to launch GELU kernel
void launch_gelu_kernel(float* d_input, float* d_output, int size, cudaStream_t stream) {
    int threads_per_block = 256;
    int blocks = (size + threads_per_block - 1) / threads_per_block;
    
    gelu_kernel<<<blocks, threads_per_block, 0, stream>>>(d_input, d_output, size);
}

// Host function to launch scaled dot-product attention kernel
void launch_attention_kernel(float* d_query, float* d_key, float* d_value,
                             float* d_attention_weights, float* d_output,
                             int batch_size, int seq_len, int head_dim, float scale,
                             cudaStream_t stream) {
    dim3 blocks(batch_size, seq_len);
    dim3 threads(seq_len);
    
    scaled_dot_product_attention_kernel<<<blocks, threads, 0, stream>>>(
        d_query, d_key, d_value, d_attention_weights, d_output,
        batch_size, seq_len, head_dim, scale);
} 