#pragma once

#include <vector>
#include <deque>
#include <string>
#include <memory>
#include <random>
#include <cmath>
#include <torch/torch.h>
#include <onnxruntime_cxx_api.h>
#include "../gpu/CudaTransformerKernels.h"

/**
 * @brief Advanced GPU-Accelerated Transformer Model for Market Prediction
 * 
 * This class implements a state-of-the-art transformer architecture with:
 * - GPU acceleration via CUDA kernels
 * - PyTorch C++ integration for model inference
 * - ONNX Runtime support for production models
 * - Multi-scale temporal attention
 * - Market-specific positional encoding
 * - Advanced financial feature engineering
 */
class AdvancedTransformerModel {
public:
    enum class ModelBackend {
        PYTORCH,      // PyTorch C++ 
        ONNX,         // ONNX Runtime
        CUDA_NATIVE   // Custom CUDA implementation
    };
    
    enum class ModelType {
        TEMPORAL_FUSION_TRANSFORMER,    // TFT for time series
        VISION_TRANSFORMER,             // ViT adapted for order book
        BERT_FINANCIAL,                 // BERT for financial sequence modeling
        LONGFORMER_MARKET,              // Longformer for long sequences
        PERFORMER_EFFICIENT             // Performer for efficient attention
    };
    
    /**
     * @brief Constructor
     * @param model_type Type of transformer architecture
     * @param backend ML inference backend
     * @param input_dim Number of input features
     * @param hidden_dim Size of hidden layers
     * @param num_heads Number of attention heads
     * @param num_layers Number of transformer layers
     * @param sequence_length Maximum sequence length
     * @param use_gpu Whether to use GPU acceleration
     */
    AdvancedTransformerModel(
        ModelType model_type = ModelType::TEMPORAL_FUSION_TRANSFORMER,
        ModelBackend backend = ModelBackend::PYTORCH,
        int input_dim = 16, 
        int hidden_dim = 256, 
        int num_heads = 16, 
        int num_layers = 12,
        int sequence_length = 128,
        bool use_gpu = true
    );
    
    /**
     * @brief Destructor
     */
    ~AdvancedTransformerModel();
    
    /**
     * @brief Initialize the model
     * @param model_path Path to pre-trained model file
     * @return Success status
     */
    bool initialize(const std::string& model_path = "");
    
    /**
     * @brief Load pre-trained model weights
     * @param model_path Path to model file (.pt, .onnx, or custom format)
     * @return Success status
     */
    bool loadModel(const std::string& model_path);
    
    /**
     * @brief Save current model weights
     * @param model_path Output path for model
     * @return Success status
     */
    bool saveModel(const std::string& model_path);
    
    /**
     * @brief Add market observation with advanced feature engineering
     * @param raw_features Raw market features
     * @param timestamp Unix timestamp
     * @param market_regime Current market regime (bull/bear/sideways)
     */
    void addObservation(const std::vector<double>& raw_features, 
                       int64_t timestamp = 0,
                       int market_regime = 0);
    
    /**
     * @brief Predict maker probability with confidence intervals
     * @param features Current market features
     * @param order_type Type of order ("market" or "limit")
     * @param confidence_level Confidence level for intervals (0.95 default)
     * @return Prediction result with confidence intervals
     */
    struct PredictionResult {
        double maker_probability;
        double confidence_lower;
        double confidence_upper;
        double model_uncertainty;
        std::vector<double> attention_weights;
        double execution_time_ms;
    };
    
    PredictionResult predictMakerProbability(const std::vector<double>& features, 
                                           const std::string& order_type,
                                           double confidence_level = 0.95);
    
    /**
     * @brief Predict multiple market quantities simultaneously
     * @param features Current market features
     * @return Multiple predictions (maker/taker, volatility, spread, etc.)
     */
    struct MultiPredictionResult {
        double maker_probability;
        double predicted_volatility;
        double predicted_spread;
        double market_impact_factor;
        double liquidity_score;
        std::vector<double> price_movement_probabilities;  // 5-class: strong_down, down, neutral, up, strong_up
    };
    
    MultiPredictionResult predictMultipleOutputs(const std::vector<double>& features,
                                               const std::string& order_type);
    
    /**
     * @brief Get model performance metrics
     */
    struct ModelMetrics {
        double accuracy;
        double precision;
        double recall;
        double f1_score;
        double avg_inference_time_ms;
        double gpu_memory_usage_mb;
    };
    
    ModelMetrics getModelMetrics() const;
    
    /**
     * @brief Fine-tune model on recent data
     * @param training_data Historical observations
     * @param labels Ground truth labels
     * @param epochs Number of training epochs
     * @return Training success status
     */
    bool fineTuneModel(const std::vector<std::vector<double>>& training_data,
                      const std::vector<double>& labels,
                      int epochs = 10);
    
    /**
     * @brief Enable/disable GPU acceleration
     * @param use_gpu GPU acceleration flag
     */
    void setGpuAcceleration(bool use_gpu);
    
    /**
     * @brief Get current GPU memory usage
     * @return Memory usage in MB
     */
    double getGpuMemoryUsage() const;
    
    /**
     * @brief Clear model history and reset state
     */
    void clearHistory();
    
    /**
     * @brief Get model information
     */
    std::string getModelInfo() const;

private:
    // Model configuration
    ModelType model_type_;
    ModelBackend backend_;
    int input_dim_;
    int hidden_dim_;
    int num_heads_;
    int num_layers_;
    int sequence_length_;
    bool use_gpu_;
    bool is_initialized_;
    
    // PyTorch components
    std::unique_ptr<torch::jit::script::Module> pytorch_model_;
    torch::Device device_;
    
    // ONNX Runtime components
    std::unique_ptr<Ort::Session> onnx_session_;
    std::unique_ptr<Ort::Env> onnx_env_;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    
    // CUDA components
    std::unique_ptr<CudaStreamManager> cuda_stream_;
    cublasHandle_t cublas_handle_;
    
    // GPU memory buffers
    float* d_input_buffer_;
    float* d_hidden_buffer_;
    float* d_output_buffer_;
    float* d_attention_weights_;
    float* d_layer_norm_gamma_;
    float* d_layer_norm_beta_;
    
    // Feature engineering and history
    std::deque<std::vector<double>> feature_history_;
    std::deque<int64_t> timestamp_history_;
    std::deque<int> regime_history_;
    
    // Performance tracking
    mutable std::vector<double> inference_times_;
    mutable ModelMetrics cached_metrics_;
    
    // Advanced feature engineering
    std::vector<double> extractAdvancedFeatures(const std::vector<double>& raw_features,
                                               int64_t timestamp,
                                               int market_regime);
    
    // Technical indicators
    double calculateRSI(const std::deque<double>& prices, int period = 14);
    double calculateMACD(const std::deque<double>& prices);
    std::vector<double> calculateBollingerBands(const std::deque<double>& prices, int period = 20);
    double calculateVWAP(const std::deque<double>& prices, const std::deque<double>& volumes);
    
    // Market microstructure features
    double calculateOrderFlowImbalance();
    double calculateEffectiveSpread();
    double calculatePriceImpactMetric();
    double calculateMarketDepthRatio();
    
    // Model-specific implementations
    PredictionResult predictWithPyTorch(const torch::Tensor& input_tensor, const std::string& order_type);
    PredictionResult predictWithONNX(const std::vector<float>& input_vector, const std::string& order_type);
    PredictionResult predictWithCUDA(const std::vector<float>& input_vector, const std::string& order_type);
    
    // GPU memory management
    void allocateGpuMemory();
    void freeGpuMemory();
    void copyToGpu(const std::vector<float>& host_data, float* device_ptr, size_t size);
    void copyFromGpu(float* host_data, const float* device_ptr, size_t size);
    
    // Model architecture builders
    void buildTemporalFusionTransformer();
    void buildVisionTransformer();
    void buildBertFinancial();
    void buildLongformerMarket();
    void buildPerformerEfficient();
    
    // Attention mechanisms
    torch::Tensor multiScaleAttention(const torch::Tensor& input, const std::vector<int>& scales);
    torch::Tensor sparseAttention(const torch::Tensor& input, double sparsity_ratio = 0.1);
    torch::Tensor linearAttention(const torch::Tensor& input);
    
    // Positional encoding
    torch::Tensor createMarketPositionalEncoding(int seq_len, int d_model, 
                                                 const std::vector<int64_t>& timestamps);
    
    // Uncertainty quantification
    double calculateModelUncertainty(const torch::Tensor& logits, int num_samples = 100);
    std::pair<double, double> calculateConfidenceInterval(double prediction, double uncertainty, 
                                                         double confidence_level);
};