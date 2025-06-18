#include "../include/AdvancedTransformerModel.h"
#include <iostream>
#include <fstream>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <cstring>

AdvancedTransformerModel::AdvancedTransformerModel(
    ModelType model_type,
    ModelBackend backend,
    int input_dim, 
    int hidden_dim, 
    int num_heads, 
    int num_layers,
    int sequence_length,
    bool use_gpu)
    : model_type_(model_type),
      backend_(backend),
      input_dim_(input_dim),
      hidden_dim_(hidden_dim),
      num_heads_(num_heads),
      num_layers_(num_layers),
      sequence_length_(sequence_length),
      use_gpu_(use_gpu),
      is_initialized_(false),
      device_(torch::kCPU),
      d_input_buffer_(nullptr),
      d_hidden_buffer_(nullptr),
      d_output_buffer_(nullptr),
      d_attention_weights_(nullptr),
      d_layer_norm_gamma_(nullptr),
      d_layer_norm_beta_(nullptr)
{
    // Initialize device
    if (use_gpu_ && torch::cuda::is_available()) {
        device_ = torch::kCUDA;
        std::cout << "AdvancedTransformerModel: GPU acceleration enabled" << std::endl;
    } else {
        device_ = torch::kCPU;
        use_gpu_ = false;
        std::cout << "AdvancedTransformerModel: Using CPU inference" << std::endl;
    }
    
    // Initialize CUDA stream manager if using GPU
    if (use_gpu_) {
        cuda_stream_ = std::make_unique<CudaStreamManager>();
        CUBLAS_CHECK(cublasCreate(&cublas_handle_));
        CUBLAS_CHECK(cublasSetStream(cublas_handle_, cuda_stream_->get_stream()));
    }
}

AdvancedTransformerModel::~AdvancedTransformerModel() {
    freeGpuMemory();
    if (cublas_handle_) {
        cublasDestroy(cublas_handle_);
    }
}

bool AdvancedTransformerModel::initialize(const std::string& model_path) {
    try {
        // Allocate GPU memory if using GPU
        if (use_gpu_) {
            allocateGpuMemory();
        }
        
        // Load model if path provided
        if (!model_path.empty()) {
            if (!loadModel(model_path)) {
                std::cerr << "Failed to load model from: " << model_path << std::endl;
                return false;
            }
        } else {
            // Build default model architecture
            switch (model_type_) {
                case ModelType::TEMPORAL_FUSION_TRANSFORMER:
                    buildTemporalFusionTransformer();
                    break;
                case ModelType::VISION_TRANSFORMER:
                    buildVisionTransformer();
                    break;
                case ModelType::BERT_FINANCIAL:
                    buildBertFinancial();
                    break;
                case ModelType::LONGFORMER_MARKET:
                    buildLongformerMarket();
                    break;
                case ModelType::PERFORMER_EFFICIENT:
                    buildPerformerEfficient();
                    break;
            }
        }
        
        is_initialized_ = true;
        std::cout << "AdvancedTransformerModel initialized successfully" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Error initializing AdvancedTransformerModel: " << e.what() << std::endl;
        return false;
    }
}

bool AdvancedTransformerModel::loadModel(const std::string& model_path) {
    try {
        std::string extension = model_path.substr(model_path.find_last_of(".") + 1);
        
        if (extension == "pt" || extension == "pth") {
            // Load PyTorch model
            backend_ = ModelBackend::PYTORCH;
            pytorch_model_ = std::make_unique<torch::jit::script::Module>(
                torch::jit::load(model_path, device_));
            pytorch_model_->eval();
            
            std::cout << "Loaded PyTorch model: " << model_path << std::endl;
            return true;
            
        } else if (extension == "onnx") {
            // Load ONNX model
            backend_ = ModelBackend::ONNX;
            onnx_env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "AdvancedTransformer");
            
            Ort::SessionOptions session_options;
            if (use_gpu_) {
                // Enable CUDA provider for ONNX
                OrtCUDAProviderOptions cuda_options = {};
                session_options.AppendExecutionProvider_CUDA(cuda_options);
            }
            
            onnx_session_ = std::make_unique<Ort::Session>(*onnx_env_, model_path.c_str(), session_options);
            
            // Get input/output names
            Ort::AllocatorWithDefaultOptions allocator;
            size_t num_input_nodes = onnx_session_->GetInputCount();
            size_t num_output_nodes = onnx_session_->GetOutputCount();
            
            for (size_t i = 0; i < num_input_nodes; i++) {
                auto input_name = onnx_session_->GetInputNameAllocated(i, allocator);
                input_names_.push_back(std::string(input_name.get()));
            }
            
            for (size_t i = 0; i < num_output_nodes; i++) {
                auto output_name = onnx_session_->GetOutputNameAllocated(i, allocator);
                output_names_.push_back(std::string(output_name.get()));
            }
            
            std::cout << "Loaded ONNX model: " << model_path << std::endl;
            return true;
            
        } else {
            std::cerr << "Unsupported model format: " << extension << std::endl;
            return false;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
        return false;
    }
}

bool AdvancedTransformerModel::saveModel(const std::string& model_path) {
    try {
        if (backend_ == ModelBackend::PYTORCH && pytorch_model_) {
            pytorch_model_->save(model_path);
            std::cout << "Saved PyTorch model: " << model_path << std::endl;
            return true;
        } else {
            std::cerr << "Model saving not supported for current backend" << std::endl;
            return false;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error saving model: " << e.what() << std::endl;
        return false;
    }
}

void AdvancedTransformerModel::addObservation(const std::vector<double>& raw_features, 
                                            int64_t timestamp,
                                            int market_regime) {
    // Extract advanced features
    auto processed_features = extractAdvancedFeatures(raw_features, timestamp, market_regime);
    
    // Add to history
    feature_history_.push_back(processed_features);
    timestamp_history_.push_back(timestamp);
    regime_history_.push_back(market_regime);
    
    // Maintain sequence length
    while (feature_history_.size() > static_cast<size_t>(sequence_length_)) {
        feature_history_.pop_front();
        timestamp_history_.pop_front();
        regime_history_.pop_front();
    }
}

AdvancedTransformerModel::PredictionResult 
AdvancedTransformerModel::predictMakerProbability(const std::vector<double>& features, 
                                                const std::string& order_type,
                                                double confidence_level) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    if (!is_initialized_) {
        std::cerr << "Model not initialized" << std::endl;
        return {0.5, 0.0, 1.0, 1.0, {}, 0.0};
    }
    
    // Add current observation
    addObservation(features, std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count());
    
    PredictionResult result;
    
    try {
        switch (backend_) {
            case ModelBackend::PYTORCH: {
                if (feature_history_.size() < 2) {
                    result.maker_probability = (order_type == "market") ? 0.25 : 0.75;
                    result.confidence_lower = result.maker_probability - 0.1;
                    result.confidence_upper = result.maker_probability + 0.1;
                    result.model_uncertainty = 0.5;
                    break;
                }
                
                // Prepare input tensor
                auto input_tensor = torch::zeros({1, static_cast<long>(feature_history_.size()), input_dim_}, 
                                               torch::TensorOptions().dtype(torch::kFloat32).device(device_));
                
                for (size_t i = 0; i < feature_history_.size(); ++i) {
                    for (int j = 0; j < input_dim_ && j < static_cast<int>(feature_history_[i].size()); ++j) {
                        input_tensor[0][i][j] = static_cast<float>(feature_history_[i][j]);
                    }
                }
                
                result = predictWithPyTorch(input_tensor, order_type);
                break;
            }
            
            case ModelBackend::ONNX: {
                // Convert to float vector
                std::vector<float> input_vector;
                for (const auto& obs : feature_history_) {
                    for (int i = 0; i < input_dim_ && i < static_cast<int>(obs.size()); ++i) {
                        input_vector.push_back(static_cast<float>(obs[i]));
                    }
                }
                
                result = predictWithONNX(input_vector, order_type);
                break;
            }
            
            case ModelBackend::CUDA_NATIVE: {
                // Convert to float vector
                std::vector<float> input_vector;
                for (const auto& obs : feature_history_) {
                    for (int i = 0; i < input_dim_ && i < static_cast<int>(obs.size()); ++i) {
                        input_vector.push_back(static_cast<float>(obs[i]));
                    }
                }
                
                result = predictWithCUDA(input_vector, order_type);
                break;
            }
        }
        
        // Calculate confidence intervals
        auto confidence_bounds = calculateConfidenceInterval(
            result.maker_probability, result.model_uncertainty, confidence_level);
        result.confidence_lower = confidence_bounds.first;
        result.confidence_upper = confidence_bounds.second;
        
    } catch (const std::exception& e) {
        std::cerr << "Error in prediction: " << e.what() << std::endl;
        result.maker_probability = (order_type == "market") ? 0.25 : 0.75;
        result.confidence_lower = 0.0;
        result.confidence_upper = 1.0;
        result.model_uncertainty = 1.0;
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    result.execution_time_ms = static_cast<double>(duration.count());
    
    // Track inference time
    inference_times_.push_back(result.execution_time_ms);
    if (inference_times_.size() > 1000) {
        inference_times_.erase(inference_times_.begin());
    }
    
    return result;
}

AdvancedTransformerModel::MultiPredictionResult 
AdvancedTransformerModel::predictMultipleOutputs(const std::vector<double>& features,
                                               const std::string& order_type) {
    MultiPredictionResult result;
    
    // Get maker probability prediction
    auto maker_pred = predictMakerProbability(features, order_type);
    result.maker_probability = maker_pred.maker_probability;
    
    // Predict other market quantities based on current features
    if (feature_history_.size() >= 2) {
        // Calculate volatility prediction based on recent price movements
        std::vector<double> recent_prices;
        for (size_t i = std::max(0, static_cast<int>(feature_history_.size()) - 20); 
             i < feature_history_.size(); ++i) {
            if (feature_history_[i].size() > 0) {
                recent_prices.push_back(feature_history_[i][0]); // Assuming first feature is price
            }
        }
        
        if (recent_prices.size() >= 2) {
            double price_volatility = 0.0;
            for (size_t i = 1; i < recent_prices.size(); ++i) {
                double return_val = (recent_prices[i] - recent_prices[i-1]) / recent_prices[i-1];
                price_volatility += return_val * return_val;
            }
            result.predicted_volatility = std::sqrt(price_volatility / (recent_prices.size() - 1));
        } else {
            result.predicted_volatility = 0.02; // Default 2% volatility
        }
        
        // Predict spread based on volatility and market conditions
        result.predicted_spread = result.predicted_volatility * 0.5; // Simple model
        
        // Market impact factor based on order size and liquidity
        if (features.size() > 1) {
            double order_size = features[1]; // Assuming second feature is order size
            result.market_impact_factor = std::min(1.0, order_size / 1000000.0); // Normalize by $1M
        } else {
            result.market_impact_factor = 0.1;
        }
        
        // Liquidity score (inverse of impact)
        result.liquidity_score = 1.0 - result.market_impact_factor;
        
        // Price movement probabilities (simplified 5-class classification)
        result.price_movement_probabilities = {0.1, 0.2, 0.4, 0.2, 0.1}; // Default distribution
        
        // Adjust based on market regime if available
        if (!regime_history_.empty()) {
            int current_regime = regime_history_.back();
            if (current_regime == 1) { // Bull market
                result.price_movement_probabilities = {0.05, 0.15, 0.3, 0.3, 0.2};
            } else if (current_regime == -1) { // Bear market
                result.price_movement_probabilities = {0.2, 0.3, 0.3, 0.15, 0.05};
            }
        }
    } else {
        // Default values when insufficient history
        result.predicted_volatility = 0.02;
        result.predicted_spread = 0.01;
        result.market_impact_factor = 0.1;
        result.liquidity_score = 0.9;
        result.price_movement_probabilities = {0.1, 0.2, 0.4, 0.2, 0.1};
    }
    
    return result;
}

std::vector<double> AdvancedTransformerModel::extractAdvancedFeatures(
    const std::vector<double>& raw_features,
    int64_t timestamp,
    int market_regime) {
    
    std::vector<double> advanced_features = raw_features;
    
    // Add timestamp-based features
    if (timestamp > 0) {
        // Hour of day (normalized 0-1)
        double hour_of_day = ((timestamp % 86400) / 3600.0) / 24.0;
        advanced_features.push_back(hour_of_day);
        
        // Day of week (normalized 0-1)
        double day_of_week = ((timestamp / 86400) % 7) / 7.0;
        advanced_features.push_back(day_of_week);
    }
    
    // Add market regime
    advanced_features.push_back(static_cast<double>(market_regime));
    
    // Calculate technical indicators if we have enough history
    if (feature_history_.size() >= 14) {
        std::deque<double> prices;
        for (const auto& obs : feature_history_) {
            if (!obs.empty()) {
                prices.push_back(obs[0]); // Assuming first feature is price
            }
        }
        
        if (prices.size() >= 14) {
            // RSI
            double rsi = calculateRSI(prices);
            advanced_features.push_back(rsi / 100.0); // Normalize to 0-1
            
            // MACD
            double macd = calculateMACD(prices);
            advanced_features.push_back(std::tanh(macd)); // Bounded activation
            
            // Bollinger Bands
            auto bb = calculateBollingerBands(prices);
            if (bb.size() >= 3) {
                double bb_position = (prices.back() - bb[1]) / (bb[2] - bb[0]); // Position in bands
                advanced_features.push_back(std::max(-1.0, std::min(1.0, bb_position)));
            }
        }
    }
    
    // Add market microstructure features
    advanced_features.push_back(calculateOrderFlowImbalance());
    advanced_features.push_back(calculateEffectiveSpread());
    advanced_features.push_back(calculatePriceImpactMetric());
    advanced_features.push_back(calculateMarketDepthRatio());
    
    // Ensure we don't exceed input dimension
    if (static_cast<int>(advanced_features.size()) > input_dim_) {
        advanced_features.resize(input_dim_);
    } else {
        // Pad with zeros if needed
        while (static_cast<int>(advanced_features.size()) < input_dim_) {
            advanced_features.push_back(0.0);
        }
    }
    
    return advanced_features;
}

// Technical indicator implementations
double AdvancedTransformerModel::calculateRSI(const std::deque<double>& prices, int period) {
    if (prices.size() < static_cast<size_t>(period + 1)) return 50.0;
    
    double avg_gain = 0.0, avg_loss = 0.0;
    
    // Calculate initial averages
    for (int i = 1; i <= period; ++i) {
        double change = prices[i] - prices[i-1];
        if (change > 0) avg_gain += change;
        else avg_loss += -change;
    }
    avg_gain /= period;
    avg_loss /= period;
    
    if (avg_loss == 0.0) return 100.0;
    
    double rs = avg_gain / avg_loss;
    return 100.0 - (100.0 / (1.0 + rs));
}

double AdvancedTransformerModel::calculateMACD(const std::deque<double>& prices) {
    if (prices.size() < 26) return 0.0;
    
    // Simple MACD calculation (12-period EMA - 26-period EMA)
    auto calculate_ema = [](const std::deque<double>& data, int period) {
        double multiplier = 2.0 / (period + 1);
        double ema = data[0];
        for (size_t i = 1; i < data.size(); ++i) {
            ema = (data[i] * multiplier) + (ema * (1 - multiplier));
        }
        return ema;
    };
    
    double ema12 = calculate_ema(prices, 12);
    double ema26 = calculate_ema(prices, 26);
    
    return ema12 - ema26;
}

std::vector<double> AdvancedTransformerModel::calculateBollingerBands(
    const std::deque<double>& prices, int period) {
    if (prices.size() < static_cast<size_t>(period)) return {};
    
    // Calculate SMA
    double sum = 0.0;
    for (int i = prices.size() - period; i < static_cast<int>(prices.size()); ++i) {
        sum += prices[i];
    }
    double sma = sum / period;
    
    // Calculate standard deviation
    double variance = 0.0;
    for (int i = prices.size() - period; i < static_cast<int>(prices.size()); ++i) {
        variance += (prices[i] - sma) * (prices[i] - sma);
    }
    double std_dev = std::sqrt(variance / period);
    
    return {sma - 2*std_dev, sma, sma + 2*std_dev}; // Lower, Middle, Upper bands
}

// Market microstructure feature implementations
double AdvancedTransformerModel::calculateOrderFlowImbalance() {
    // Simplified implementation - would need real order flow data
    if (feature_history_.size() < 2) return 0.0;
    
    // Use bid/ask volumes if available (assuming features 3, 4 are bid/ask volumes)
    const auto& current = feature_history_.back();
    if (current.size() >= 5) {
        double bid_vol = current[3];
        double ask_vol = current[4];
        double total_vol = bid_vol + ask_vol;
        if (total_vol > 0) {
            return (bid_vol - ask_vol) / total_vol;
        }
    }
    return 0.0;
}

double AdvancedTransformerModel::calculateEffectiveSpread() {
    // Simplified implementation
    if (feature_history_.size() < 1) return 0.01;
    
    const auto& current = feature_history_.back();
    if (current.size() >= 2) {
        return current[1]; // Assuming second feature is spread
    }
    return 0.01;
}

double AdvancedTransformerModel::calculatePriceImpactMetric() {
    // Calculate price impact based on recent price changes and volumes
    if (feature_history_.size() < 3) return 0.0;
    
    double total_impact = 0.0;
    for (size_t i = 1; i < std::min(feature_history_.size(), size_t(10)); ++i) {
        if (feature_history_[i].size() >= 2 && feature_history_[i-1].size() >= 2) {
            double price_change = std::abs(feature_history_[i][0] - feature_history_[i-1][0]);
            double volume = feature_history_[i].size() > 3 ? feature_history_[i][3] : 1.0;
            if (volume > 0) {
                total_impact += price_change / volume;
            }
        }
    }
    
    return total_impact / std::min(feature_history_.size() - 1, size_t(10));
}

double AdvancedTransformerModel::calculateMarketDepthRatio() {
    // Calculate ratio of available liquidity at different price levels
    if (feature_history_.size() < 1) return 0.5;
    
    const auto& current = feature_history_.back();
    if (current.size() >= 5) {
        double near_depth = current[3] + current[4]; // Bid + Ask volumes
        double far_depth = current.size() > 6 ? current[5] + current[6] : near_depth;
        if (far_depth > 0) {
            return near_depth / far_depth;
        }
    }
    return 0.5;
}

// GPU memory management implementations
void AdvancedTransformerModel::allocateGpuMemory() {
    if (!use_gpu_) return;
    
    try {
        size_t input_size = sequence_length_ * input_dim_;
        size_t hidden_size = sequence_length_ * hidden_dim_;
        size_t output_size = sequence_length_ * hidden_dim_;
        size_t attention_size = sequence_length_ * sequence_length_;
        
        d_input_buffer_ = CudaMemoryManager::allocate_device_memory(input_size);
        d_hidden_buffer_ = CudaMemoryManager::allocate_device_memory(hidden_size);
        d_output_buffer_ = CudaMemoryManager::allocate_device_memory(output_size);
        d_attention_weights_ = CudaMemoryManager::allocate_device_memory(attention_size);
        d_layer_norm_gamma_ = CudaMemoryManager::allocate_device_memory(hidden_dim_);
        d_layer_norm_beta_ = CudaMemoryManager::allocate_device_memory(hidden_dim_);
        
        std::cout << "GPU memory allocated successfully" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Failed to allocate GPU memory: " << e.what() << std::endl;
        freeGpuMemory();
        use_gpu_ = false;
    }
}

void AdvancedTransformerModel::freeGpuMemory() {
    if (d_input_buffer_) {
        CudaMemoryManager::free_device_memory(d_input_buffer_);
        d_input_buffer_ = nullptr;
    }
    if (d_hidden_buffer_) {
        CudaMemoryManager::free_device_memory(d_hidden_buffer_);
        d_hidden_buffer_ = nullptr;
    }
    if (d_output_buffer_) {
        CudaMemoryManager::free_device_memory(d_output_buffer_);
        d_output_buffer_ = nullptr;
    }
    if (d_attention_weights_) {
        CudaMemoryManager::free_device_memory(d_attention_weights_);
        d_attention_weights_ = nullptr;
    }
    if (d_layer_norm_gamma_) {
        CudaMemoryManager::free_device_memory(d_layer_norm_gamma_);
        d_layer_norm_gamma_ = nullptr;
    }
    if (d_layer_norm_beta_) {
        CudaMemoryManager::free_device_memory(d_layer_norm_beta_);
        d_layer_norm_beta_ = nullptr;
    }
}

// Model architecture builders (simplified implementations)
void AdvancedTransformerModel::buildTemporalFusionTransformer() {
    // This would build a TFT architecture - simplified for demonstration
    std::cout << "Building Temporal Fusion Transformer architecture" << std::endl;
    backend_ = ModelBackend::CUDA_NATIVE;
}

void AdvancedTransformerModel::buildVisionTransformer() {
    std::cout << "Building Vision Transformer architecture" << std::endl;
    backend_ = ModelBackend::CUDA_NATIVE;
}

void AdvancedTransformerModel::buildBertFinancial() {
    std::cout << "Building BERT Financial architecture" << std::endl;
    backend_ = ModelBackend::CUDA_NATIVE;
}

void AdvancedTransformerModel::buildLongformerMarket() {
    std::cout << "Building Longformer Market architecture" << std::endl;
    backend_ = ModelBackend::CUDA_NATIVE;
}

void AdvancedTransformerModel::buildPerformerEfficient() {
    std::cout << "Building Performer Efficient architecture" << std::endl;
    backend_ = ModelBackend::CUDA_NATIVE;
}

// Prediction implementations for different backends
AdvancedTransformerModel::PredictionResult 
AdvancedTransformerModel::predictWithPyTorch(const torch::Tensor& input_tensor, const std::string& order_type) {
    PredictionResult result;
    
    try {
        torch::NoGradGuard no_grad;
        
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input_tensor);
        
        auto output = pytorch_model_->forward(inputs).toTensor();
        
        // Apply sigmoid to get probability
        auto probabilities = torch::sigmoid(output);
        
        result.maker_probability = probabilities[0].item<float>();
        result.model_uncertainty = 0.1; // Would calculate actual uncertainty in real implementation
        
        // Extract attention weights if available
        // This would require model to return attention weights
        
    } catch (const std::exception& e) {
        std::cerr << "PyTorch prediction error: " << e.what() << std::endl;
        result.maker_probability = (order_type == "market") ? 0.25 : 0.75;
        result.model_uncertainty = 1.0;
    }
    
    return result;
}

AdvancedTransformerModel::PredictionResult 
AdvancedTransformerModel::predictWithONNX(const std::vector<float>& input_vector, const std::string& order_type) {
    PredictionResult result;
    
    try {
        // Prepare input tensor
        std::vector<int64_t> input_shape = {1, static_cast<int64_t>(feature_history_.size()), input_dim_};
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, const_cast<float*>(input_vector.data()), input_vector.size(),
            input_shape.data(), input_shape.size());
        
        // Run inference
        std::vector<const char*> input_names_cstr;
        for (const auto& name : input_names_) {
            input_names_cstr.push_back(name.c_str());
        }
        
        std::vector<const char*> output_names_cstr;
        for (const auto& name : output_names_) {
            output_names_cstr.push_back(name.c_str());
        }
        
        auto output_tensors = onnx_session_->Run(Ort::RunOptions{nullptr},
                                               input_names_cstr.data(), &input_tensor, 1,
                                               output_names_cstr.data(), output_names_cstr.size());
        
        // Extract prediction
        if (!output_tensors.empty()) {
            float* output_data = output_tensors[0].GetTensorMutableData<float>();
            result.maker_probability = static_cast<double>(output_data[0]);
            result.model_uncertainty = 0.1;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "ONNX prediction error: " << e.what() << std::endl;
        result.maker_probability = (order_type == "market") ? 0.25 : 0.75;
        result.model_uncertainty = 1.0;
    }
    
    return result;
}

AdvancedTransformerModel::PredictionResult 
AdvancedTransformerModel::predictWithCUDA(const std::vector<float>& input_vector, const std::string& order_type) {
    PredictionResult result;
    
    if (!use_gpu_ || !d_input_buffer_) {
        result.maker_probability = (order_type == "market") ? 0.25 : 0.75;
        result.model_uncertainty = 1.0;
        return result;
    }
    
    try {
        // Copy input to GPU
        copyToGpu(input_vector, d_input_buffer_, input_vector.size());
        
        // Run CUDA kernels for transformer operations
        int batch_size = 1;
        int seq_len = feature_history_.size();
        int head_dim = hidden_dim_ / num_heads_;
        float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        
        // Simplified transformer forward pass using CUDA kernels
        launch_attention_kernel(d_input_buffer_, d_input_buffer_, d_input_buffer_,
                               d_attention_weights_, d_hidden_buffer_,
                               batch_size, seq_len, head_dim, scale, cuda_stream_->get_stream());
        
        launch_layer_norm_kernel(d_hidden_buffer_, d_output_buffer_, d_layer_norm_gamma_, d_layer_norm_beta_,
                               batch_size, seq_len, hidden_dim_, 1e-6, cuda_stream_->get_stream());
        
        // Copy result back to host
        std::vector<float> output_data(seq_len * hidden_dim_);
        copyFromGpu(output_data.data(), d_output_buffer_, output_data.size());
        
        // Simple aggregation for final prediction
        float sum = 0.0f;
        for (float val : output_data) {
            sum += val;
        }
        float raw_prediction = sum / output_data.size();
        
        // Apply sigmoid
        result.maker_probability = 1.0 / (1.0 + std::exp(-raw_prediction));
        result.model_uncertainty = 0.1;
        
        cuda_stream_->synchronize();
        
    } catch (const std::exception& e) {
        std::cerr << "CUDA prediction error: " << e.what() << std::endl;
        result.maker_probability = (order_type == "market") ? 0.25 : 0.75;
        result.model_uncertainty = 1.0;
    }
    
    return result;
}

void AdvancedTransformerModel::copyToGpu(const std::vector<float>& host_data, float* device_ptr, size_t size) {
    CudaMemoryManager::copy_host_to_device(device_ptr, host_data.data(), size, cuda_stream_->get_stream());
}

void AdvancedTransformerModel::copyFromGpu(float* host_data, const float* device_ptr, size_t size) {
    CudaMemoryManager::copy_device_to_host(host_data, device_ptr, size, cuda_stream_->get_stream());
}

// Utility functions
std::pair<double, double> AdvancedTransformerModel::calculateConfidenceInterval(
    double prediction, double uncertainty, double confidence_level) {
    // Simple confidence interval calculation
    double z_score = 1.96; // For 95% confidence
    if (confidence_level == 0.99) z_score = 2.576;
    else if (confidence_level == 0.90) z_score = 1.645;
    
    double margin = z_score * uncertainty;
    return {std::max(0.0, prediction - margin), std::min(1.0, prediction + margin)};
}

AdvancedTransformerModel::ModelMetrics AdvancedTransformerModel::getModelMetrics() const {
    ModelMetrics metrics;
    
    // Calculate average inference time
    if (!inference_times_.empty()) {
        metrics.avg_inference_time_ms = std::accumulate(inference_times_.begin(), inference_times_.end(), 0.0) 
                                       / inference_times_.size();
    } else {
        metrics.avg_inference_time_ms = 0.0;
    }
    
    // GPU memory usage
    if (use_gpu_) {
        size_t free_mem, total_mem;
        cudaMemGetInfo(&free_mem, &total_mem);
        metrics.gpu_memory_usage_mb = static_cast<double>(total_mem - free_mem) / (1024 * 1024);
    } else {
        metrics.gpu_memory_usage_mb = 0.0;
    }
    
    // Placeholder values for accuracy metrics (would be calculated from validation data in real implementation)
    metrics.accuracy = 0.85;
    metrics.precision = 0.82;
    metrics.recall = 0.88;
    metrics.f1_score = 0.85;
    
    return metrics;
}

double AdvancedTransformerModel::getGpuMemoryUsage() const {
    if (!use_gpu_) return 0.0;
    
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    return static_cast<double>(total_mem - free_mem) / (1024 * 1024);
}

void AdvancedTransformerModel::clearHistory() {
    feature_history_.clear();
    timestamp_history_.clear();
    regime_history_.clear();
    inference_times_.clear();
}

std::string AdvancedTransformerModel::getModelInfo() const {
    std::string info = "AdvancedTransformerModel Configuration:\n";
    info += "- Model Type: ";
    switch (model_type_) {
        case ModelType::TEMPORAL_FUSION_TRANSFORMER: info += "Temporal Fusion Transformer"; break;
        case ModelType::VISION_TRANSFORMER: info += "Vision Transformer"; break;
        case ModelType::BERT_FINANCIAL: info += "BERT Financial"; break;
        case ModelType::LONGFORMER_MARKET: info += "Longformer Market"; break;
        case ModelType::PERFORMER_EFFICIENT: info += "Performer Efficient"; break;
    }
    info += "\n- Backend: ";
    switch (backend_) {
        case ModelBackend::PYTORCH: info += "PyTorch C++"; break;
        case ModelBackend::ONNX: info += "ONNX Runtime"; break;
        case ModelBackend::CUDA_NATIVE: info += "CUDA Native"; break;
    }
    info += "\n- Input Dimension: " + std::to_string(input_dim_);
    info += "\n- Hidden Dimension: " + std::to_string(hidden_dim_);
    info += "\n- Number of Heads: " + std::to_string(num_heads_);
    info += "\n- Number of Layers: " + std::to_string(num_layers_);
    info += "\n- Sequence Length: " + std::to_string(sequence_length_);
    info += "\n- GPU Acceleration: " + std::string(use_gpu_ ? "Enabled" : "Disabled");
    info += "\n- Initialized: " + std::string(is_initialized_ ? "Yes" : "No");
    
    return info;
} 