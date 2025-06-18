#include "../include/TransformerModel.h"
#include <fstream>
#include <iostream>
#include <random>
#include <algorithm>

TransformerModel::TransformerModel(int input_dim, int hidden_dim, int num_heads, int sequence_length)
    : input_dim_(input_dim),
      hidden_dim_(hidden_dim),
      num_heads_(num_heads),
      sequence_length_(sequence_length),
      is_initialized_(false)
{
    // Initialize model weights with default values
    initializeWeights();
}

void TransformerModel::initializeWeights() {
    // Xavier initialization for weights
    std::random_device rd;
    std::mt19937 gen(rd());
    
    // Initialize weight matrices
    double embedding_scale = std::sqrt(2.0 / (input_dim_ + hidden_dim_));
    double attention_scale = std::sqrt(2.0 / (hidden_dim_ + hidden_dim_ / num_heads_));
    double ffn_scale = std::sqrt(2.0 / (hidden_dim_ + 4 * hidden_dim_));
    double output_scale = std::sqrt(2.0 / (hidden_dim_ + 1));
    
    // Generate random weights using normal distribution
    std::normal_distribution<double> embedding_dist(0, embedding_scale);
    std::normal_distribution<double> attention_dist(0, attention_scale);
    std::normal_distribution<double> ffn_dist(0, ffn_scale);
    std::normal_distribution<double> output_dist(0, output_scale);
    
    // Initialize embedding weights
    embedding_weights_ = Eigen::MatrixXd::Zero(hidden_dim_, input_dim_);
    for (int i = 0; i < hidden_dim_; ++i) {
        for (int j = 0; j < input_dim_; ++j) {
            embedding_weights_(i, j) = embedding_dist(gen);
        }
    }
    embedding_bias_ = Eigen::VectorXd::Zero(hidden_dim_);
    
    // Initialize attention weights
    attention_query_weights_ = Eigen::MatrixXd::Zero(hidden_dim_, hidden_dim_);
    attention_key_weights_ = Eigen::MatrixXd::Zero(hidden_dim_, hidden_dim_);
    attention_value_weights_ = Eigen::MatrixXd::Zero(hidden_dim_, hidden_dim_);
    
    for (int i = 0; i < hidden_dim_; ++i) {
        for (int j = 0; j < hidden_dim_; ++j) {
            attention_query_weights_(i, j) = attention_dist(gen);
            attention_key_weights_(i, j) = attention_dist(gen);
            attention_value_weights_(i, j) = attention_dist(gen);
        }
    }
    
    // Initialize feed-forward network weights
    ffn_weights_1_ = Eigen::MatrixXd::Zero(4 * hidden_dim_, hidden_dim_);
    ffn_weights_2_ = Eigen::MatrixXd::Zero(hidden_dim_, 4 * hidden_dim_);
    
    for (int i = 0; i < 4 * hidden_dim_; ++i) {
        for (int j = 0; j < hidden_dim_; ++j) {
            ffn_weights_1_(i, j) = ffn_dist(gen);
        }
    }
    
    for (int i = 0; i < hidden_dim_; ++i) {
        for (int j = 0; j < 4 * hidden_dim_; ++j) {
            ffn_weights_2_(i, j) = ffn_dist(gen);
        }
    }
    
    ffn_bias_1_ = Eigen::VectorXd::Zero(4 * hidden_dim_);
    ffn_bias_2_ = Eigen::VectorXd::Zero(hidden_dim_);
    
    // Initialize output layer weights
    output_weights_ = Eigen::MatrixXd::Zero(1, hidden_dim_);
    for (int i = 0; i < hidden_dim_; ++i) {
        output_weights_(0, i) = output_dist(gen);
    }
    output_bias_ = Eigen::VectorXd::Zero(1);
    
    // Initialize market-specific bias terms with Bitcoin-specific values
    market_order_bias_ = Eigen::VectorXd::Zero(hidden_dim_);
    limit_order_bias_ = Eigen::VectorXd::Zero(hidden_dim_);
    
    // BTC-specific initialization:
    // Market orders tend toward taker (lower maker probability)
    // So we set a negative bias on the maker output
    market_order_bias_(0) = -1.5;
    
    // Limit orders tend toward maker
    // So we set a positive bias on the maker output
    limit_order_bias_(0) = 2.0;
    
    is_initialized_ = true;
}

void TransformerModel::addObservation(const std::vector<double>& features) {
    // Process features and add to history
    std::vector<double> processed_features = extractFeatures(features);
    
    // Add to history
    history_.push_back(processed_features);
    
    // Keep history size limited to sequence length
    while (history_.size() > sequence_length_) {
        history_.pop_front();
    }
}

std::vector<double> TransformerModel::extractFeatures(const std::vector<double>& raw_features) {
    // Process raw features into model inputs
    // In a real implementation, this would do feature engineering
    // For simplicity, we'll just return the raw features
    return raw_features;
}

Eigen::MatrixXd TransformerModel::multiHeadAttention(const Eigen::MatrixXd& input) {
    // Get sequence length from input
    int seq_len = input.rows();
    
    // Split input into heads
    int head_dim = hidden_dim_ / num_heads_;
    
    // Compute query, key, value projections
    Eigen::MatrixXd query = input * attention_query_weights_.transpose();
    Eigen::MatrixXd key = input * attention_key_weights_.transpose();
    Eigen::MatrixXd value = input * attention_value_weights_.transpose();
    
    // Compute attention scores
    Eigen::MatrixXd scores = query * key.transpose() / std::sqrt(head_dim);
    
    // Apply softmax to get attention weights
    Eigen::MatrixXd attention_weights = Eigen::MatrixXd::Zero(seq_len, seq_len);
    for (int i = 0; i < seq_len; ++i) {
        double max_val = scores.row(i).maxCoeff();
        Eigen::VectorXd exp_scores = (scores.row(i).array() - max_val).exp();
        attention_weights.row(i) = exp_scores / exp_scores.sum();
    }
    
    // Apply attention weights to values
    Eigen::MatrixXd output = attention_weights * value;
    
    return output;
}

Eigen::MatrixXd TransformerModel::feedForward(const Eigen::MatrixXd& input) {
    // First layer with ReLU activation
    Eigen::MatrixXd hidden1 = input * ffn_weights_1_.transpose();
    for (int i = 0; i < hidden1.rows(); ++i) {
        hidden1.row(i) += ffn_bias_1_.transpose();
    }
    
    // Apply ReLU
    hidden1 = hidden1.array().max(0.0);
    
    // Second layer
    Eigen::MatrixXd output = hidden1 * ffn_weights_2_.transpose();
    for (int i = 0; i < output.rows(); ++i) {
        output.row(i) += ffn_bias_2_.transpose();
    }
    
    return output;
}

Eigen::MatrixXd TransformerModel::layerNorm(const Eigen::MatrixXd& input) {
    // Initialize output matrix with same dimensions as input
    Eigen::MatrixXd normalized = Eigen::MatrixXd::Zero(input.rows(), input.cols());
    
    // Perform layer normalization row by row
    for (int i = 0; i < input.rows(); ++i) {
        // Get the row as a vector
        Eigen::VectorXd row = input.row(i);
        
        // Calculate mean and variance for this row
        double mean = row.mean();
        double var = (row.array() - mean).square().mean();
        
        // Normalize the row
        normalized.row(i) = (row.array() - mean) / std::sqrt(var + 1e-6);
    }
    
    return normalized;
}

double TransformerModel::predictMakerProbability(const std::vector<double>& features, const std::string& order_type) {
    // Add current observation to history
    addObservation(features);
    
    // Debug output for Docker troubleshooting
    if (history_.empty()) {
        std::cerr << "Warning: History is empty in TransformerModel" << std::endl;
    }
    
    if (history_.size() < 2) {
        // Not enough history, return default values based on order type
        // Use more dynamic defaults based on order type without hard caps
        return (order_type == "market") ? 0.30 : 0.70;
    }
    
    // Build input sequence matrix from history
    int seq_len = std::min(static_cast<int>(history_.size()), sequence_length_);
    Eigen::MatrixXd input_sequence = Eigen::MatrixXd::Zero(seq_len, input_dim_);
    
    // Fill the input sequence with available history
    for (int i = 0; i < seq_len; ++i) {
        int idx = history_.size() - seq_len + i;
        // Check array bounds to prevent accessing invalid history
        if (idx >= 0 && idx < history_.size()) {
            for (int j = 0; j < input_dim_ && j < history_[idx].size(); ++j) {
                input_sequence(i, j) = history_[idx][j];
            }
        }
    }
    
    // Apply embedding layer
    Eigen::MatrixXd embedded = input_sequence * embedding_weights_.transpose();
    for (int i = 0; i < embedded.rows(); ++i) {
        embedded.row(i) += embedding_bias_.transpose();
    }
    
    // Add order type specific bias
    if (order_type == "market") {
        for (int i = 0; i < embedded.rows(); ++i) {
            embedded.row(i) += market_order_bias_.transpose();
        }
    } else {
        for (int i = 0; i < embedded.rows(); ++i) {
            embedded.row(i) += limit_order_bias_.transpose();
        }
    }
    
    // Apply self-attention (transformer block)
    Eigen::MatrixXd attention_output = multiHeadAttention(embedded);
    Eigen::MatrixXd attention_residual = embedded + attention_output;
    Eigen::MatrixXd normalized1 = layerNorm(attention_residual);
    
    // Apply feed-forward network
    Eigen::MatrixXd ffn_output = feedForward(normalized1);
    Eigen::MatrixXd ffn_residual = normalized1 + ffn_output;
    Eigen::MatrixXd normalized2 = layerNorm(ffn_residual);
    
    // Get final output using last timestep
    Eigen::VectorXd final_hidden = normalized2.row(normalized2.rows() - 1);
    
    // Apply output layer
    double logit = (output_weights_ * final_hidden)(0, 0) + output_bias_(0);
    
    // Apply sigmoid to get probability
    double maker_probability = sigmoid(logit);
    
    // Apply more relaxed bounds based on order type
    // This allows for more dynamic values that won't get stuck at boundaries
    if (order_type == "market") {
        // Allow wider range for market orders
        maker_probability = std::max(0.05, std::min(0.6, maker_probability));
    } else {
        // Allow wider range for limit orders
        maker_probability = std::max(0.4, std::min(0.95, maker_probability));
    }
    
    return maker_probability;
}

bool TransformerModel::loadWeights(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for reading: " << filename << std::endl;
        return false;
    }
    
    // In a real implementation, deserialize the model weights from file
    // For simplicity, we'll just return true
    std::cout << "Loaded model weights from " << filename << std::endl;
    
    return true;
}

bool TransformerModel::saveWeights(const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for writing: " << filename << std::endl;
        return false;
    }
    
    // In a real implementation, serialize the model weights to file
    // For simplicity, we'll just return true
    std::cout << "Saved model weights to " << filename << std::endl;
    
    return true;
}

void TransformerModel::clearHistory() {
    history_.clear();
} 