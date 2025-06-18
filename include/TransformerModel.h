#pragma once

#include <vector>
#include <deque>
#include <string>
#include <memory>
#include <random>
#include <cmath>
#include <Eigen/Dense>

/**
 * @brief Transformer-based model for maker/taker proportion prediction
 * 
 * Implements a lightweight Temporal Fusion Transformer (TFT) architecture
 * optimized for order execution type prediction in cryptocurrency markets.
 */
class TransformerModel {
public:
    /**
     * @brief Constructor
     * @param input_dim Number of input features
     * @param hidden_dim Size of hidden layers
     * @param num_heads Number of attention heads
     * @param sequence_length Length of input sequence
     */
    TransformerModel(int input_dim = 8, 
                    int hidden_dim = 32, 
                    int num_heads = 4, 
                    int sequence_length = 24);
    
    /**
     * @brief Destructor
     */
    ~TransformerModel() = default;
    
    /**
     * @brief Add market observation to history
     * @param features Vector of market features
     */
    void addObservation(const std::vector<double>& features);
    
    /**
     * @brief Predict maker probability
     * @param features Current market features
     * @param order_type Type of order ("market" or "limit")
     * @return Probability of order being a maker (0.0-1.0)
     */
    double predictMakerProbability(const std::vector<double>& features, const std::string& order_type);
    
    /**
     * @brief Load pretrained model weights
     * @param filename Path to model weights file
     * @return Whether loading was successful
     */
    bool loadWeights(const std::string& filename);
    
    /**
     * @brief Save model weights
     * @param filename Path to save model weights
     * @return Whether saving was successful
     */
    bool saveWeights(const std::string& filename);
    
    /**
     * @brief Clear history
     */
    void clearHistory();

private:
    // Model parameters
    int input_dim_;
    int hidden_dim_;
    int num_heads_;
    int sequence_length_;
    bool is_initialized_;
    
    // History of market observations
    std::deque<std::vector<double>> history_;
    
    // Model weights
    Eigen::MatrixXd embedding_weights_;
    Eigen::MatrixXd attention_query_weights_;
    Eigen::MatrixXd attention_key_weights_;
    Eigen::MatrixXd attention_value_weights_;
    Eigen::MatrixXd ffn_weights_1_;
    Eigen::MatrixXd ffn_weights_2_;
    Eigen::MatrixXd output_weights_;
    
    // Bias terms
    Eigen::VectorXd embedding_bias_;
    Eigen::VectorXd ffn_bias_1_;
    Eigen::VectorXd ffn_bias_2_;
    Eigen::VectorXd output_bias_;
    
    // Market-specific parameters
    Eigen::VectorXd market_order_bias_;
    Eigen::VectorXd limit_order_bias_;
    
    /**
     * @brief Initialize model weights
     */
    void initializeWeights();
    
    /**
     * @brief Apply multi-head attention
     * @param input Input matrix
     * @return Output after attention
     */
    Eigen::MatrixXd multiHeadAttention(const Eigen::MatrixXd& input);
    
    /**
     * @brief Apply feed-forward network
     * @param input Input matrix
     * @return Output after FFN
     */
    Eigen::MatrixXd feedForward(const Eigen::MatrixXd& input);
    
    /**
     * @brief Apply layer normalization
     * @param input Input matrix
     * @return Normalized output
     */
    Eigen::MatrixXd layerNorm(const Eigen::MatrixXd& input);
    
    /**
     * @brief Extract features from raw market data
     * @param raw_features Raw market features
     * @return Processed features
     */
    std::vector<double> extractFeatures(const std::vector<double>& raw_features);
    
    /**
     * @brief Apply sigmoid activation
     * @param x Input value
     * @return Sigmoid output
     */
    double sigmoid(double x) const {
        return 1.0 / (1.0 + std::exp(-x));
    }
}; 