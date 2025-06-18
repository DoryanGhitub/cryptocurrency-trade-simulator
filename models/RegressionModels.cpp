#include "RegressionModels.h"
#include <stdexcept>
#include <cmath>
#include <random>
#include <iostream>

// LinearRegression implementation
void LinearRegression::fit(const std::vector<std::vector<double>>& X, const std::vector<double>& y) {
    if (X.empty() || y.empty() || X.size() != y.size()) {
        throw std::invalid_argument("Invalid input dimensions");
    }
    
    int n_samples = X.size();
    int n_features = X[0].size();
    
    // Create Eigen matrices
    Eigen::MatrixXd X_eigen(n_samples, n_features + 1);
    Eigen::VectorXd y_eigen(n_samples);
    
    // Fill matrices
    for (int i = 0; i < n_samples; ++i) {
        X_eigen(i, 0) = 1.0;  // Intercept term
        for (int j = 0; j < n_features; ++j) {
            X_eigen(i, j + 1) = X[i][j];
        }
        y_eigen(i) = y[i];
    }
    
    // Solve the normal equation: β = (X^T X)^(-1) X^T y
    Eigen::VectorXd beta = (X_eigen.transpose() * X_eigen).ldlt().solve(X_eigen.transpose() * y_eigen);
    
    // Extract coefficients
    intercept_ = beta(0);
    coefficients_.resize(n_features);
    for (int j = 0; j < n_features; ++j) {
        coefficients_[j] = beta(j + 1);
    }
}

std::vector<double> LinearRegression::predict(const std::vector<std::vector<double>>& X) const {
    if (X.empty() || X[0].size() != coefficients_.size()) {
        throw std::invalid_argument("Invalid input dimensions");
    }
    
    int n_samples = X.size();
    std::vector<double> predictions(n_samples);
    
    for (int i = 0; i < n_samples; ++i) {
        predictions[i] = predict(X[i]);
    }
    
    return predictions;
}

double LinearRegression::predict(const std::vector<double>& x) const {
    if (x.size() != coefficients_.size()) {
        throw std::invalid_argument("Invalid input dimensions");
    }
    
    double prediction = intercept_;
    for (size_t j = 0; j < coefficients_.size(); ++j) {
        prediction += coefficients_[j] * x[j];
    }
    
    return prediction;
}

std::vector<double> LinearRegression::getCoefficients() const {
    return coefficients_;
}

double LinearRegression::getIntercept() const {
    return intercept_;
}

// QuantileRegression implementation
QuantileRegression::QuantileRegression(double quantile)
    : quantile_(quantile)
{
    if (quantile <= 0.0 || quantile >= 1.0) {
        throw std::invalid_argument("Quantile must be between 0 and 1");
    }
}

void QuantileRegression::fit(const std::vector<std::vector<double>>& X, const std::vector<double>& y,
                        int max_iterations, double tolerance) {
    if (X.empty() || y.empty() || X.size() != y.size()) {
        throw std::invalid_argument("Invalid input dimensions");
    }
    
    int n_samples = X.size();
    int n_features = X[0].size();
    
    // Initialize coefficients and intercept
    coefficients_.resize(n_features, 0.0);
    intercept_ = 0.0;
    
    // Create design matrix with intercept
    std::vector<std::vector<double>> X_design(n_samples, std::vector<double>(n_features + 1));
    for (int i = 0; i < n_samples; ++i) {
        X_design[i][0] = 1.0;  // Intercept
        for (int j = 0; j < n_features; ++j) {
            X_design[i][j + 1] = X[i][j];
        }
    }
    
    // Initialize parameters
    std::vector<double> params(n_features + 1, 0.0);
    std::vector<double> prev_params(n_features + 1, 0.0);
    
    // Learning rate
    double learning_rate = 0.01;
    
    // Gradient descent
    for (int iter = 0; iter < max_iterations; ++iter) {
        // Save previous parameters
        prev_params = params;
        
        // Compute predictions
        std::vector<double> predictions(n_samples);
        for (int i = 0; i < n_samples; ++i) {
            double pred = 0.0;
            for (int j = 0; j < n_features + 1; ++j) {
                pred += params[j] * X_design[i][j];
            }
            predictions[i] = pred;
        }
        
        // Compute gradients
        std::vector<double> gradients(n_features + 1, 0.0);
        for (int i = 0; i < n_samples; ++i) {
            double error = y[i] - predictions[i];
            double gradient_factor = (error < 0) ? (quantile_ - 1.0) : quantile_;
            
            for (int j = 0; j < n_features + 1; ++j) {
                gradients[j] += gradient_factor * X_design[i][j];
            }
        }
        
        // Update parameters
        for (int j = 0; j < n_features + 1; ++j) {
            params[j] += learning_rate * gradients[j] / n_samples;
        }
        
        // Check convergence
        double param_diff = 0.0;
        for (int j = 0; j < n_features + 1; ++j) {
            param_diff += std::abs(params[j] - prev_params[j]);
        }
        
        if (param_diff < tolerance) {
            break;
        }
    }
    
    // Extract intercept and coefficients
    intercept_ = params[0];
    for (int j = 0; j < n_features; ++j) {
        coefficients_[j] = params[j + 1];
    }
}

double QuantileRegression::quantileLoss(const std::vector<double>& y_true, const std::vector<double>& y_pred) const {
    if (y_true.size() != y_pred.size()) {
        throw std::invalid_argument("Dimensions mismatch");
    }
    
    double loss = 0.0;
    for (size_t i = 0; i < y_true.size(); ++i) {
        double error = y_true[i] - y_pred[i];
        loss += (error >= 0) ? quantile_ * error : (quantile_ - 1.0) * error;
    }
    
    return loss / y_true.size();
}

std::vector<double> QuantileRegression::predict(const std::vector<std::vector<double>>& X) const {
    if (X.empty() || X[0].size() != coefficients_.size()) {
        throw std::invalid_argument("Invalid input dimensions");
    }
    
    int n_samples = X.size();
    std::vector<double> predictions(n_samples);
    
    for (int i = 0; i < n_samples; ++i) {
        predictions[i] = predict(X[i]);
    }
    
    return predictions;
}

double QuantileRegression::predict(const std::vector<double>& x) const {
    if (x.size() != coefficients_.size()) {
        throw std::invalid_argument("Invalid input dimensions");
    }
    
    double prediction = intercept_;
    for (size_t j = 0; j < coefficients_.size(); ++j) {
        prediction += coefficients_[j] * x[j];
    }
    
    return prediction;
}

std::vector<double> QuantileRegression::getCoefficients() const {
    return coefficients_;
}

double QuantileRegression::getIntercept() const {
    return intercept_;
}

void QuantileRegression::setQuantile(double quantile) {
    if (quantile <= 0.0 || quantile >= 1.0) {
        throw std::invalid_argument("Quantile must be between 0 and 1");
    }
    quantile_ = quantile;
}

double QuantileRegression::getQuantile() const {
    return quantile_;
}

// LogisticRegression implementation
LogisticRegression::LogisticRegression(double learning_rate, int max_iterations)
    : learning_rate_(learning_rate),
      max_iterations_(max_iterations)
{
}

double LogisticRegression::sigmoid(double z) {
    return 1.0 / (1.0 + std::exp(-z));
}

void LogisticRegression::fit(const std::vector<std::vector<double>>& X, const std::vector<int>& y) {
    if (X.empty() || y.empty() || X.size() != y.size()) {
        throw std::invalid_argument("Invalid input dimensions");
    }
    
    int n_samples = X.size();
    int n_features = X[0].size();
    
    // Initialize coefficients and intercept
    coefficients_ = std::vector<double>(n_features, 0.0);
    intercept_ = 0.0;
    
    // Gradient descent
    for (int iter = 0; iter < max_iterations_; ++iter) {
        // Compute predictions
        std::vector<double> predictions(n_samples);
        for (int i = 0; i < n_samples; ++i) {
            double z = intercept_;
            for (int j = 0; j < n_features; ++j) {
                z += coefficients_[j] * X[i][j];
            }
            predictions[i] = sigmoid(z);
        }
        
        // Compute gradients
        double intercept_gradient = 0.0;
        std::vector<double> coefficient_gradients(n_features, 0.0);
        
        for (int i = 0; i < n_samples; ++i) {
            double error = predictions[i] - y[i];
            intercept_gradient += error;
            
            for (int j = 0; j < n_features; ++j) {
                coefficient_gradients[j] += error * X[i][j];
            }
        }
        
        // Update parameters
        intercept_ -= learning_rate_ * intercept_gradient / n_samples;
        for (int j = 0; j < n_features; ++j) {
            coefficients_[j] -= learning_rate_ * coefficient_gradients[j] / n_samples;
        }
    }
}

std::vector<double> LogisticRegression::predictProbability(const std::vector<std::vector<double>>& X) const {
    if (X.empty() || X[0].size() != coefficients_.size()) {
        throw std::invalid_argument("Invalid input dimensions");
    }
    
    int n_samples = X.size();
    std::vector<double> probabilities(n_samples);
    
    for (int i = 0; i < n_samples; ++i) {
        probabilities[i] = predictProbability(X[i]);
    }
    
    return probabilities;
}

double LogisticRegression::predictProbability(const std::vector<double>& x) const {
    if (x.size() != coefficients_.size()) {
        throw std::invalid_argument("Invalid input dimensions");
    }
    
    double z = intercept_;
    for (size_t j = 0; j < coefficients_.size(); ++j) {
        z += coefficients_[j] * x[j];
    }
    
    return sigmoid(z);
}

std::vector<int> LogisticRegression::predict(const std::vector<std::vector<double>>& X, double threshold) const {
    std::vector<double> probabilities = predictProbability(X);
    int n_samples = X.size();
    std::vector<int> predictions(n_samples);
    
    for (int i = 0; i < n_samples; ++i) {
        predictions[i] = (probabilities[i] >= threshold) ? 1 : 0;
    }
    
    return predictions;
}

std::vector<double> LogisticRegression::getCoefficients() const {
    return coefficients_;
}

double LogisticRegression::getIntercept() const {
    return intercept_;
} 