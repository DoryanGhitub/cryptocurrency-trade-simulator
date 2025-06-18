#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <eigen3/Eigen/Dense>

/**
 * @brief Implementation of linear regression for slippage prediction
 */
class LinearRegression {
public:
    /**
     * @brief Constructor
     */
    LinearRegression() = default;
    
    /**
     * @brief Fit the model to the training data
     * @param X Features (independent variables)
     * @param y Target (dependent variable)
     */
    void fit(const std::vector<std::vector<double>>& X, const std::vector<double>& y);
    
    /**
     * @brief Predict using the fitted model
     * @param X Features for prediction
     * @return Predicted values
     */
    std::vector<double> predict(const std::vector<std::vector<double>>& X) const;
    
    /**
     * @brief Predict a single value
     * @param x Features for a single prediction
     * @return Predicted value
     */
    double predict(const std::vector<double>& x) const;
    
    /**
     * @brief Get the model's coefficients
     * @return Vector of coefficients
     */
    std::vector<double> getCoefficients() const;
    
    /**
     * @brief Get the model's intercept
     * @return Intercept value
     */
    double getIntercept() const;

private:
    std::vector<double> coefficients_;
    double intercept_ = 0.0;
};

/**
 * @brief Implementation of quantile regression for slippage prediction
 */
class QuantileRegression {
public:
    /**
     * @brief Constructor
     * @param quantile The quantile to estimate (between 0 and 1)
     */
    explicit QuantileRegression(double quantile = 0.5);
    
    /**
     * @brief Fit the model to the training data
     * @param X Features (independent variables)
     * @param y Target (dependent variable)
     * @param max_iterations Maximum number of iterations for optimization
     * @param tolerance Convergence tolerance
     */
    void fit(const std::vector<std::vector<double>>& X, const std::vector<double>& y,
             int max_iterations = 1000, double tolerance = 1e-6);
    
    /**
     * @brief Predict using the fitted model
     * @param X Features for prediction
     * @return Predicted values at the specified quantile
     */
    std::vector<double> predict(const std::vector<std::vector<double>>& X) const;
    
    /**
     * @brief Predict a single value
     * @param x Features for a single prediction
     * @return Predicted value at the specified quantile
     */
    double predict(const std::vector<double>& x) const;
    
    /**
     * @brief Get the model's coefficients
     * @return Vector of coefficients
     */
    std::vector<double> getCoefficients() const;
    
    /**
     * @brief Get the model's intercept
     * @return Intercept value
     */
    double getIntercept() const;
    
    /**
     * @brief Set the quantile
     * @param quantile The quantile to estimate (between 0 and 1)
     */
    void setQuantile(double quantile);
    
    /**
     * @brief Get the quantile
     * @return The current quantile
     */
    double getQuantile() const;

private:
    double quantile_;
    std::vector<double> coefficients_;
    double intercept_ = 0.0;
    
    /**
     * @brief Calculate the quantile loss
     * @param y_true True values
     * @param y_pred Predicted values
     * @return Quantile loss
     */
    double quantileLoss(const std::vector<double>& y_true, const std::vector<double>& y_pred) const;
};

/**
 * @brief Implementation of logistic regression for maker/taker proportion prediction
 */
class LogisticRegression {
public:
    /**
     * @brief Constructor
     * @param learning_rate Learning rate for gradient descent
     * @param max_iterations Maximum number of iterations
     */
    LogisticRegression(double learning_rate = 0.01, int max_iterations = 1000);
    
    /**
     * @brief Fit the model to the training data
     * @param X Features (independent variables)
     * @param y Target (dependent variable, binary 0/1)
     */
    void fit(const std::vector<std::vector<double>>& X, const std::vector<int>& y);
    
    /**
     * @brief Predict probabilities using the fitted model
     * @param X Features for prediction
     * @return Predicted probabilities (between 0 and 1)
     */
    std::vector<double> predictProbability(const std::vector<std::vector<double>>& X) const;
    
    /**
     * @brief Predict a single probability
     * @param x Features for a single prediction
     * @return Predicted probability (between 0 and 1)
     */
    double predictProbability(const std::vector<double>& x) const;
    
    /**
     * @brief Predict classes using the fitted model
     * @param X Features for prediction
     * @param threshold Probability threshold for class prediction
     * @return Predicted classes (0 or 1)
     */
    std::vector<int> predict(const std::vector<std::vector<double>>& X, double threshold = 0.5) const;
    
    /**
     * @brief Get the model's coefficients
     * @return Vector of coefficients
     */
    std::vector<double> getCoefficients() const;
    
    /**
     * @brief Get the model's intercept
     * @return Intercept value
     */
    double getIntercept() const;

private:
    double learning_rate_;
    int max_iterations_;
    std::vector<double> coefficients_;
    double intercept_ = 0.0;
    
    /**
     * @brief Sigmoid function
     * @param z Input value
     * @return Sigmoid of z
     */
    static double sigmoid(double z);
}; 