#include "FeeModel.h"
#include <stdexcept>
#include <iostream>

FeeModel::FeeModel(const std::string& exchange)
    : exchange_(exchange)
{
    if (exchange == "OKX") {
        initializeOKXFeeTiers();
    } else {
        throw std::invalid_argument("Unsupported exchange: " + exchange);
    }
}

void FeeModel::initializeOKXFeeTiers() {
    // Initialize OKX fee tiers based on their documentation
    // Format: <tier, fee_rate>
    
    // Maker fee tiers for OKX
    // Updated rates for OKX exchange
    std::vector<std::pair<int, double>> maker_tiers = {
        {1, 0.0010}, // VIP 0, spot maker fee
        {2, 0.0008}, // VIP 1
        {3, 0.0006}, // VIP 2
        {4, 0.0004}, // VIP 3
        {5, 0.0002}, // VIP 4
        {6, 0.0000}, // VIP 5
    };
    
    // Taker fee tiers for OKX
    std::vector<std::pair<int, double>> taker_tiers = {
        {1, 0.0015}, // VIP 0, spot taker fee
        {2, 0.0012}, // VIP 1
        {3, 0.0010}, // VIP 2
        {4, 0.0008}, // VIP 3
        {5, 0.0006}, // VIP 4
        {6, 0.0005}, // VIP 5
    };
    
    maker_fee_tiers_["OKX"] = maker_tiers;
    taker_fee_tiers_["OKX"] = taker_tiers;
}

double FeeModel::calculateFee(const std::string& orderType, const std::string& side, 
                           double quantity, double price, int feeTier, double makerProbability) const {
    if (exchange_.empty()) {
        throw std::invalid_argument("Exchange not set");
    }
    
    if (maker_fee_tiers_.find(exchange_) == maker_fee_tiers_.end() ||
        taker_fee_tiers_.find(exchange_) == taker_fee_tiers_.end()) {
        throw std::invalid_argument("Fee tiers not found for exchange: " + exchange_);
    }
    
    // Get fee rates
    double maker_fee_rate = getMakerFeeRate(feeTier);
    double taker_fee_rate = getTakerFeeRate(feeTier);
    
    // Calculate the weighted fee rate based on maker/taker probability
    double weighted_fee_rate = getWeightedFeeRate(maker_fee_rate, taker_fee_rate, makerProbability);
    
    // Calculate the fee amount
    // If price is not provided or is too low, use a default price of $100,000 for BTC-USDT
    double effective_price = (price <= 1.0) ? 100000.0 : price;
    double notional_value = quantity * effective_price;
    double fee = notional_value * weighted_fee_rate;
    
    // Ensure a minimum fee of $0.01 to avoid zero values
    return std::max(0.01, fee);
}

double FeeModel::getMakerFeeRate(int feeTier) const {
    if (exchange_.empty()) {
        throw std::invalid_argument("Exchange not set");
    }
    
    if (maker_fee_tiers_.find(exchange_) == maker_fee_tiers_.end()) {
        throw std::invalid_argument("Maker fee tiers not found for exchange: " + exchange_);
    }
    
    const auto& tiers = maker_fee_tiers_.at(exchange_);
    
    // Find the tier closest to the requested tier
    double fee_rate = tiers[0].second; // Default to the first tier
    for (const auto& tier : tiers) {
        if (tier.first <= feeTier) {
            fee_rate = tier.second;
        } else {
            break;
        }
    }
    
    return fee_rate;
}

double FeeModel::getTakerFeeRate(int feeTier) const {
    if (exchange_.empty()) {
        throw std::invalid_argument("Exchange not set");
    }
    
    if (taker_fee_tiers_.find(exchange_) == taker_fee_tiers_.end()) {
        throw std::invalid_argument("Taker fee tiers not found for exchange: " + exchange_);
    }
    
    const auto& tiers = taker_fee_tiers_.at(exchange_);
    
    // Find the tier closest to the requested tier
    double fee_rate = tiers[0].second; // Default to the first tier
    for (const auto& tier : tiers) {
        if (tier.first <= feeTier) {
            fee_rate = tier.second;
        } else {
            break;
        }
    }
    
    return fee_rate;
}

double FeeModel::getWeightedFeeRate(double makerFeeRate, double takerFeeRate, double makerProbability) const {
    if (makerProbability < 0.0 || makerProbability > 1.0) {
        throw std::invalid_argument("Maker probability must be between 0 and 1");
    }
    
    return makerFeeRate * makerProbability + takerFeeRate * (1.0 - makerProbability);
}

void FeeModel::setExchange(const std::string& exchange) {
    if (exchange == "OKX") {
        exchange_ = exchange;
        initializeOKXFeeTiers();
    } else {
        throw std::invalid_argument("Unsupported exchange: " + exchange);
    }
}

std::string FeeModel::getExchange() const {
    return exchange_;
} 