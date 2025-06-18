#pragma once

#include <string>
#include <unordered_map>
#include <vector>
#include <cmath>

/**
 * @brief Implementation of a fee model for exchange fee calculation
 */
class FeeModel {
public:
    /**
     * @brief Constructor
     * @param exchange The exchange name
     */
    explicit FeeModel(const std::string& exchange);
    
    /**
     * @brief Destructor
     */
    ~FeeModel() = default;
    
    /**
     * @brief Calculate the fee for a given trade
     * @param orderType Order type (e.g., "market", "limit")
     * @param side Order side ("buy" or "sell")
     * @param quantity Quantity to trade
     * @param price Price of the asset
     * @param feeTier Fee tier level
     * @param makerProbability Probability of the order being a maker
     * @return Calculated fee
     */
    double calculateFee(const std::string& orderType, const std::string& side, 
                       double quantity, double price, int feeTier, double makerProbability) const;
    
    /**
     * @brief Get maker fee rate for a given tier
     * @param feeTier Fee tier level
     * @return Maker fee rate
     */
    double getMakerFeeRate(int feeTier) const;
    
    /**
     * @brief Get taker fee rate for a given tier
     * @param feeTier Fee tier level
     * @return Taker fee rate
     */
    double getTakerFeeRate(int feeTier) const;
    
    /**
     * @brief Calculate the maker/taker weighted fee
     * @param makerFeeRate Maker fee rate
     * @param takerFeeRate Taker fee rate
     * @param makerProbability Probability of the order being a maker
     * @return Weighted fee rate
     */
    double getWeightedFeeRate(double makerFeeRate, double takerFeeRate, double makerProbability) const;
    
    /**
     * @brief Set the exchange
     * @param exchange The exchange name
     */
    void setExchange(const std::string& exchange);
    
    /**
     * @brief Get the current exchange
     * @return The exchange name
     */
    std::string getExchange() const;
    
private:
    std::string exchange_;
    std::unordered_map<std::string, std::vector<std::pair<int, double>>> maker_fee_tiers_;
    std::unordered_map<std::string, std::vector<std::pair<int, double>>> taker_fee_tiers_;
    
    /**
     * @brief Initialize fee tiers for OKX
     */
    void initializeOKXFeeTiers();
}; 