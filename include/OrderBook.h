#pragma once

#include <string>
#include <vector>
#include <map>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <chrono>
#include <mutex>

/**
 * @brief Class for handling and processing L2 orderbook data
 */
class OrderBook {
public:
    OrderBook(const std::string& symbol);
    ~OrderBook() = default;

    /**
     * @brief Update the orderbook with new data
     * @param data JSON object containing the orderbook data
     */
    void updateFromJson(const nlohmann::json& data);

    /**
     * @brief Get mid price
     * @return The mid price between best bid and best ask
     */
    double getMidPrice() const;

    /**
     * @brief Get best bid price
     * @return The highest bid price
     */
    double getBestBid() const;

    /**
     * @brief Get best ask price
     * @return The lowest ask price
     */
    double getBestAsk() const;

    /**
     * @brief Get the spread
     * @return The spread between best bid and best ask
     */
    double getSpread() const;

    /**
     * @brief Get the spread as a percentage of the mid price
     * @return The spread percentage
     */
    double getSpreadPercentage() const;

    /**
     * @brief Calculate market depth up to a certain price level
     * @param side "bids" or "asks"
     * @param depth How deep to calculate
     * @return Total volume available
     */
    double getDepth(const std::string& side, double depth) const;

    /**
     * @brief Get weighted average price for a given volume
     * @param side "bids" or "asks"
     * @param volume The volume to execute
     * @return Weighted average price
     */
    double getWeightedAveragePrice(const std::string& side, double volume) const;

    /**
     * @brief Calculate market impact for a given volume
     * @param side "bids" or "asks"
     * @param volume The volume to execute
     * @return Estimated market impact
     */
    double estimateMarketImpact(const std::string& side, double volume) const;

    /**
     * @brief Get the current timestamp of the orderbook
     * @return Timestamp as string
     */
    std::string getTimestamp() const;

    /**
     * @brief Get the symbol for this orderbook
     * @return Symbol string
     */
    std::string getSymbol() const;

    /**
     * @brief Get processing latency
     * @return Processing time in microseconds
     */
    int64_t getProcessingLatency() const;

    /**
     * @brief Calculate order book pressure (imbalance) for a given number of levels
     * @param levels Number of price levels to consider
     * @return Pressure value between -1.0 and 1.0 (negative: sell pressure, positive: buy pressure)
     */
    double getBookPressure(int levels) const;

private:
    /**
     * @brief Get best bid price (internal method, assumes lock is held)
     * @return The highest bid price
     */
    double getBestBidInternal() const;

    /**
     * @brief Get best ask price (internal method, assumes lock is held)
     * @return The lowest ask price
     */
    double getBestAskInternal() const;

    std::string symbol_;
    std::string timestamp_;
    std::vector<std::pair<double, double>> bids_; // price, quantity
    std::vector<std::pair<double, double>> asks_; // price, quantity
    int64_t processing_latency_;
    
    // Mutex for thread safety
    mutable std::mutex orderbook_mutex_;
}; 