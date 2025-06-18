#include "../include/OrderBook.h"
#include <iostream>
#include <algorithm>
#include <numeric>
#include <stdexcept>

OrderBook::OrderBook(const std::string& symbol)
    : symbol_(symbol),
      processing_latency_(0)
{
}

void OrderBook::updateFromJson(const nlohmann::json& data) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // std::cout << "OrderBook: Processing update for " << symbol_ << std::endl;
    
    // Create local variables to store the parsed data
    std::string timestamp;
    std::vector<std::pair<double, double>> new_bids;
    std::vector<std::pair<double, double>> new_asks;
    
    // Debug output the received data structure
    // std::cout << "OrderBook: Raw data: " << data.dump(2) << std::endl;
    
    // Handle OKX specific format which uses the following structure:
    // { "data": [{ "asks": [...], "bids": [...], "ts": "..." }] }
    if (data.contains("data") && data["data"].is_array() && !data["data"].empty()) {
        auto& orderbook_data = data["data"][0];
        
        if (orderbook_data.contains("ts") && orderbook_data["ts"].is_string()) {
            timestamp = orderbook_data["ts"];
            // std::cout << "OrderBook: OKX timestamp found: " << timestamp << std::endl;
        }
        
        if (orderbook_data.contains("bids") && orderbook_data["bids"].is_array()) {
            // std::cout << "OrderBook: Processing " << orderbook_data["bids"].size() << " OKX bid levels" << std::endl;
            for (const auto& bid : orderbook_data["bids"]) {
                if (bid.is_array() && bid.size() >= 2) {
                    double price = std::stod(bid[0].get<std::string>());
                    double quantity = std::stod(bid[1].get<std::string>());
                    new_bids.emplace_back(price, quantity);
                }
            }
            // Sort bids in descending order by price
            std::sort(new_bids.begin(), new_bids.end(), [](const auto& a, const auto& b) {
                return a.first > b.first;
            });
            // std::cout << "OrderBook: Processed " << new_bids.size() << " valid bid levels" << std::endl;
            // if (!new_bids.empty()) {
            //     std::cout << "OrderBook: Best bid: " << new_bids[0].first << " x " << new_bids[0].second << std::endl;
            // }
        }
        
        if (orderbook_data.contains("asks") && orderbook_data["asks"].is_array()) {
            // std::cout << "OrderBook: Processing " << orderbook_data["asks"].size() << " OKX ask levels" << std::endl;
            for (const auto& ask : orderbook_data["asks"]) {
                if (ask.is_array() && ask.size() >= 2) {
                    double price = std::stod(ask[0].get<std::string>());
                    double quantity = std::stod(ask[1].get<std::string>());
                    new_asks.emplace_back(price, quantity);
                }
            }
            // Sort asks in ascending order by price
            std::sort(new_asks.begin(), new_asks.end(), [](const auto& a, const auto& b) {
                return a.first < b.first;
            });
            // std::cout << "OrderBook: Processed " << new_asks.size() << " valid ask levels" << std::endl;
            // if (!new_asks.empty()) {
            //     std::cout << "OrderBook: Best ask: " << new_asks[0].first << " x " << new_asks[0].second << std::endl;
            // }
        }
    } 
    // Handle direct format (original implementation)
    else {
        if (data.contains("timestamp") && data["timestamp"].is_string()) {
            timestamp = data["timestamp"];
            // std::cout << "OrderBook: Timestamp found: " << timestamp << std::endl;
        } else {
            // std::cout << "OrderBook: No timestamp found in data" << std::endl;
        }
        
        if (data.contains("bids") && data["bids"].is_array()) {
            // std::cout << "OrderBook: Processing " << data["bids"].size() << " bid levels" << std::endl;
            for (const auto& bid : data["bids"]) {
                if (bid.is_array() && bid.size() >= 2) {
                    double price = std::stod(bid[0].get<std::string>());
                    double quantity = std::stod(bid[1].get<std::string>());
                    new_bids.emplace_back(price, quantity);
                }
            }
            // Sort bids in descending order by price
            std::sort(new_bids.begin(), new_bids.end(), [](const auto& a, const auto& b) {
                return a.first > b.first;
            });
            // std::cout << "OrderBook: Processed " << new_bids.size() << " valid bid levels" << std::endl;
            // if (!new_bids.empty()) {
            //     std::cout << "OrderBook: Best bid: " << new_bids[0].first << " x " << new_bids[0].second << std::endl;
            // }
        } else {
            // std::cout << "OrderBook: No bids found in data" << std::endl;
        }
        
        if (data.contains("asks") && data["asks"].is_array()) {
            // std::cout << "OrderBook: Processing " << data["asks"].size() << " ask levels" << std::endl;
            for (const auto& ask : data["asks"]) {
                if (ask.is_array() && ask.size() >= 2) {
                    double price = std::stod(ask[0].get<std::string>());
                    double quantity = std::stod(ask[1].get<std::string>());
                    new_asks.emplace_back(price, quantity);
                }
            }
            // Sort asks in ascending order by price
            std::sort(new_asks.begin(), new_asks.end(), [](const auto& a, const auto& b) {
                return a.first < b.first;
            });
            // std::cout << "OrderBook: Processed " << new_asks.size() << " valid ask levels" << std::endl;
            // if (!new_asks.empty()) {
            //     std::cout << "OrderBook: Best ask: " << new_asks[0].first << " x " << new_asks[0].second << std::endl;
            // }
        } else {
            // std::cout << "OrderBook: No asks found in data" << std::endl;
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    int64_t latency = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    
    // Update the member variables with a lock
    {
        std::lock_guard<std::mutex> lock(orderbook_mutex_);
        timestamp_ = timestamp;
        bids_ = std::move(new_bids);
        asks_ = std::move(new_asks);
        processing_latency_ = latency;
        // std::cout << "OrderBook: Updated orderbook with " << bids_.size() << " bids and " << asks_.size() 
        //           << " asks, processing took " << latency << " μs" << std::endl;
    }
}

double OrderBook::getMidPrice() const {
    std::lock_guard<std::mutex> lock(orderbook_mutex_);
    if (bids_.empty() || asks_.empty()) {
        return 0.0;
    }
    return (getBestBidInternal() + getBestAskInternal()) / 2.0;
}

double OrderBook::getBestBid() const {
    std::lock_guard<std::mutex> lock(orderbook_mutex_);
    return getBestBidInternal();
}

double OrderBook::getBestBidInternal() const {
    // Internal version without locking - assumes lock is already held
    if (bids_.empty()) {
        return 0.0;
    }
    return bids_[0].first;
}

double OrderBook::getBestAsk() const {
    std::lock_guard<std::mutex> lock(orderbook_mutex_);
    return getBestAskInternal();
}

double OrderBook::getBestAskInternal() const {
    // Internal version without locking - assumes lock is already held
    if (asks_.empty()) {
        return 0.0;
    }
    return asks_[0].first;
}

double OrderBook::getSpread() const {
    std::lock_guard<std::mutex> lock(orderbook_mutex_);
    if (bids_.empty() || asks_.empty()) {
        return 0.0;
    }
    return getBestAskInternal() - getBestBidInternal();
}

double OrderBook::getSpreadPercentage() const {
    std::lock_guard<std::mutex> lock(orderbook_mutex_);
    if (bids_.empty() || asks_.empty()) {
        return 0.0;
    }
    double mid_price = (getBestBidInternal() + getBestAskInternal()) / 2.0;
    if (mid_price <= 0.0) {
        return 0.0;
    }
    return (getBestAskInternal() - getBestBidInternal()) / mid_price * 100.0;
}

double OrderBook::getDepth(const std::string& side, double depth) const {
    std::lock_guard<std::mutex> lock(orderbook_mutex_);
    
    if (side != "bids" && side != "asks") {
        throw std::invalid_argument("Side must be 'bids' or 'asks'");
    }
    
    const auto& levels = (side == "bids") ? bids_ : asks_;
    if (levels.empty()) {
        return 0.0;
    }
    
    double base_price = (side == "bids") ? levels[0].first * (1.0 - depth) : levels[0].first * (1.0 + depth);
    
    double total_volume = 0.0;
    for (const auto& level : levels) {
        if ((side == "bids" && level.first >= base_price) ||
            (side == "asks" && level.first <= base_price)) {
            total_volume += level.second;
        }
    }
    
    return total_volume;
}

double OrderBook::getWeightedAveragePrice(const std::string& side, double volume) const {
    std::lock_guard<std::mutex> lock(orderbook_mutex_);
    
    if (side != "bids" && side != "asks") {
        throw std::invalid_argument("Side must be 'bids' or 'asks'");
    }
    
    const auto& levels = (side == "bids") ? bids_ : asks_;
    if (levels.empty()) {
        return 0.0;
    }
    
    double remaining_volume = volume;
    double price_volume_sum = 0.0;
    double filled_volume = 0.0;
    
    for (const auto& level : levels) {
        double level_price = level.first;
        double level_volume = level.second;
        
        double executed_volume = std::min(remaining_volume, level_volume);
        price_volume_sum += level_price * executed_volume;
        filled_volume += executed_volume;
        
        remaining_volume -= executed_volume;
        if (remaining_volume <= 0.0) {
            break;
        }
    }
    
    if (filled_volume <= 0.0) {
        return 0.0;
    }
    
    return price_volume_sum / filled_volume;
}

double OrderBook::estimateMarketImpact(const std::string& side, double volume) const {
    std::lock_guard<std::mutex> lock(orderbook_mutex_);
    
    if (side != "bids" && side != "asks") {
        throw std::invalid_argument("Side must be 'bids' or 'asks'");
    }
    
    if (bids_.empty() || asks_.empty()) {
        return 0.0;
    }
    
    double mid_price = (getBestBidInternal() + getBestAskInternal()) / 2.0;
    if (mid_price <= 0.0) {
        return 0.0;
    }
    
    // Copy necessary data before releasing the lock for the potentially expensive calculation
    std::vector<std::pair<double, double>> levels = (side == "bids") ? bids_ : asks_;
    
    // Unlock for the calculation to prevent holding lock too long
    orderbook_mutex_.unlock();
    
    double remaining_volume = volume;
    double price_volume_sum = 0.0;
    double filled_volume = 0.0;
    
    for (const auto& level : levels) {
        double level_price = level.first;
        double level_volume = level.second;
        
        double executed_volume = std::min(remaining_volume, level_volume);
        price_volume_sum += level_price * executed_volume;
        filled_volume += executed_volume;
        
        remaining_volume -= executed_volume;
        if (remaining_volume <= 0.0) {
            break;
        }
    }
    
    if (filled_volume <= 0.0) {
        // Re-lock before return
        orderbook_mutex_.lock();
        return 0.0;
    }
    
    double weighted_avg_price = price_volume_sum / filled_volume;
    
    // Re-lock for consistency
    orderbook_mutex_.lock();
    
    // Calculate percentage difference between weighted average price and mid price
    double impact = (side == "bids") 
                    ? (mid_price - weighted_avg_price) / mid_price
                    : (weighted_avg_price - mid_price) / mid_price;
    
    return impact * 100.0; // Return as percentage
}

std::string OrderBook::getTimestamp() const {
    std::lock_guard<std::mutex> lock(orderbook_mutex_);
    return timestamp_;
}

std::string OrderBook::getSymbol() const {
    // Symbol doesn't change, no need for lock
    return symbol_;
}

int64_t OrderBook::getProcessingLatency() const {
    std::lock_guard<std::mutex> lock(orderbook_mutex_);
    return processing_latency_;
}

double OrderBook::getBookPressure(int levels) const {
    std::lock_guard<std::mutex> lock(orderbook_mutex_);
    
    if (bids_.empty() || asks_.empty()) {
        return 0.0;  // No pressure if orderbook is empty
    }
    
    // Calculate total volume for top N levels on each side
    double bid_volume = 0.0;
    double ask_volume = 0.0;
    
    int bid_count = 0;
    int ask_count = 0;
    
    // Sum bid volume for top N levels
    for (const auto& bid : bids_) {
        if (bid_count >= levels) break;
        bid_volume += bid.second;
        bid_count++;
    }
    
    // Sum ask volume for top N levels
    for (const auto& ask : asks_) {
        if (ask_count >= levels) break;
        ask_volume += ask.second;
        ask_count++;
    }
    
    // Calculate pressure as normalized imbalance
    if (bid_volume + ask_volume > 0.0) {
        return (bid_volume - ask_volume) / (bid_volume + ask_volume);
    } else {
        return 0.0;
    }
} 