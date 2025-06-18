#pragma once

#include <string>
#include <memory>
#include <atomic>
#include <chrono>
#include <mutex>
#include <nlohmann/json.hpp>

#include "OrderBook.h"
#include "WebSocketClient.h"
#include "../models/AlmgrenChriss.h"
#include "../models/RegressionModels.h"
#include "../models/FeeModel.h"
#include "TransformerModel.h"

/**
 * @brief Main simulator class coordinating all components
 */
class TradeSimulator {
public:
    /**
     * @brief Constructor
     */
    TradeSimulator();
    
    /**
     * @brief Destructor
     */
    ~TradeSimulator();
    
    /**
     * @brief Initialize the simulator
     * @param websocketUrl WebSocket URL for L2 orderbook data
     * @return true if initialization succeeded, false otherwise
     */
    bool initialize(const std::string& websocketUrl);
    
    /**
     * @brief Connect to the WebSocket
     * @return true if connection succeeded, false otherwise
     */
    bool connect();
    
    /**
     * @brief Disconnect from the WebSocket
     */
    void disconnect();
    
    /**
     * @brief Check if connected to the WebSocket
     * @return true if connected, false otherwise
     */
    bool isConnected() const;
    
    /**
     * @brief Set parameters for simulation
     * @param exchange Exchange name
     * @param symbol Asset symbol
     * @param orderType Order type
     * @param quantity Quantity to trade
     * @param volatility Asset volatility
     * @param feeTier Fee tier level
     */
    void setParameters(const std::string& exchange, const std::string& symbol,
                      const std::string& orderType, double quantity,
                      double volatility, int feeTier);
    
    /**
     * @brief Get the estimated slippage
     * @return Estimated slippage
     */
    double getExpectedSlippage() const;
    
    /**
     * @brief Get the expected fees
     * @return Expected fees
     */
    double getExpectedFees() const;
    
    /**
     * @brief Get the expected market impact
     * @return Expected market impact
     */
    double getExpectedMarketImpact() const;
    
    /**
     * @brief Get the net cost
     * @return Net cost (slippage + fees + market impact)
     */
    double getNetCost() const;
    
    /**
     * @brief Get the maker/taker proportion
     * @return Maker/taker proportion
     */
    double getMakerTakerProportion() const;
    
    /**
     * @brief Get the internal latency
     * @return Internal latency in microseconds
     */
    int64_t getInternalLatency() const;
    
    /**
     * @brief Get the current price
     * @return Current price
     */
    double getCurrentPrice() const;
    
    /**
     * @brief Get the spread
     * @return Spread
     */
    double getSpread() const;
    
    /**
     * @brief Get the orderbook depth
     * @return Orderbook depth
     */
    double getOrderbookDepth() const;
    
    /**
     * @brief Get the last update timestamp
     * @return Timestamp as string
     */
    std::string getLastUpdateTimestamp() const;
    
    /**
     * @brief Get the current exchange
     * @return Exchange name
     */
    std::string getExchange() const;
    
    /**
     * @brief Get the current symbol
     * @return Symbol name
     */
    std::string getSymbol() const;
    
    /**
     * @brief Get the current order type
     * @return Order type (market/limit)
     */
    std::string getOrderType() const;
    
    /**
     * @brief Get the current quantity
     * @return Quantity
     */
    double getQuantity() const;
    
    /**
     * @brief Get the current volatility
     * @return Volatility
     */
    double getVolatility() const;
    
    /**
     * @brief Get the current fee tier
     * @return Fee tier
     */
    int getFeeTier() const;
    
private:
    std::string exchange_;
    std::string symbol_;
    std::string order_type_;
    double quantity_;
    double volatility_;
    int fee_tier_;
    
    std::unique_ptr<OrderBook> orderbook_;
    std::unique_ptr<WebSocketClient> websocket_client_;
    std::unique_ptr<AlmgrenChriss> almgren_chriss_model_;
    std::unique_ptr<LinearRegression> slippage_model_;
    std::unique_ptr<TransformerModel> maker_taker_model_;
    std::unique_ptr<FeeModel> fee_model_;
    
    mutable std::mutex data_mutex_;
    std::atomic<bool> running_;
    
    double expected_slippage_;
    double expected_fees_;
    double expected_market_impact_;
    double maker_taker_proportion_;
    int64_t internal_latency_;
    
    /**
     * @brief Process received WebSocket message
     * @param message JSON message from WebSocket
     */
    void processMessage(const nlohmann::json& message);
    
    /**
     * @brief Calculate all output parameters based on current data
     */
    void calculateOutputParameters();
}; 