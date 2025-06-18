#pragma once

#include <string>
#include <memory>
#include <functional>
#include <atomic>
#include <thread>
#include <mutex>
#include "../include/TradeSimulator.h"

/**
 * @brief Class for handling the user interface components
 */
class UserInterface {
public:
    /**
     * @brief Constructor
     * @param simulator Pointer to the trade simulator
     */
    explicit UserInterface(std::shared_ptr<TradeSimulator> simulator);
    
    /**
     * @brief Destructor
     */
    ~UserInterface();
    
    /**
     * @brief Initialize the UI
     */
    void initialize();
    
    /**
     * @brief Run the UI main loop
     */
    void run();
    
    /**
     * @brief Stop the UI
     */
    void stop();
    
    /**
     * @brief Check if UI is running
     * @return true if running, false otherwise
     */
    bool isRunning() const;
    
    /**
     * @brief Get the selected volatility value
     * @return Current volatility value
     */
    double getVolatility() const;
    
    /**
     * @brief Update the UI with latest data
     */
    void update();
    
    /**
     * @brief Update the selected parameters from the simulator
     */
    void updateInputParameters();
    
    /**
     * @brief Set callback for exchange selection
     * @param callback Function to be called when exchange is selected
     */
    void setExchangeCallback(std::function<void(const std::string&)> callback);
    
    /**
     * @brief Set callback for symbol selection
     * @param callback Function to be called when symbol is selected
     */
    void setSymbolCallback(std::function<void(const std::string&)> callback);
    
    /**
     * @brief Set callback for order type selection
     * @param callback Function to be called when order type is selected
     */
    void setOrderTypeCallback(std::function<void(const std::string&)> callback);
    
    /**
     * @brief Set callback for quantity selection
     * @param callback Function to be called when quantity is selected
     */
    void setQuantityCallback(std::function<void(double)> callback);
    
    /**
     * @brief Set callback for volatility selection
     * @param callback Function to be called when volatility is selected
     */
    void setVolatilityCallback(std::function<void(double)> callback);
    
    /**
     * @brief Set callback for fee tier selection
     * @param callback Function to be called when fee tier is selected
     */
    void setFeeTierCallback(std::function<void(int)> callback);
    
private:
    std::shared_ptr<TradeSimulator> simulator_;
    
    std::string selected_exchange_;
    std::string selected_symbol_;
    std::string selected_order_type_;
    double selected_quantity_;
    double selected_volatility_;
    int selected_fee_tier_;
    
    std::atomic<bool> running_;
    std::unique_ptr<std::thread> ui_thread_;
    mutable std::mutex ui_mutex_;
    
    // Visual state for orderbook
    std::vector<std::pair<std::string, std::string>> bid_levels_;
    std::vector<std::pair<std::string, std::string>> ask_levels_;
    std::vector<int> bid_bars_;
    std::vector<int> ask_bars_;
    
    std::function<void(const std::string&)> exchange_callback_;
    std::function<void(const std::string&)> symbol_callback_;
    std::function<void(const std::string&)> order_type_callback_;
    std::function<void(double)> quantity_callback_;
    std::function<void(double)> volatility_callback_;
    std::function<void(int)> fee_tier_callback_;
    
    /**
     * @brief Draw the input parameters panel
     * @param row The current row to draw
     * @param width The width of the panel
     */
    void drawInputPanel(int row, int width);
    
    /**
     * @brief Draw the output parameters panel
     * @param row The current row to draw
     * @param width The width of the panel
     */
    void drawOutputPanel(int row, int width);
    
    /**
     * @brief Draw the orderbook visualization
     */
    void drawOrderbook();
    
    /**
     * @brief Draw the orderbook header
     * @param width The width of the header
     */
    void drawOrderbookHeader(int width);
    
    /**
     * @brief Draw a row of the orderbook
     * @param row The row number
     * @param width The width of the row
     */
    void drawOrderbookRow(int row, int width);
    
    /**
     * @brief Update orderbook data for visualization
     */
    void updateOrderbookData();
    
    /**
     * @brief Draw a horizontal separator
     * @param width Width of the separator
     * @param character Character to use for the separator
     */
    void drawSeparator(int width = 80, char character = '-');
    
    /**
     * @brief Draw a UI box with title
     * @param title Title for the box
     * @param width Width of the box
     */
    void drawBoxTitle(const std::string& title, int width = 80);
}; 