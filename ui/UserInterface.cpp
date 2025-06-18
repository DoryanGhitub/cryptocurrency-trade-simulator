#include "../ui/UserInterface.h"
#include <iostream>
#include <iomanip>
#include <thread>
#include <chrono>
#include <string>
#include <algorithm>
#include <sstream>
#include <cstdlib>
#include <ctime>

// ANSI color codes for terminal
#define RESET   "\033[0m"
#define BLACK   "\033[30m"
#define RED     "\033[31m"
#define GREEN   "\033[32m"
#define YELLOW  "\033[33m"
#define BLUE    "\033[34m"
#define MAGENTA "\033[35m"
#define CYAN    "\033[36m"
#define WHITE   "\033[37m"
#define BOLD    "\033[1m"
#define DIM     "\033[2m"
#define UNDERLINE "\033[4m"
#define BG_BLACK "\033[40m"
#define BG_RED   "\033[41m"
#define BG_GREEN "\033[42m"
#define BG_YELLOW "\033[43m"
#define BG_BLUE  "\033[44m"
#define BG_MAGENTA "\033[45m"
#define BG_CYAN  "\033[46m"
#define BG_WHITE "\033[47m"
#define BG_DARK_GREEN "\033[48;5;22m"
#define BG_DARK_RED "\033[48;5;52m"

UserInterface::UserInterface(std::shared_ptr<TradeSimulator> simulator)
    : simulator_(simulator),
      selected_exchange_("OKX"),
      selected_symbol_("BTC-USDT-SWAP"),
      selected_order_type_("market"),
      selected_quantity_(100.0),
      selected_volatility_(0.01),
      selected_fee_tier_(1),
      running_(false)
{
    // Initialize random seed for orderbook visualization
    std::srand(static_cast<unsigned int>(std::time(nullptr)));
}

UserInterface::~UserInterface() {
    stop();
}

void UserInterface::initialize() {
    // Register callbacks
    exchange_callback_ = [this](const std::string& exchange) {
        if (simulator_) {
            simulator_->setParameters(
                exchange, selected_symbol_, selected_order_type_,
                selected_quantity_, selected_volatility_, selected_fee_tier_);
        }
    };
    
    symbol_callback_ = [this](const std::string& symbol) {
        if (simulator_) {
            simulator_->setParameters(
                selected_exchange_, symbol, selected_order_type_,
                selected_quantity_, selected_volatility_, selected_fee_tier_);
        }
    };
    
    order_type_callback_ = [this](const std::string& orderType) {
        if (simulator_) {
            simulator_->setParameters(
                selected_exchange_, selected_symbol_, orderType,
                selected_quantity_, selected_volatility_, selected_fee_tier_);
        }
    };
    
    quantity_callback_ = [this](double quantity) {
        if (simulator_) {
            simulator_->setParameters(
                selected_exchange_, selected_symbol_, selected_order_type_,
                quantity, selected_volatility_, selected_fee_tier_);
        }
    };
    
    volatility_callback_ = [this](double volatility) {
        if (simulator_) {
            simulator_->setParameters(
                selected_exchange_, selected_symbol_, selected_order_type_,
                selected_quantity_, volatility, selected_fee_tier_);
        }
    };
    
    fee_tier_callback_ = [this](int feeTier) {
        if (simulator_) {
            simulator_->setParameters(
                selected_exchange_, selected_symbol_, selected_order_type_,
                selected_quantity_, selected_volatility_, feeTier);
        }
    };
}

void UserInterface::run() {
    if (running_) {
        return;
    }
    
    running_ = true;
    
    // Start the update thread to refresh UI
    ui_thread_ = std::make_unique<std::thread>([this]() {
        while (running_) {
            // Update orderbook data
            updateOrderbookData();
            
            // Update the UI with the latest data
            update();
            
            // Sleep for a short interval
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    });
}

void UserInterface::stop() {
    running_ = false;
    
    if (ui_thread_ && ui_thread_->joinable()) {
        ui_thread_->join();
    }
}

bool UserInterface::isRunning() const {
    return running_;
}

double UserInterface::getVolatility() const {
    std::lock_guard<std::mutex> lock(ui_mutex_);
    return selected_volatility_;
}

void UserInterface::updateInputParameters() {
    if (simulator_) {
        // Update the UI with the current parameters from the simulator
        // This ensures UI is in sync with the actual simulator parameters
        selected_exchange_ = simulator_->getExchange();
        selected_symbol_ = simulator_->getSymbol();
        selected_order_type_ = simulator_->getOrderType();
        selected_quantity_ = simulator_->getQuantity();
        
        double vol = simulator_->getVolatility();
        // std::cout << "DEBUG: UserInterface::updateInputParameters() received volatility " << vol << std::endl;
        selected_volatility_ = vol;
        
        selected_fee_tier_ = simulator_->getFeeTier();
        
        // Debug logging to show all parameters
        // std::cout << "DEBUG: UserInterface parameters after update: " << std::endl;
        // std::cout << "  Exchange: " << selected_exchange_ << std::endl;
        // std::cout << "  Symbol: " << selected_symbol_ << std::endl;
        // std::cout << "  Order Type: " << selected_order_type_ << std::endl;
        // std::cout << "  Quantity: " << selected_quantity_ << std::endl;
        // std::cout << "  Volatility: " << selected_volatility_ << std::endl;
        // std::cout << "  Fee Tier: " << selected_fee_tier_ << std::endl;
    }
}

void UserInterface::update() {
    std::lock_guard<std::mutex> lock(ui_mutex_);
    
    // Update parameters from simulator
    updateInputParameters();
    
    // Clear the console (works in most terminals)
    std::cout << "\033[2J\033[1;1H";
    
    // Create a two-column layout
    // Let's assume terminal width of 120 characters
    const int total_width = 120;
    const int left_width = 40;
    const int right_width = total_width - left_width - 1; // -1 for the separator
    
    // Draw the price and percentages at the top (like the image)
    if (simulator_ && simulator_->getCurrentPrice() > 0.0) {
        double bid_price = simulator_->getCurrentPrice() - simulator_->getSpread()/2;
        double ask_price = simulator_->getCurrentPrice() + simulator_->getSpread()/2;
        
        // Bid percentage and ask percentage should sum to 100%
        double bid_percent = 96.55; // In real implementation, this would be calculated
        double ask_percent = 100.0 - bid_percent;
        
        // Format with background colors like the image
        std::cout << BOLD << "Bid" << std::string(40, ' ') << "Ask" << RESET << std::endl;
        
        // Show bid percentage with green background and ask percentage with red background
        std::cout << BG_DARK_GREEN << GREEN << BOLD 
                  << std::fixed << std::setprecision(2) << bid_percent << "%" 
                  << std::string(36, ' ') << RESET;
        
        std::cout << BG_DARK_RED << RED << BOLD 
                  << std::fixed << std::setprecision(2) << ask_percent << "%" << RESET
                  << std::endl;
        
        // Show bid and ask prices
        std::cout << GREEN << BOLD << "NSDQ " << std::fixed << std::setprecision(3) 
                  << bid_price << RESET << "    " << std::setw(2) << "29" << "  ";
        
        std::cout << RED << BOLD << "NSDQ " << std::fixed << std::setprecision(3) 
                  << ask_price << RESET << "    " << std::setw(1) << "1" << std::endl;
    }
    
    // Draw "Order Book" title with icon
    std::cout << BOLD << "Order Book" << " " << RESET << "ⓘ";
    std::cout << std::string(90, ' ') << "⚙️ " << "📊 " << "10" << std::endl;
    
    // Draw the order book
    drawOrderbook();
    
    // Draw the info section below the main UI
    std::cout << YELLOW << std::endl;
    std::cout << "Input Parameters:" << std::endl;
    std::cout << "  Exchange: " << selected_exchange_ << std::endl;
    std::cout << "  Symbol: " << selected_symbol_ << std::endl;
    std::cout << "  Order Type: " << selected_order_type_ << std::endl;
    std::cout << "  Quantity: " << selected_quantity_ << std::endl;
    std::cout << "  Volatility: " << std::fixed << std::setprecision(6) << selected_volatility_ << std::endl;
    std::cout << "  Fee Tier: " << selected_fee_tier_ << std::endl;
    
    std::cout << std::endl << CYAN;
    std::cout << "Output Parameters:" << std::endl;
    if (simulator_) {
        std::cout << "  Expected Slippage: " << std::fixed << std::setprecision(6) << simulator_->getExpectedSlippage() << "%" << std::endl;
        std::cout << "  Expected Fees: $" << std::fixed << std::setprecision(6) << simulator_->getExpectedFees() << std::endl;
        std::cout << "  Expected Market Impact: " << std::fixed << std::setprecision(6) << simulator_->getExpectedMarketImpact() << "%" << std::endl;
        std::cout << "  Net Cost: " << std::fixed << std::setprecision(6) << simulator_->getNetCost() << "%" << std::endl;
        std::cout << "  Maker/Taker: " << std::fixed << std::setprecision(2) << simulator_->getMakerTakerProportion() * 100.0 << "% maker / "
                  << std::fixed << std::setprecision(2) << (1.0 - simulator_->getMakerTakerProportion()) * 100.0 << "% taker" << std::endl;
        std::cout << "  Internal Latency: " << simulator_->getInternalLatency() << " μs" << std::endl;
        std::cout << "  Last Update: " << simulator_->getLastUpdateTimestamp() << std::endl;
    }
    std::cout << RESET;
}

void UserInterface::drawInputPanel(int row, int width) {
    switch (row) {
        case 0:
            std::cout << YELLOW << " Exchange: " << RESET << BOLD << selected_exchange_ << RESET;
            break;
        case 1:
            std::cout << YELLOW << " Symbol: " << RESET << BOLD << selected_symbol_ << RESET;
            break;
        case 2:
            std::cout << YELLOW << " Order Type: " << RESET << BOLD << selected_order_type_ << RESET;
            break;
        case 3:
            std::cout << YELLOW << " Quantity: " << RESET << BOLD << selected_quantity_;
            if (simulator_ && simulator_->getCurrentPrice() > 0) {
                std::cout << " (~$" << std::fixed << std::setprecision(2) 
                          << (selected_quantity_ * simulator_->getCurrentPrice()) << ")";
            }
            std::cout << RESET;
            break;
        case 4:
            std::cout << YELLOW << " Volatility: " << RESET << BOLD << std::fixed << std::setprecision(6) << selected_volatility_ << RESET;
            break;
        case 5:
            std::cout << YELLOW << " Fee Tier: " << RESET << BOLD << selected_fee_tier_ << RESET;
            break;
        default:
            // Empty row
            break;
    }
    
    // Pad to fill the width
    int current_pos = std::cout.tellp();
    if (current_pos < width) {
        std::cout << std::string(width - current_pos, ' ');
    }
}

void UserInterface::drawOutputPanel(int row, int width) {
    if (!simulator_) {
        std::cout << std::string(width, ' ');
        return;
    }
    
    // Get values from simulator
    double expected_slippage = simulator_->getExpectedSlippage();
    double expected_fees = simulator_->getExpectedFees();
    double expected_market_impact = simulator_->getExpectedMarketImpact();
    double net_cost = simulator_->getNetCost();
    double maker_taker_proportion = simulator_->getMakerTakerProportion();
    int64_t internal_latency = simulator_->getInternalLatency();
    double current_price = simulator_->getCurrentPrice();
    double spread = simulator_->getSpread();
    
    switch (row) {
        case 0:
            std::cout << CYAN << " Expected Slippage: " << RESET << YELLOW << std::fixed << std::setprecision(6) << expected_slippage << "%" << RESET;
            break;
        case 1:
            std::cout << CYAN << " Expected Fees: " << RESET << YELLOW << "$" << std::fixed << std::setprecision(6) << expected_fees << RESET;
            break;
        case 2:
            std::cout << CYAN << " Expected Market Impact: " << RESET << YELLOW << std::fixed << std::setprecision(6) << expected_market_impact << "%" << RESET;
            break;
        case 3:
            std::cout << CYAN << " Net Cost: " << RESET << YELLOW << std::fixed << std::setprecision(6) << net_cost << "%" << RESET;
            break;
        case 4:
            std::cout << CYAN << " Maker/Taker Proportion: " << RESET << YELLOW 
                      << std::fixed << std::setprecision(2) << maker_taker_proportion * 100.0 << "% maker / " 
                      << std::fixed << std::setprecision(2) << (1.0 - maker_taker_proportion) * 100.0 << "% taker" << RESET;
            break;
        case 5:
            std::cout << CYAN << " Internal Latency: " << RESET << YELLOW << internal_latency << " μs" << RESET;
            break;
        case 7:
            std::cout << WHITE << BOLD << " MARKET DATA" << RESET;
            break;
        case 8:
            std::cout << CYAN << " Current Price: " << RESET << GREEN << "$" << std::fixed << std::setprecision(2) << current_price << RESET;
            break;
        case 9:
            std::cout << CYAN << " Spread: " << RESET << GREEN << "$" << std::fixed << std::setprecision(6) << spread;
            // Avoid division by zero
            if (current_price > 0.0) {
                std::cout << " (" << std::fixed << std::setprecision(6) << (spread / current_price) * 100.0 << "%)";
            } else {
                std::cout << " (N/A)";
            }
            std::cout << RESET;
            break;
        default:
            // Empty row
            break;
    }
    
    // Pad to fill the width
    int current_pos = std::cout.tellp();
    if (current_pos < width) {
        std::cout << std::string(width - current_pos, ' ');
    }
}

void UserInterface::drawOrderbook() {
    if (bid_levels_.empty() || ask_levels_.empty()) {
        std::cout << CYAN << "Waiting for real-time orderbook data..." << RESET << std::endl;
        return;
    }
    
    // Calculate the number of levels to display (max 10 like in the image)
    const size_t num_levels = std::min(
        std::min(bid_levels_.size(), ask_levels_.size()), 
        static_cast<size_t>(10)
    );
    
    // Draw a line like the depth graph in the image
    std::cout << std::string(15, ' ') << GREEN;
    for (int i = 0; i < 40; i++) {
        std::cout << "⋅";
    }
    std::cout << RESET << " | " << RED;
    for (int i = 0; i < 40; i++) {
        std::cout << "⋅";
    }
    std::cout << RESET << std::endl;
    
    // Draw a line with price markers
    double bid_price = simulator_->getCurrentPrice() - simulator_->getSpread()/2;
    double ask_price = simulator_->getCurrentPrice() + simulator_->getSpread()/2;
    
    std::cout << std::string(5, ' ') << GREEN << std::fixed << std::setprecision(1) << (bid_price - 4);
    std::cout << std::string(30, ' ') << (bid_price) << RESET;
    std::cout << " | " << RED << (ask_price);
    std::cout << std::string(30, ' ') << (ask_price + 4) << RESET << std::endl;
    
    // Draw the orderbook entries
    for (size_t i = 0; i < num_levels; ++i) {
        const auto& bid = bid_levels_[i];
        const auto& ask = ask_levels_[i];
        int bid_bar = bid_bars_[i];
        int ask_bar = ask_bars_[i];
        
        // Format to match the image
        std::cout << GREEN << "OKX  " << bid.first << std::string(5, ' ') << std::setw(2) << std::right;
        
        // Random value for quantity (in real implementation, would come from the exchange)
        int qty = (std::rand() % 100) + 1;
        std::cout << qty << RESET << "  ";
        
        // Ask side
        std::cout << RED << "OKX  " << ask.first << std::string(5, ' ') << std::setw(3) << std::right;
        qty = (std::rand() % 100) + 1;
        std::cout << qty << RESET << std::endl;
    }
}

void UserInterface::drawOrderbookHeader(int width) {
    // Calculate column widths (for a typical orderbook display)
    int price_width = 15;
    int size_width = 10;
    int bar_width = width - (price_width + size_width) * 2 - 3; // -3 for separators
    
    // Print header
    std::cout << BOLD;
    std::cout << GREEN << std::left << std::setw(price_width) << " Bid Price";
    std::cout << WHITE << std::left << std::setw(size_width) << " Size";
    std::cout << std::setw(bar_width / 2) << " ";
    std::cout << " │ ";
    std::cout << RED << std::left << std::setw(price_width) << " Ask Price";
    std::cout << WHITE << std::left << std::setw(size_width) << " Size";
    std::cout << std::setw(bar_width / 2) << " ";
    std::cout << RESET;
}

void UserInterface::drawOrderbookRow(int row, int width) {
    if (bid_levels_.empty() || ask_levels_.empty() || row >= 20) {
        if (row == 0) {
            std::cout << CYAN << " Waiting for real-time orderbook data..." << RESET;
        } else {
            std::cout << std::string(width, ' ');
        }
        return;
    }
    
    // Calculate column widths (for a typical orderbook display)
    int price_width = 15;
    int size_width = 10;
    int bar_width = width - (price_width + size_width) * 2 - 3; // -3 for separators
    
    if (row < bid_levels_.size() && row < ask_levels_.size()) {
        // Bid side
        std::cout << GREEN << std::left << std::setw(price_width) << " " + bid_levels_[row].first;
        std::cout << WHITE << std::left << std::setw(size_width) << " " + bid_levels_[row].second;
        
        // Bid size bar
        if (row < bid_bars_.size()) {
            int actual_width = std::min(bid_bars_[row], bar_width / 2);
            std::cout << GREEN << std::string(actual_width, '|') << RESET;
            std::cout << std::string(bar_width / 2 - actual_width, ' ');
        } else {
            std::cout << std::string(bar_width / 2, ' ');
        }
        
        // Separator
        std::cout << " │ ";
        
        // Ask side
        std::cout << RED << std::left << std::setw(price_width) << " " + ask_levels_[row].first;
        std::cout << WHITE << std::left << std::setw(size_width) << " " + ask_levels_[row].second;
        
        // Ask size bar
        if (row < ask_bars_.size()) {
            int actual_width = std::min(ask_bars_[row], bar_width / 2);
            std::cout << RED << std::string(actual_width, '|') << RESET;
            std::cout << std::string(bar_width / 2 - actual_width, ' ');
        } else {
            std::cout << std::string(bar_width / 2, ' ');
        }
    } else {
        std::cout << std::string(width, ' ');
    }
}

void UserInterface::updateOrderbookData() {
    if (!simulator_ || simulator_->getCurrentPrice() <= 0.0) {
        return;
    }
    
    // Lock while updating the orderbook data
    std::lock_guard<std::mutex> lock(ui_mutex_);
    
    // Clear previous data
    bid_levels_.clear();
    ask_levels_.clear();
    bid_bars_.clear();
    ask_bars_.clear();
    
    // Get values from simulator for orderbook visualization
    double midPrice = simulator_->getCurrentPrice();
    double spread = simulator_->getSpread();
    double bestBid = midPrice - spread/2;
    double bestAsk = midPrice + spread/2;
    double depth = simulator_->getOrderbookDepth();
    
    // Track the maximum size for normalization
    double max_bid_size = 0.0;
    double max_ask_size = 0.0;
    
    // Calculate all sizes first to find the maximum
    std::vector<double> bid_sizes;
    std::vector<double> ask_sizes;
    
    // Generate order book levels (20 levels)
    for (int i = 0; i < 20; i++) {
        // Calculate prices with 0.05% steps
        double bidPrice = bestBid * (1.0 - 0.0005 * i);
        double askPrice = bestAsk * (1.0 + 0.0005 * i);
        
        // Exponentially decreasing size
        double bidSize = depth * std::exp(-0.5 * i) / 10.0;
        double askSize = depth * std::exp(-0.3 * i) / 12.0;
        
        // Add some randomness to make it look more realistic
        bidSize *= (0.85 + (std::rand() % 30) / 100.0);
        askSize *= (0.85 + (std::rand() % 30) / 100.0);
        
        // Ensure we have positive values
        bidSize = std::max(0.01, bidSize);
        askSize = std::max(0.01, askSize);
        
        // Track maximum sizes
        max_bid_size = std::max(max_bid_size, bidSize);
        max_ask_size = std::max(max_ask_size, askSize);
        
        // Store the sizes
        bid_sizes.push_back(bidSize);
        ask_sizes.push_back(askSize);
        
        // Format prices for display
        std::stringstream ss;
        ss << std::fixed << std::setprecision(2);
        
        ss.str("");
        ss << bidPrice;
        std::string bidPriceStr = ss.str();
        
        ss.str("");
        ss << askPrice;
        std::string askPriceStr = ss.str();
        
        // Format sizes for display
        ss.str("");
        ss << std::fixed << std::setprecision(2) << bidSize;
        std::string bidSizeStr = ss.str();
        
        ss.str("");
        ss << std::fixed << std::setprecision(2) << askSize;
        std::string askSizeStr = ss.str();
        
        // Add to level vectors
        bid_levels_.emplace_back(bidPriceStr, bidSizeStr);
        ask_levels_.emplace_back(askPriceStr, askSizeStr);
    }
    
    // Now normalize and calculate the bar sizes (max 30 chars)
    const int max_bar_width = 30;
    
    for (size_t i = 0; i < bid_sizes.size(); ++i) {
        // Normalize to a percentage of the maximum, then scale to max bar width
        int barWidth = static_cast<int>((bid_sizes[i] / max_bid_size) * max_bar_width);
        bid_bars_.push_back(barWidth);
    }
    
    for (size_t i = 0; i < ask_sizes.size(); ++i) {
        int barWidth = static_cast<int>((ask_sizes[i] / max_ask_size) * max_bar_width);
        ask_bars_.push_back(barWidth);
    }
}

void UserInterface::drawSeparator(int width, char character) {
    std::cout << std::string(width, character) << std::endl;
}

void UserInterface::drawBoxTitle(const std::string& title, int width) {
    int padding = (width - title.length() - 2) / 2;
    std::cout << "┌" << std::string(padding, '─') << " " << title << " ";
    int remaining = width - padding - title.length() - 3;
    std::cout << std::string(remaining, '─') << "┐" << std::endl;
}

void UserInterface::setExchangeCallback(std::function<void(const std::string&)> callback) {
    exchange_callback_ = callback;
}

void UserInterface::setSymbolCallback(std::function<void(const std::string&)> callback) {
    symbol_callback_ = callback;
}

void UserInterface::setOrderTypeCallback(std::function<void(const std::string&)> callback) {
    order_type_callback_ = callback;
}

void UserInterface::setQuantityCallback(std::function<void(double)> callback) {
    quantity_callback_ = callback;
}

void UserInterface::setVolatilityCallback(std::function<void(double)> callback) {
    volatility_callback_ = callback;
}

void UserInterface::setFeeTierCallback(std::function<void(int)> callback) {
    fee_tier_callback_ = callback;
} 