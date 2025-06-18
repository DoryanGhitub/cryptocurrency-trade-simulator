#include "../include/TradeSimulator.h"
#include <iostream>
#include <chrono>
#include <thread>
#include <algorithm>
#include <functional>

TradeSimulator::TradeSimulator()
    : exchange_("OKX"),
      symbol_("BTC-USDT-SWAP"),
      order_type_("market"),
      quantity_(100.0),
      volatility_(0.01),
      fee_tier_(1),
      running_(false),
      expected_slippage_(0.0),
      expected_fees_(0.0),
      expected_market_impact_(0.0),
      maker_taker_proportion_(0.1),
      confidence_lower_(0.0),
      confidence_upper_(1.0),
      internal_latency_(0),
      ai_inference_time_(0.0)
{
    // std::cout << "TradeSimulator: Initialized with symbol " << symbol_ << " and volatility " << volatility_ << std::endl;
}

TradeSimulator::~TradeSimulator() {
    disconnect();
}

bool TradeSimulator::initialize(const std::string& websocketUrl) {
    // Initialize the components
    try {
        orderbook_ = std::make_unique<OrderBook>(symbol_);
        websocket_client_ = std::make_unique<WebSocketClient>(websocketUrl);
        
        // Initialize Almgren-Chriss model with default parameters
        double temporary_impact_factor = 0.1;
        double permanent_impact_factor = 0.1;
        double risk_aversion = 1.0;
        almgren_chriss_model_ = std::make_unique<AlmgrenChriss>(
            volatility_, temporary_impact_factor, permanent_impact_factor, risk_aversion);
        
        // Initialize regression models
        slippage_model_ = std::make_unique<LinearRegression>();
        
        // Initialize advanced GPU-accelerated transformer model
        advanced_ai_model_ = std::make_unique<AdvancedTransformerModel>(
            AdvancedTransformerModel::ModelType::TEMPORAL_FUSION_TRANSFORMER,
            AdvancedTransformerModel::ModelBackend::CUDA_NATIVE,
            16,   // input dimensions (enhanced features)
            256,  // hidden dimensions (larger for better performance)
            16,   // attention heads
            12,   // transformer layers
            128,  // sequence length
            true  // use GPU acceleration
        );
        
        // Initialize the AI model
        if (!advanced_ai_model_->initialize()) {
            std::cerr << "Warning: Advanced AI model initialization failed, using fallback" << std::endl;
        } else {
            std::cout << "Advanced GPU-accelerated AI model initialized successfully" << std::endl;
            std::cout << advanced_ai_model_->getModelInfo() << std::endl;
        }
        
        // Initialize fee model
        fee_model_ = std::make_unique<FeeModel>(exchange_);
        
        // Set the WebSocket message callback
        websocket_client_->setMessageCallback(
            std::bind(&TradeSimulator::processMessage, this, std::placeholders::_1));
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "TradeSimulator initialization error: " << e.what() << std::endl;
        return false;
    }
}

bool TradeSimulator::connect() {
    if (!websocket_client_) {
        return false;
    }
    
    running_ = true;
    bool connected = websocket_client_->connect();
    
    if (connected) {
        // Send explicit subscription request for OKX
        nlohmann::json subscription = {
            {"op", "subscribe"},
            {"args", {
                {
                    {"channel", "books"},
                    {"instId", "BTC-USDT"}
                }
            }}
        };
        
        std::string subMessage = subscription.dump();
        // std::cout << "TradeSimulator: Sending OKX subscription: " << subMessage << std::endl;
        websocket_client_->sendMessage(subMessage);
    }
    
    return connected;
}

void TradeSimulator::disconnect() {
    running_ = false;
    
    if (websocket_client_) {
        websocket_client_->disconnect();
    }
}

bool TradeSimulator::isConnected() const {
    if (!websocket_client_) {
        return false;
    }
    
    return websocket_client_->isConnected();
}

void TradeSimulator::setParameters(const std::string& exchange, const std::string& symbol,
                               const std::string& orderType, double quantity,
                               double volatility, int feeTier) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    
    exchange_ = exchange;
    symbol_ = symbol;
    order_type_ = orderType;
    quantity_ = quantity;
    volatility_ = volatility;
    fee_tier_ = feeTier;
    
    // Update models with new parameters
    if (almgren_chriss_model_) {
        almgren_chriss_model_->setVolatility(volatility);
    }
    
    if (fee_model_) {
        fee_model_->setExchange(exchange);
    }
}

void TradeSimulator::processMessage(const nlohmann::json& message) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    try {
        // Create a local copy of the data to work with
        nlohmann::json message_copy = message;
        
        // Update the orderbook with the received data
        if (orderbook_) {
            orderbook_->updateFromJson(message_copy);
        }
        
        // Calculate output parameters based on the new data
        // Use a separate mutex lock to avoid deadlock
        calculateOutputParameters();
        
        auto end_time = std::chrono::high_resolution_clock::now();
        
        // Only lock when updating the latency value
        {
            std::lock_guard<std::mutex> lock(data_mutex_);
            internal_latency_ = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
        }
    } catch (const std::exception& e) {
        std::cerr << "Error processing message: " << e.what() << std::endl;
    }
}

void TradeSimulator::calculateOutputParameters() {
    if (!orderbook_ || !almgren_chriss_model_ || !slippage_model_ || !advanced_ai_model_ || !fee_model_) {
        std::cerr << "TradeSimulator: Missing required components for calculation" << std::endl;
        return;
    }
    
    // Get current market data without holding a lock
    double mid_price = orderbook_->getMidPrice();
    double spread = orderbook_->getSpread();
    double spread_percentage = orderbook_->getSpreadPercentage();
    std::string timestamp = orderbook_->getTimestamp();
    
    if (mid_price <= 0.0) {
        std::cerr << "TradeSimulator: Invalid mid price, cannot calculate parameters" << std::endl;
        return;
    }
    
    // Copy the necessary parameters to local variables to avoid holding locks
    double quantity, volatility;
    std::string order_type, exchange, symbol;
    int fee_tier;
    
    {
        std::lock_guard<std::mutex> lock(data_mutex_);
        quantity = quantity_;
        volatility = volatility_;
        order_type = order_type_;
        exchange = exchange_;
        fee_tier = fee_tier_;
        symbol = symbol_;
    }
    
    // Calculate the USD equivalent quantity
    double usd_quantity = quantity;
    if (symbol.find("-") != std::string::npos) {
        // If symbol is in format BTC-USDT, we need to convert
        usd_quantity = quantity * mid_price;
    }
    
    // Calculate expected slippage (simple model: half the spread)
    double calculated_slippage = (order_type == "market") ? spread_percentage / 2.0 : 0.0;
    
    // Extract market features for transformer model
    double bid_depth = orderbook_->getDepth("bids", 0.005);
    double ask_depth = orderbook_->getDepth("asks", 0.005);
    double total_depth = bid_depth + ask_depth;
    double imbalance = total_depth > 0 ? (bid_depth - ask_depth) / total_depth : 0.0;
    double book_pressure = orderbook_->getBookPressure(5);
    
    // Create feature vector for transformer model
    std::vector<double> market_features = {
        mid_price,
        spread_percentage,
        volatility,
        bid_depth,
        ask_depth,
        imbalance,
        book_pressure,
        usd_quantity / total_depth  // normalized quantity
    };
    
    // Use advanced AI model to predict maker/taker proportion with confidence intervals
    double calculated_maker_taker = 0.3; // Default value
    if (advanced_ai_model_) {
        try {
            auto start_time = std::chrono::high_resolution_clock::now();
            
            auto prediction_result = advanced_ai_model_->predictMakerProbability(market_features, order_type, 0.95);
            calculated_maker_taker = prediction_result.maker_probability;
            confidence_lower_ = prediction_result.confidence_lower;
            confidence_upper_ = prediction_result.confidence_upper;
            
            auto end_time = std::chrono::high_resolution_clock::now();
            ai_inference_time_ = std::chrono::duration<double, std::milli>(end_time - start_time).count();
            
        } catch (const std::exception& e) {
            std::cerr << "Error in AI maker/taker prediction: " << e.what() << std::endl;
            confidence_lower_ = 0.0;
            confidence_upper_ = 1.0;
            ai_inference_time_ = 0.0;
        }
    } else {
        std::cerr << "Warning: advanced_ai_model is null" << std::endl;
        confidence_lower_ = 0.0;
        confidence_upper_ = 1.0;
        ai_inference_time_ = 0.0;
    }
    
    // Calculate expected fees based on maker/taker proportion
    double maker_fee_rate = fee_model_->getMakerFeeRate(fee_tier);
    double taker_fee_rate = fee_model_->getTakerFeeRate(fee_tier);
    double calculated_fees = (calculated_maker_taker * maker_fee_rate + 
                             (1.0 - calculated_maker_taker) * taker_fee_rate) * usd_quantity / 100.0;
    
    // Calculate expected market impact using Almgren-Chriss model
    // Adjust market impact based on maker/taker proportion
    // Maker orders have less impact than taker orders
    double maker_impact_factor = 0.2;  // Maker orders have 20% of the impact of taker orders
    double taker_impact_factor = 1.0;
    double impact_factor = calculated_maker_taker * maker_impact_factor + (1.0 - calculated_maker_taker) * taker_impact_factor;
    
    // Calculate final market impact
    double calculated_market_impact = almgren_chriss_model_->calculateMarketImpact(
        usd_quantity, spread_percentage, volatility) * impact_factor;
    
    // Update the output parameters with a lock
    {
        std::lock_guard<std::mutex> lock(data_mutex_);
        expected_slippage_ = calculated_slippage;
        expected_fees_ = calculated_fees;
        expected_market_impact_ = calculated_market_impact;
        maker_taker_proportion_ = calculated_maker_taker;
    }
}

double TradeSimulator::getExpectedSlippage() const {
    std::lock_guard<std::mutex> lock(data_mutex_);
    return expected_slippage_;
}

double TradeSimulator::getExpectedFees() const {
    std::lock_guard<std::mutex> lock(data_mutex_);
    return expected_fees_;
}

double TradeSimulator::getExpectedMarketImpact() const {
    std::lock_guard<std::mutex> lock(data_mutex_);
    return expected_market_impact_;
}

double TradeSimulator::getNetCost() const {
    std::lock_guard<std::mutex> lock(data_mutex_);
    return expected_slippage_ + expected_fees_ + expected_market_impact_;
}

bool TradeSimulator::loadAIModel(const std::string& model_path) {
    if (!advanced_ai_model_) {
        std::cerr << "Advanced AI model not initialized" << std::endl;
        return false;
    }
    
    bool success = advanced_ai_model_->loadModel(model_path);
    if (success) {
        std::cout << "Successfully loaded AI model: " << model_path << std::endl;
    } else {
        std::cerr << "Failed to load AI model: " << model_path << std::endl;
    }
    
    return success;
}

AdvancedTransformerModel::ModelMetrics TradeSimulator::getAIModelMetrics() const {
    if (advanced_ai_model_) {
        return advanced_ai_model_->getModelMetrics();
    }
    
    // Return default metrics if model not available
    AdvancedTransformerModel::ModelMetrics default_metrics;
    default_metrics.accuracy = 0.0;
    default_metrics.precision = 0.0;
    default_metrics.recall = 0.0;
    default_metrics.f1_score = 0.0;
    default_metrics.avg_inference_time_ms = 0.0;
    default_metrics.gpu_memory_usage_mb = 0.0;
    
    return default_metrics;
}

double TradeSimulator::getMakerTakerProportion() const {
    std::lock_guard<std::mutex> lock(data_mutex_);
    return maker_taker_proportion_;
}

std::pair<double, double> TradeSimulator::getMakerTakerConfidence() const {
    std::lock_guard<std::mutex> lock(data_mutex_);
    return {confidence_lower_, confidence_upper_};
}

double TradeSimulator::getAIInferenceTime() const {
    std::lock_guard<std::mutex> lock(data_mutex_);
    return ai_inference_time_;
}

double TradeSimulator::getGpuMemoryUsage() const {
    if (advanced_ai_model_) {
        return advanced_ai_model_->getModelMetrics().gpu_memory_usage_mb;
    }
    return 0.0;
}

int64_t TradeSimulator::getInternalLatency() const {
    std::lock_guard<std::mutex> lock(data_mutex_);
    return internal_latency_;
}

double TradeSimulator::getCurrentPrice() const {
    // Don't lock here as we're just calling into OrderBook
    if (!orderbook_) {
        return 0.0;
    }
    return orderbook_->getMidPrice();
}

double TradeSimulator::getSpread() const {
    // Don't lock here as we're just calling into OrderBook
    if (!orderbook_) {
        return 0.0;
    }
    return orderbook_->getSpread();
}

double TradeSimulator::getOrderbookDepth() const {
    // Don't lock here as we're just calling into OrderBook
    if (!orderbook_) {
        return 0.0;
    }
    // Get depth at 1% level
    return orderbook_->getDepth("bids", 0.01) + orderbook_->getDepth("asks", 0.01);
}

std::string TradeSimulator::getLastUpdateTimestamp() const {
    // Don't lock here as we're just calling into OrderBook
    if (!orderbook_) {
        return "";
    }
    return orderbook_->getTimestamp();
}

std::string TradeSimulator::getExchange() const {
    std::lock_guard<std::mutex> lock(data_mutex_);
    return exchange_;
}

std::string TradeSimulator::getSymbol() const {
    std::lock_guard<std::mutex> lock(data_mutex_);
    return symbol_;
}

std::string TradeSimulator::getOrderType() const {
    std::lock_guard<std::mutex> lock(data_mutex_);
    return order_type_;
}

double TradeSimulator::getQuantity() const {
    std::lock_guard<std::mutex> lock(data_mutex_);
    return quantity_;
}

double TradeSimulator::getVolatility() const {
    std::lock_guard<std::mutex> lock(data_mutex_);
    // std::cout << "DEBUG: TradeSimulator::getVolatility() returning " << volatility_ << std::endl;
    return volatility_;
}

int TradeSimulator::getFeeTier() const {
    std::lock_guard<std::mutex> lock(data_mutex_);
    return fee_tier_;
} 