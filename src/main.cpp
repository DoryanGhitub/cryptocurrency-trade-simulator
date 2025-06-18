#include <iostream>
#include <memory>
#include <string>
#include <csignal>
#include <chrono>
#include <thread>
#include <fstream>
#include <nlohmann/json.hpp>
#include "../include/TradeSimulator.h"
#include "../ui/UserInterface.h"

std::shared_ptr<TradeSimulator> g_simulator;
std::unique_ptr<UserInterface> g_ui;
bool g_running = true;
bool g_performance_test = false;

void signalHandler(int signal) {
    if (signal == SIGINT || signal == SIGTERM) {
        g_running = false;
        if (g_ui) {
            g_ui->stop();
        }
        if (g_simulator) {
            g_simulator->disconnect();
        }
        std::cout << "Application shutting down..." << std::endl;
    }
}

// Helper function to load configuration from JSON file
nlohmann::json loadConfig(const std::string& filename) {
    try {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Could not open config file: " << filename << std::endl;
            return nlohmann::json();
        }
        
        nlohmann::json config;
        file >> config;
        return config;
    } catch (const std::exception& e) {
        std::cerr << "Error loading config: " << e.what() << std::endl;
        return nlohmann::json();
    }
}

void printHelp(const char* programName) {
    std::cout << "Usage: " << programName << " [OPTIONS]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --url URL               WebSocket URL for L2 orderbook data" << std::endl;
    std::cout << "  --exchange EXCHANGE     Exchange name (default: OKX)" << std::endl;
    std::cout << "  --symbol SYMBOL         Asset symbol (default: BTC-USDT-SWAP)" << std::endl;
    std::cout << "  --order-type TYPE       Order type (default: market)" << std::endl;
    std::cout << "  --quantity QTY          Quantity to trade (default: 1.0)" << std::endl;
    std::cout << "  --volatility VOL        Asset volatility (default: 0.01)" << std::endl;
    std::cout << "  --fee-tier TIER         Fee tier level (default: 1)" << std::endl;
    std::cout << "  --config FILE           Configuration file (default: resources/config.json)" << std::endl;
    std::cout << "  --performance-test      Run performance test with high frequency updates" << std::endl;
    std::cout << "  --help                  Show this help message" << std::endl;
}

int main(int argc, char* argv[]) {
    std::cout << "Trade Simulator Starting..." << std::endl;
    
    // Register signal handler
    std::signal(SIGINT, signalHandler);
    std::signal(SIGTERM, signalHandler);
    
    // Default values
    std::string websocket_url = "wss://ws.gomarket-cpp.goquant.io/ws/l2-orderbook/okx/BTC-USDT-SWAP";
    std::string exchange = "OKX";
    std::string symbol = "BTC-USDT-SWAP";
    std::string order_type = "market";
    double quantity = 100.0;
    double volatility = 0.01;
    int fee_tier = 1;
    bool test_mode = false;
    std::string config_file = "resources/config.json";
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--help") {
            printHelp(argv[0]);
            return 0;
        } else if (arg == "--url" && i + 1 < argc) {
            websocket_url = argv[++i];
        } else if (arg == "--exchange" && i + 1 < argc) {
            exchange = argv[++i];
        } else if (arg == "--symbol" && i + 1 < argc) {
            symbol = argv[++i];
        } else if (arg == "--order-type" && i + 1 < argc) {
            order_type = argv[++i];
        } else if (arg == "--quantity" && i + 1 < argc) {
            quantity = std::stod(argv[++i]);
        } else if (arg == "--volatility" && i + 1 < argc) {
            volatility = std::stod(argv[++i]);
        } else if (arg == "--fee-tier" && i + 1 < argc) {
            fee_tier = std::stoi(argv[++i]);
        } else if (arg == "--config" && i + 1 < argc) {
            config_file = argv[++i];
        } else if (arg == "--performance-test") {
            g_performance_test = true;
        } else if (arg == "--test-mode") {
            test_mode = true;
        }
    }
    
    // Load configuration from file
    auto config = loadConfig(config_file);
    if (!config.empty()) {
        if (config.contains("websocket") && config["websocket"].contains("url")) {
            websocket_url = config["websocket"]["url"];
        }
        if (config.contains("simulator")) {
            if (config["simulator"].contains("exchange")) {
                exchange = config["simulator"]["exchange"];
            }
            if (config["simulator"].contains("symbol")) {
                symbol = config["simulator"]["symbol"];
            }
            if (config["simulator"].contains("order_type")) {
                order_type = config["simulator"]["order_type"];
            }
            if (config["simulator"].contains("quantity")) {
                quantity = config["simulator"]["quantity"];
            }
            if (config["simulator"].contains("volatility")) {
                volatility = config["simulator"]["volatility"];
            }
            if (config["simulator"].contains("fee_tier")) {
                fee_tier = config["simulator"]["fee_tier"];
            }
        }
    }
    
    // Initialize the simulator
    g_simulator = std::make_shared<TradeSimulator>();
    g_simulator->setParameters(exchange, symbol, order_type, quantity, volatility, fee_tier);
    
    // std::cout << "DEBUG: Using parameters - exchange=" << exchange 
    //           << ", symbol=" << symbol
    //           << ", order_type=" << order_type 
    //           << ", quantity=" << quantity
    //           << ", volatility=" << volatility
    //           << ", fee_tier=" << fee_tier << std::endl;
    
    // Initialize and connect
    if (!g_simulator->initialize(websocket_url)) {
        std::cerr << "Failed to initialize the simulator" << std::endl;
        return 1;
    }
    
    // Create and initialize the UI
    std::cout << "Initializing UI..." << std::endl;
    g_ui = std::make_unique<UserInterface>(g_simulator);
    g_ui->initialize();
    
    // Connect to the WebSocket
    std::cout << "Connecting to WebSocket..." << std::endl;
    if (!g_simulator->connect()) {
        std::cerr << "Failed to connect to WebSocket" << std::endl;
        return 1;
    }
    std::cout << "WebSocket connection established successfully" << std::endl;
    
    // Run the UI
    g_ui->run();
    
    // Performance test mode
    if (g_performance_test) {
        std::cout << "Running performance test..." << std::endl;
        auto start_time = std::chrono::high_resolution_clock::now();
        std::this_thread::sleep_for(std::chrono::seconds(10));
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        
        // Report performance metrics
        int64_t avg_latency = g_simulator->getInternalLatency();
        std::cout << "Performance Test Results:" << std::endl;
        std::cout << "  Test Duration: " << duration << " ms" << std::endl;
        std::cout << "  Average Processing Latency: " << avg_latency << " μs" << std::endl;
        
        g_running = false;
    }
    
    // If in test mode, run for 10 seconds and exit
    if (test_mode) {
        std::cout << "Running in test mode. Will exit after 10 seconds..." << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(10));
        g_ui->stop();
        g_simulator->disconnect();
        return 0;
    }
    
    // Wait for the UI to finish
    while (g_ui->isRunning() && g_running) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    // Clean up
    g_ui.reset();
    g_simulator.reset();
    
    std::cout << "Trade Simulator Terminated" << std::endl;
    return 0;
} 