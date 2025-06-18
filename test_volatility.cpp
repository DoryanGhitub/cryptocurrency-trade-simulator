#include <iostream>
#include <memory>
#include <string>
#include "./include/TradeSimulator.h"
#include "./ui/UserInterface.h"

int main() {
    std::cout << "Volatility Test Program" << std::endl;
    
    // Create simulator with default volatility (0.01)
    auto simulator = std::make_shared<TradeSimulator>();
    
    std::cout << "Default volatility in simulator: " << simulator->getVolatility() << std::endl;
    
    // Create UI with simulator
    UserInterface ui(simulator);
    
    // Check initial values
    std::cout << "Initial UI volatility: " << ui.getVolatility() << std::endl;
    
    // Test with different volatility values
    simulator->setParameters("OKX", "BTC-USDT", "market", 100.0, 0.02, 1);
    std::cout << "After setParameters(0.02), simulator volatility: " << simulator->getVolatility() << std::endl;
    
    // Force UI update
    ui.updateInputParameters();
    std::cout << "After updateInputParameters(), UI volatility: " << ui.getVolatility() << std::endl;
    
    return 0;
} 