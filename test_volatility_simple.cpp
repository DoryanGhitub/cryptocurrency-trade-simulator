#include <iostream>
#include <iomanip>

class TestSimulator {
private:
    double volatility_;
    
public:
    TestSimulator() : volatility_(0.01) {
        std::cout << "TestSimulator constructor, volatility_=" << std::fixed << std::setprecision(6) << volatility_ << std::endl;
    }
    
    void setVolatility(double volatility) {
        std::cout << "TestSimulator::setVolatility(" << std::fixed << std::setprecision(6) << volatility << ")" << std::endl;
        volatility_ = volatility;
    }
    
    double getVolatility() const {
        std::cout << "TestSimulator::getVolatility() returning " << std::fixed << std::setprecision(6) << volatility_ << std::endl;
        return volatility_;
    }
};

class TestUI {
private:
    double selected_volatility_;
    TestSimulator& simulator_;
    
public:
    TestUI(TestSimulator& simulator) 
        : simulator_(simulator), 
          selected_volatility_(0.01) {
        std::cout << "TestUI constructor, selected_volatility_=" << std::fixed << std::setprecision(6) << selected_volatility_ << std::endl;
    }
    
    void updateParameters() {
        std::cout << "TestUI::updateParameters() called" << std::endl;
        std::cout << "Before update: selected_volatility_=" << std::fixed << std::setprecision(6) << selected_volatility_ << std::endl;
        
        double vol = simulator_.getVolatility();
        selected_volatility_ = vol;
        
        std::cout << "After update: selected_volatility_=" << std::fixed << std::setprecision(6) << selected_volatility_ << std::endl;
    }
    
    void display() {
        std::cout << "UI Display:" << std::endl;
        std::cout << "  Volatility: " << std::fixed << std::setprecision(6) << selected_volatility_ << std::endl;
    }
};

int main() {
    std::cout << "== Volatility Simple Test ==" << std::endl;
    
    TestSimulator simulator;
    TestUI ui(simulator);
    
    // Display initial values
    ui.display();
    
    // Change simulator volatility
    simulator.setVolatility(0.02);
    
    // Display values without update
    ui.display();
    
    // Update UI from simulator
    ui.updateParameters();
    
    // Display updated values
    ui.display();
    
    return 0;
} 