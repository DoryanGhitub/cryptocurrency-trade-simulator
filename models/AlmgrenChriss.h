#pragma once

#include <vector>
#include <cmath>
#include <chrono>
#include <algorithm>

/**
 * @brief Implementation of the Almgren-Chriss model for optimal execution
 * 
 * This model provides a mathematical approach to executing large trades optimally
 * by balancing the trade-off between market impact and execution risk.
 */
class AlmgrenChriss {
public:
    /**
     * @brief Constructor with parameters
     * @param volatility Asset volatility
     * @param temporaryImpactFactor Temporary impact coefficient (gamma)
     * @param permanentImpactFactor Permanent impact coefficient (eta)
     * @param riskAversion Risk aversion parameter
     */
    AlmgrenChriss(double volatility, double temporaryImpactFactor, double permanentImpactFactor, double riskAversion);
    
    /**
     * @brief Calculate optimal execution schedule
     * @param totalQuantity Total quantity to execute
     * @param timeHorizon Total time for execution (in seconds)
     * @param numPeriods Number of trading periods
     * @return Vector of quantities to execute in each period
     */
    std::vector<double> calculateOptimalTrajectory(double totalQuantity, double timeHorizon, int numPeriods);
    
    /**
     * @brief Calculate the Hamiltonian function for the dynamic programming approach
     * @param inventory Current inventory level
     * @param sellAmount Amount to sell in the current period
     * @param remainingInventory Remaining inventory after selling
     * @param timeStep Time step size
     * @return Value of the Hamiltonian
     */
    double hamiltonian(int inventory, int sellAmount, int remainingInventory, double timeStep);
    
    /**
     * @brief Estimate market impact for a given volume
     * @param quantity The quantity to execute
     * @param price Current price
     * @param timeHorizon Time horizon for execution (in seconds)
     * @return Estimated market impact
     */
    double estimateMarketImpact(double quantity, double price, double timeHorizon);
    
    /**
     * @brief Calculate total execution cost
     * @param quantity The quantity to execute
     * @param price Current price
     * @param timeHorizon Time horizon for execution (in seconds)
     * @return Estimated total execution cost
     */
    double calculateExecutionCost(double quantity, double price, double timeHorizon);
    
    /**
     * @brief Calculate temporary impact component
     * @param rate Execution rate (quantity per unit time)
     * @return Temporary impact value
     */
    double calculateTemporaryImpact(double rate) const;
    
    /**
     * @brief Calculate permanent impact component
     * @param quantity Total quantity executed
     * @return Permanent impact value
     */
    double calculatePermanentImpact(double quantity) const;
    
    /**
     * @brief Calculate market impact as a percentage based on BTC-specific model
     * @param quantity The quantity to execute (in USD)
     * @param spreadPercentage Current spread percentage 
     * @param volatility Current volatility estimate
     * @return Estimated market impact as a percentage
     */
    double calculateMarketImpact(double quantity, double spreadPercentage, double volatility);
    
    // Getters and setters
    double getVolatility() const { return volatility_; }
    void setVolatility(double volatility) { volatility_ = volatility; }
    
    double getTemporaryImpactFactor() const { return temporary_impact_factor_; }
    void setTemporaryImpactFactor(double factor) { temporary_impact_factor_ = factor; }
    
    double getPermanentImpactFactor() const { return permanent_impact_factor_; }
    void setPermanentImpactFactor(double factor) { permanent_impact_factor_ = factor; }
    
    double getRiskAversion() const { return risk_aversion_; }
    void setRiskAversion(double aversion) { risk_aversion_ = aversion; }
    
private:
    double volatility_;                // Asset volatility
    double temporary_impact_factor_;   // Temporary impact coefficient (gamma)
    double permanent_impact_factor_;   // Permanent impact coefficient (eta)
    double risk_aversion_;             // Risk aversion parameter
}; 