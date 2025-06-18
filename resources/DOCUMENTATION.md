# Trade Simulator: Detailed Documentation

## Architecture Overview

The trade simulator is built with a modular architecture to ensure separation of concerns, maintainability, and extensibility. The main components are:

1. **OrderBook**: Processes L2 orderbook data, calculates market metrics
2. **WebSocketClient**: Handles the WebSocket connection and message parsing
3. **Models**: 
   - AlmgrenChriss: Estimates market impact
   - RegressionModels: Predicts slippage and maker/taker proportions
   - FeeModel: Calculates exchange fees
4. **TradeSimulator**: Coordinates all components
5. **UserInterface**: Handles UI rendering and user interactions

## Algorithm Implementation Details

### Model Selection and Parameters

The choice of models for different components was based on:

1. **Market Impact Model**: Almgren-Chriss was selected because it provides a robust theoretical framework that balances execution costs against price risk. The model is widely used in algorithmic trading and provides a balance between complexity and practicality.

   Key parameters:
   - Volatility (σ): Measures price volatility over the execution horizon
   - Market depth: Derived from the orderbook to estimate liquidity
   - Temporary impact factor (γ): Calibrated from historical data
   - Permanent impact factor (η): Calibrated from market observations
   - Risk aversion parameter (λ): Controls the trade-off between execution cost and risk

2. **Slippage Estimation**: We implemented both linear and quantile regression models.
   
   Linear regression was chosen for its:
   - Computational efficiency
   - Interpretability of coefficients
   - Ability to capture linear relationships between order size and slippage

   Quantile regression was implemented as an alternative that:
   - Provides more robust estimates in the presence of outliers
   - Can model different percentiles of slippage distribution
   - Better represents tail risk in execution

3. **Maker/Taker Proportion**: Logistic regression was selected because:
   - The prediction target is binary/proportional in nature
   - The model provides probabilistic outputs
   - It handles the nonlinear relationship between features and maker/taker classification
   - Computational efficiency allows for real-time updates

### Almgren-Chriss Model

The Almgren-Chriss model balances market impact and execution risk. 

#### Market Impact Components:

1. **Temporary Impact**: The immediate price change caused by executing part of the order.
   ```
   Temporary Impact = γ * (quantity / timeHorizon)
   ```
   where γ is the temporary impact factor.

2. **Permanent Impact**: The lasting change in the asset's price due to the execution.
   ```
   Permanent Impact = η * quantity
   ```
   where η is the permanent impact factor.

#### Execution Risk:

Execution risk arises from the uncertainty in price movements during the execution period.
```
Risk Cost = 0.5 * risk_aversion * volatility^2 * quantity^2 * timeHorizon
```

#### Optimal Execution Trajectory:

The model provides an analytical solution for the optimal execution strategy:
```
x_k = x_0 * (sinh(κ * (T - t_k)) / sinh(κ * T))
```
where κ = sqrt(risk_aversion * volatility^2 / (2 * temporary_impact_factor))

#### Market Impact Calculation Methodology

Our implementation of the Almgren-Chriss model follows these steps:

1. **Parameter Calibration**:
   - Volatility (σ) is calculated using an exponentially weighted moving average of returns
   - Market depth is estimated from the current L2 orderbook
   - Impact factors are calibrated using historical trade data and orderbook snapshots

2. **Market Impact Estimation**:
   - For each order size, we simulate walking the orderbook to determine immediate price impact
   - The temporary impact is calculated as the difference between VWAP execution price and mid price
   - Permanent impact is estimated based on historical post-trade price movements

3. **Risk Adjustment**:
   - The total impact is adjusted based on current market conditions
   - Volatility scaling ensures the model adapts to changing market regimes
   - Liquidity adjustments account for order book depth variations

4. **Model Integration**:
   - Impact estimates are continuously updated with each new orderbook tick
   - Time-varying parameters adjust to market conditions in real-time
   - Confidence intervals are computed to provide uncertainty measures

### Regression Models

#### Linear Regression for Slippage Estimation

The linear regression model is implemented using the normal equation:
```
β = (X^T X)^(-1) X^T y
```
where X is the feature matrix and y is the target vector.

Our feature set includes:
- Order size relative to available liquidity
- Bid-ask spread
- Order book imbalance
- Recent price volatility
- Time of day factors

The model is retrained periodically with recent market data to ensure accuracy.

#### Quantile Regression for Slippage Estimation

Quantile regression minimizes the quantile loss function:
```
L(y, ŷ) = Σ ρ_τ(y - ŷ)
```
where ρ_τ(u) = u * (τ - 1(u < 0)) and τ is the quantile.

Our implementation:
- Uses the 95th percentile for conservative risk estimation
- Employs an interior point method for optimization
- Regularizes the model to prevent overfitting
- Updates coefficients using a stochastic approach for computational efficiency

#### Logistic Regression for Maker/Taker Proportion

Logistic regression uses the sigmoid function to model probabilities:
```
P(y=1|x) = 1 / (1 + e^(-β^T x))
```
and is trained using gradient descent to minimize the log loss.

The maker/taker classification model:
- Takes order book structure features as input
- Uses L1 regularization for feature selection
- Employs adaptive learning rates for faster convergence
- Outputs probability of execution as maker vs. taker

### Fee Model

The fee model calculates fees based on:
- Exchange fee tiers
- Order type (market or limit)
- Maker/taker proportion

For OKX, the fee structure is as follows:
- VIP 0: 0.0008 (maker) / 0.0010 (taker)
- VIP 1: 0.0006 (maker) / 0.0008 (taker)
- VIP 2: 0.0004 (maker) / 0.0006 (taker)
- VIP 3: 0.0002 (maker) / 0.0005 (taker)
- VIP 4: 0.0000 (maker) / 0.0003 (taker)
- VIP 5: -0.0001 (maker) / 0.0003 (taker)

## Performance Optimization Approaches

Our system employs several key optimization techniques to ensure high performance:

### Memory Management

1. **Pre-allocated Buffers**: 
   - The WebSocket client uses pre-allocated buffers sized to typical message lengths
   - This avoids frequent memory allocations/deallocations during high message throughput
   - Buffer size is dynamically adjusted based on message size statistics

2. **Efficient Data Structures**: 
   - The OrderBook uses sorted vectors for efficient price level access
   - Price levels are indexed using binary search for O(log n) lookups
   - Custom memory pools for high-frequency small allocations
   - Use of reserve() for vectors to prevent reallocation during updates

3. **Zero-copy Processing**:
   - Message parsing avoids unnecessary data copying when possible
   - Views into the original data are used instead of creating new strings
   - In-place updates are performed where appropriate

### Thread Management

1. **Lock-Free Algorithms**:
   - Critical sections use atomic operations where possible
   - Lock-free queues for message passing between threads
   - Fine-grained locking to minimize contention

2. **Thread Specialization**:
   - Dedicated WebSocket Thread: Reading from the WebSocket happens in a dedicated thread
   - Processing Thread: Orderbook updates and model calculations run in a separate thread
   - UI Thread: The UI updates happen in a separate thread to avoid blocking the WebSocket processing
   - Each thread is optimized for its specific task with appropriate priorities

3. **Parallelism**:
   - Parallel processing of independent calculations
   - Work stealing thread pool for load balancing
   - SIMD instructions for vectorized numerical operations

### Network Optimization

1. **Connection Management**:
   - Heartbeat monitoring to detect connection issues early
   - Automatic reconnection with exponential backoff
   - Connection pooling for multiple endpoints

2. **Message Compression**:
   - Efficient binary protocols where supported
   - Compression for large payloads
   - Minimal handshaking overhead

### Algorithmic Optimizations

1. **Incremental Calculations**:
   - Models update incrementally when possible, rather than recalculating from scratch
   - Partial orderbook updates processed efficiently
   - Caching of intermediate results for repeated calculations

2. **Approximation Algorithms**:
   - Fast approximations used where exact calculations are too expensive
   - Lookup tables for common mathematical functions
   - Early stopping criteria for iterative algorithms when sufficient precision is reached

3. **Batch Processing**:
   - Multiple updates are batched when possible to amortize processing overhead
   - Vectorized operations for numerical calculations
   - Asynchronous processing of non-critical updates

### Latency Measurement and Monitoring

The system measures and reports:
1. **Processing Latency**: Time taken to process each orderbook update
2. **Model Calculation Latency**: Time spent in each model component
3. **End-to-End Latency**: Time from WebSocket message receipt to UI update
4. **99th Percentile Latency**: To identify worst-case performance issues

Latency statistics are used to:
- Identify performance bottlenecks
- Adjust processing parameters dynamically
- Trigger alerts for performance degradation
- Optimize component interactions

## Example Usage

### Initialization
```cpp
auto simulator = std::make_shared<TradeSimulator>();
simulator->initialize("wss://ws.gomarket-cpp.goquant.io/ws/l2-orderbook/okx/BTC-USDT-SWAP");
simulator->setParameters("OKX", "BTC-USDT-SWAP", "market", 100.0, 0.01, 1);
simulator->connect();
```

### Retrieving Estimates
```cpp
double slippage = simulator->getExpectedSlippage();
double fees = simulator->getExpectedFees();
double impact = simulator->getExpectedMarketImpact();
double netCost = simulator->getNetCost();
```

## WebSocket Data Format

The WebSocket endpoint provides L2 orderbook data in the following JSON format:
```json
{
  "timestamp": "2025-05-04T10:39:13Z",
  "exchange": "OKX",
  "symbol": "BTC-USDT-SWAP",
  "asks": [
    ["95445.5", "9.06"],
    ["95448", "2.05"],
    // ... more ask levels ...
  ],
  "bids": [
    ["95445.4", "1104.23"],
    ["95445.3", "0.02"],
    // ... more bid levels ...
  ]
}
```

## Future Enhancements

1. **Additional Exchanges**: Support for more exchanges beyond OKX
2. **Advanced Models**: Implementation of more sophisticated market impact models
3. **Machine Learning**: Integration of ML models trained on historical data
4. **Graphical UI**: Implementation of a proper graphical UI with orderbook visualization
5. **Backtesting**: Support for backtesting against historical data 