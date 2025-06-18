# Trade Simulator

A high-performance trade simulator for estimating transaction costs and market impact, leveraging real-time market data from cryptocurrency exchanges.

## Overview

This system connects to WebSocket endpoints to stream full L2 orderbook data and provides real-time estimations of:

- Expected Slippage (using linear regression modeling)
- Expected Fees (rule-based fee model)
- Expected Market Impact (Almgren-Chriss model)
- Net Cost (Slippage + Fees + Market Impact)
- Maker/Taker proportion (Transformer-based deep learning model)
- Internal Latency (measured as processing time per tick)

## Requirements

- C++17 compatible compiler
- CMake 3.16 or higher
- Boost 1.66 or higher
- OpenSSL
- nlohmann_json
- Eigen3 (required for transformer model implementation)

## Building

```bash
# Create a build directory
mkdir build
cd build

# Configure with CMake
cmake ..

# Build
make
```

## Running

```bash
# Run the simulator with default settings
cd build
./trade_simulator

# Run with a custom WebSocket URL
./trade_simulator --url wss://custom-websocket-url.com/path

# Additional command line options
./trade_simulator --help
```

## Features

### 1. WebSocket Implementation

The simulator connects to the WebSocket endpoint: `wss://ws.gomarket-cpp.goquant.io/ws/l2-orderbook/okx/BTC-USDT-SWAP` to receive real-time L2 orderbook data.

### 2. Console-based UI

The application features a clean console-based UI that displays:
- Real-time orderbook visualization with bid/ask prices and quantities
- Current parameter settings
- Calculated output values with proper precision formatting

### 3. Input Parameters

- Exchange (OKX)
- Symbol (BTC-USDT-SWAP by default)
- Order Type (market)
- Quantity (100 by default)
- Volatility (market parameter, with 6-digit precision display)
- Fee Tier (based on exchange documentation)

### 4. Output Parameters

- Expected Slippage (with 6-digit precision)
- Expected Fees (with 6-digit precision)
- Expected Market Impact (with 6-digit precision)
- Net Cost (with 6-digit precision)
- Maker/Taker proportion (with 2-digit precision)
- Internal Latency (measured as processing time per tick)

## Implementation Details

### Almgren-Chriss Model

The Almgren-Chriss model provides a mathematical approach to executing large trades optimally by balancing the trade-off between market impact and execution risk.

The model divides market impact into two components:
- Temporary Impact: The immediate price change caused by executing part of the order
- Permanent Impact: The lasting change in the asset's price due to the execution

### Regression Models

- Linear Regression: Used for slippage prediction
- Quantile Regression: Alternative approach for slippage estimation

### Transformer-based Maker/Taker Model

The system now uses a state-of-the-art Temporal Fusion Transformer (TFT) model for maker/taker prediction:

- Multi-head self-attention mechanism to capture complex market dependencies
- Specialized for cryptocurrency trading patterns, particularly Bitcoin
- Takes into account order book depth, imbalance, and spread dynamics
- Adaptive to changing market conditions with contextual feature extraction
- More accurate than previous logistic regression approach, especially during high volatility periods

### Fee Model

Implements a rule-based fee model based on exchange documentation, considering:
- Order type (market or limit)
- Fee tier
- Maker/taker proportion predicted by the transformer model

## Recent Improvements

### Transformer Model Implementation
- Replaced logistic regression with Temporal Fusion Transformer for maker/taker prediction
- Added order book pressure analysis for improved market imbalance detection
- Bitcoin-specific market impact calibration for more accurate cost estimates
- Enhanced feature extraction for better pattern recognition

### Parameter Display Enhancement
- Improved display formatting for all parameters
- Added consistent precision settings (6 digits for volatility, slippage, fees, and market impact)
- Fixed issue with parameters showing as zeros despite receiving real data

### Code Optimization
- Removed excessive debug logging for cleaner console output
- Optimized WebSocket data handling for better performance
- Standardized symbol representation throughout the codebase

### Configuration
- Easily configurable parameters through command line arguments or config file
- Consistent use of "BTC-USDT-SWAP" format for crypto symbol

## Performance Optimization

The implementation includes various performance optimizations:
- Efficient data structures for orderbook processing
- Thread management for UI and WebSocket communications
- Memory management optimizations
- Latency benchmarking

## Docker Support

This project includes Docker configuration for both production and development environments.

### Prerequisites

- Docker
- Docker Compose

### Production Container

Build and run the complete application:

```bash
# Build and run using docker-compose
docker-compose up --build trade-simulator

# Or using plain Docker
docker build -t trade-simulator .
docker run -it trade-simulator
```

### Development Container

Use the development container for a pre-configured environment:

```bash
# Start the development container
docker-compose up -d dev

# Access the container shell
docker exec -it trade-simulator-dev bash
```

For more details on Docker usage, see [docker_readme.md](docker_readme.md).

## License

MIT License 