#!/bin/bash

# Exit on error
set -e

echo "Installing dependencies..."
apt-get update -y
apt-get install -y libeigen3-dev

# Create build directory if it doesn't exist
mkdir -p build
cd build

# Configure with CMake
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-O3 -march=native"

# Build with multiple cores
cmake --build . --config Release -- -j$(nproc)

echo "Build successful!"
echo "--------------------------------------------------------------"
echo "To run the trade simulator with Transformer-based maker/taker model:"
echo "./TradeSimulator --exchange=OKX --symbol=BTC-USDT-SWAP --order-type=market --quantity=1.0 --volatility=0.02 --tier=1"
echo ""
echo "For limit orders:"
echo "./TradeSimulator --exchange=OKX --symbol=BTC-USDT-SWAP --order-type=limit --quantity=1.0 --volatility=0.02 --tier=1"
echo "--------------------------------------------------------------"
echo "Add --debug flag to see detailed model outputs and intermediate calculations" 