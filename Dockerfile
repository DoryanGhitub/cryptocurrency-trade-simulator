FROM ubuntu:22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update -y && \
    apt-get install -y \
    build-essential \
    cmake \
    libssl-dev \
    libboost-all-dev \
    libeigen3-dev \
    nlohmann-json3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy source files
COPY . .

# Make sure Eigen headers are findable
RUN ln -sf /usr/include/eigen3/Eigen /usr/include/Eigen

# Build the project with debug symbols
RUN mkdir -p build && \
    cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_CXX_FLAGS="-O2 -g" && \
    cmake --build . --config RelWithDebInfo -- -j$(nproc)

# Set the entrypoint
ENTRYPOINT ["/app/build/trade_simulator"]

# Default command
CMD ["--exchange=OKX", "--symbol=BTC-USDT-SWAP", "--order-type=market", "--quantity=1.0", "--volatility=0.02", "--tier=1"] 