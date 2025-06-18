# Using Docker with Trade Simulator

This project includes Docker configuration for both production and development environments.

## Prerequisites

- Docker
- Docker Compose

## Production Container

The production container builds and runs the trade simulator application.

### Building and Running

```bash
# Build and run using docker-compose
docker-compose up --build trade-simulator

# Or using plain Docker
docker build -t trade-simulator .
docker run -it trade-simulator
```

### Configuration

You can modify the runtime parameters in several ways:

1. **Through environment variables in docker-compose.yml**:
   ```yaml
   environment:
     - EXCHANGE=OKX
     - SYMBOL=BTC-USDT-SWAP
     - ORDER_TYPE=market
     - QUANTITY=1.0
     - VOLATILITY=0.02
     - TIER=1
   ```

2. **By passing command-line arguments**:
   ```bash
   docker run -it trade-simulator --exchange=OKX --symbol=BTC-USDT-SWAP --order-type=limit --quantity=2.0
   ```

## Development Container

The development container provides an environment with all dependencies installed but doesn't build the project. This is useful for development and debugging.

### Starting Development Environment

```bash
# Start the development container
docker-compose up -d dev

# Access the container shell
docker exec -it trade-simulator-dev bash
```

### Building Inside Development Container

Once inside the container:

```bash
# Create build directory
mkdir -p build && cd build

# Configure with CMake (Debug mode for development)
cmake .. -DCMAKE_BUILD_TYPE=Debug

# Build
cmake --build . -- -j$(nproc)

# Run
./trade_simulator
```

## Data Persistence

The containers mount a `./data` directory to `/app/data` inside the container. Use this for any data that needs to persist between container runs.

## Advanced Usage

### Running with Different Parameters

```bash
# Or manually override the command
docker-compose run trade-simulator --exchange=Binance --symbol=ETH-USDT-SWAP
``` 