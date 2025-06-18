#!/bin/bash

# GPU Dependencies Setup Script for Advanced Trading Simulator
# This script installs PyTorch C++, ONNX Runtime, and CUDA dependencies

set -e

echo "🚀 Setting up GPU dependencies for Advanced Trading Simulator..."

# Check if CUDA is available
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep -oP "release \K[0-9.]+")
    echo "✅ CUDA $CUDA_VERSION detected"
else
    echo "⚠️  CUDA not found. Please install CUDA Toolkit first:"
    echo "   https://developer.nvidia.com/cuda-downloads"
    echo "   Continuing with CPU-only setup..."
fi

# Create dependencies directory
mkdir -p deps
cd deps

# Download and install PyTorch C++ (LibTorch)
echo "📦 Downloading PyTorch C++ (LibTorch)..."

# Check CUDA availability for appropriate download
if command -v nvcc &> /dev/null; then
    # CUDA version - download GPU-enabled LibTorch
    LIBTORCH_URL="https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.0.1%2Bcu118.zip"
    LIBTORCH_FILE="libtorch-gpu.zip"
    echo "📱 Downloading GPU-enabled LibTorch..."
else
    # CPU-only version
    LIBTORCH_URL="https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.0.1%2Bcpu.zip"
    LIBTORCH_FILE="libtorch-cpu.zip"
    echo "💻 Downloading CPU-only LibTorch..."
fi

if [ ! -f "$LIBTORCH_FILE" ]; then
    wget -O "$LIBTORCH_FILE" "$LIBTORCH_URL"
fi

if [ ! -d "libtorch" ]; then
    echo "📂 Extracting LibTorch..."
    unzip -q "$LIBTORCH_FILE"
fi

# Download and install ONNX Runtime
echo "📦 Downloading ONNX Runtime..."

if command -v nvcc &> /dev/null; then
    # GPU version
    ONNX_URL="https://github.com/microsoft/onnxruntime/releases/download/v1.15.1/onnxruntime-linux-x64-gpu-1.15.1.tgz"
    ONNX_FILE="onnxruntime-gpu.tgz"
    echo "📱 Downloading GPU-enabled ONNX Runtime..."
else
    # CPU version
    ONNX_URL="https://github.com/microsoft/onnxruntime/releases/download/v1.15.1/onnxruntime-linux-x64-1.15.1.tgz"
    ONNX_FILE="onnxruntime-cpu.tgz"
    echo "💻 Downloading CPU-only ONNX Runtime..."
fi

if [ ! -f "$ONNX_FILE" ]; then
    wget -O "$ONNX_FILE" "$ONNX_URL"
fi

if [ ! -d "onnxruntime" ]; then
    echo "📂 Extracting ONNX Runtime..."
    tar -xzf "$ONNX_FILE"
    mv onnxruntime-* onnxruntime
fi

# Set up environment variables
echo "🔧 Setting up environment variables..."

DEPS_DIR=$(pwd)
LIBTORCH_PATH="$DEPS_DIR/libtorch"
ONNX_PATH="$DEPS_DIR/onnxruntime"

# Create environment setup script
cat > ../setup_env.sh << EOF
#!/bin/bash
# Environment setup for GPU-accelerated trading simulator

export LIBTORCH_PATH="$LIBTORCH_PATH"
export ONNX_PATH="$ONNX_PATH"

# Add to library paths
export LD_LIBRARY_PATH="\$LIBTORCH_PATH/lib:\$ONNX_PATH/lib:\$LD_LIBRARY_PATH"

# Add to CMake paths
export CMAKE_PREFIX_PATH="\$LIBTORCH_PATH:\$ONNX_PATH:\$CMAKE_PREFIX_PATH"

# CUDA paths (if available)
if command -v nvcc &> /dev/null; then
    export CUDA_HOME=\$(dirname \$(dirname \$(which nvcc)))
    export PATH="\$CUDA_HOME/bin:\$PATH"
    export LD_LIBRARY_PATH="\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH"
fi

echo "✅ GPU environment configured"
echo "LibTorch: \$LIBTORCH_PATH"
echo "ONNX Runtime: \$ONNX_PATH"
if command -v nvcc &> /dev/null; then
    echo "CUDA: \$CUDA_HOME"
fi
EOF

chmod +x ../setup_env.sh

cd ..

echo ""
echo "🎉 GPU dependencies setup complete!"
echo ""
echo "Next steps:"
echo "1. Source the environment: source setup_env.sh"
echo "2. Build the project: ./build.sh"
echo ""
echo "Dependencies installed:"
echo "  - LibTorch: deps/libtorch/"
echo "  - ONNX Runtime: deps/onnxruntime/"
echo ""

if command -v nvcc &> /dev/null; then
    echo "🔥 GPU acceleration is available!"
    echo "   CUDA Version: $(nvcc --version | grep -oP "release \K[0-9.]+")"
    echo "   GPU Memory: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1) MB"
else
    echo "💡 For GPU acceleration, install CUDA Toolkit and re-run this script"
fi

echo ""
echo "For more information, see: https://pytorch.org/cppdocs/" 