cmake_minimum_required(VERSION 3.16)
project(TradeSimulator VERSION 1.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable CUDA support
enable_language(CUDA)
find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Find PyTorch C++
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Find ONNX Runtime
find_package(PkgConfig REQUIRED)
pkg_check_modules(ONNXRUNTIME REQUIRED libonnxruntime)

# Find required packages
find_package(Threads REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(Boost REQUIRED COMPONENTS system regex)
find_package(Eigen3 REQUIRED)

# CUDA compute capability
set(CMAKE_CUDA_ARCHITECTURES 60 61 70 75 80 86)
set(CUDA_SEPARABLE_COMPILATION ON)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/models
    ${CMAKE_CURRENT_SOURCE_DIR}/ui
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu
    ${EIGEN3_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIRS}
    ${ONNXRUNTIME_INCLUDE_DIRS}
)

# Add source files
file(GLOB_RECURSE SOURCES 
    "src/*.cpp"
    "models/*.cpp"
    "ui/*.cpp"
    "gpu/*.cu"
    "gpu/*.cpp"
)

# Remove main.cpp from sources for non-main executables
list(FILTER SOURCES EXCLUDE REGEX ".*src/main\\.cpp$")

# Add main executable
add_executable(trade_simulator ${SOURCES} "src/main.cpp")
set_target_properties(trade_simulator PROPERTIES OUTPUT_NAME "trade_simulator")

# Add volatility test executable
add_executable(volatility_test ${SOURCES} "test_volatility.cpp")
set_target_properties(volatility_test PROPERTIES OUTPUT_NAME "volatility_test")

# Set CUDA properties
set_property(TARGET trade_simulator PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET volatility_test PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# Link libraries
target_link_libraries(trade_simulator
    PRIVATE
    Threads::Threads
    nlohmann_json::nlohmann_json
    OpenSSL::SSL
    OpenSSL::Crypto
    Boost::system
    Boost::regex
    Eigen3::Eigen
    ${TORCH_LIBRARIES}
    ${ONNXRUNTIME_LIBRARIES}
    CUDA::cudart
    CUDA::cublas
    CUDA::curand
    CUDA::cufft
)

target_link_libraries(volatility_test
    PRIVATE
    Threads::Threads
    nlohmann_json::nlohmann_json
    OpenSSL::SSL
    OpenSSL::Crypto
    Boost::system
    Boost::regex
    Eigen3::Eigen
    ${TORCH_LIBRARIES}
    ${ONNXRUNTIME_LIBRARIES}
    CUDA::cudart
    CUDA::cublas
    CUDA::curand
    CUDA::cufft
)

# Install target
install(TARGETS trade_simulator volatility_test DESTINATION bin) 