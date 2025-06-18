cmake_minimum_required(VERSION 3.16)
project(TradeSimulator VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(Threads REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(Boost REQUIRED COMPONENTS system regex)
find_package(Eigen3 REQUIRED)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/models
    ${CMAKE_CURRENT_SOURCE_DIR}/ui
    ${EIGEN3_INCLUDE_DIR}
)

# Add source files
file(GLOB_RECURSE SOURCES 
    "src/*.cpp"
    "models/*.cpp"
    "ui/*.cpp"
)

# Remove main.cpp from sources for non-main executables
list(FILTER SOURCES EXCLUDE REGEX ".*src/main\\.cpp$")

# Add main executable
add_executable(trade_simulator ${SOURCES} "src/main.cpp")
set_target_properties(trade_simulator PROPERTIES OUTPUT_NAME "trade_simulator")

# Add volatility test executable
add_executable(volatility_test ${SOURCES} "test_volatility.cpp")
set_target_properties(volatility_test PROPERTIES OUTPUT_NAME "volatility_test")

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
)

# Install target
install(TARGETS trade_simulator volatility_test DESTINATION bin) 