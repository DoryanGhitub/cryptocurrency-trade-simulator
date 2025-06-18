#pragma once

#include <string>
#include <functional>
#include <memory>
#include <thread>
#include <atomic>
#include <boost/beast/core.hpp>
#include <boost/beast/websocket.hpp>
#include <boost/beast/ssl.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/asio/ssl.hpp>
#include <nlohmann/json.hpp>

/**
 * @brief WebSocket client for connecting to the L2 orderbook feed
 */
class WebSocketClient {
public:
    /**
     * @brief Constructor
     * @param url The WebSocket URL to connect to
     */
    WebSocketClient(const std::string& url);
    
    /**
     * @brief Destructor
     */
    ~WebSocketClient();

    /**
     * @brief Connect to the WebSocket server
     * @return true if connection succeeded, false otherwise
     */
    bool connect();

    /**
     * @brief Disconnect from the WebSocket server
     */
    void disconnect();

    /**
     * @brief Check if connected to the WebSocket server
     * @return true if connected, false otherwise
     */
    bool isConnected() const;

    /**
     * @brief Set callback for received messages
     * @param callback Function to be called when a message is received
     */
    void setMessageCallback(std::function<void(const nlohmann::json&)> callback);
    
    /**
     * @brief Send a message to the WebSocket server
     * @param message The message to send
     * @return true if send succeeded, false otherwise
     */
    bool sendMessage(const std::string& message);

private:
    /**
     * @brief Parse the URL into host, port, and target
     * @param url The WebSocket URL
     */
    void parseUrl(const std::string& url);
    
    /**
     * @brief Read messages from the WebSocket
     */
    void readMessages();

    std::string url_;
    std::string host_;
    std::string port_;
    std::string target_;
    bool use_ssl_;
    
    std::atomic<bool> connected_;
    std::atomic<bool> should_run_;
    
    std::function<void(const nlohmann::json&)> message_callback_;
    
    std::unique_ptr<std::thread> read_thread_;
    
    boost::asio::io_context io_context_;
    std::unique_ptr<boost::beast::websocket::stream<boost::beast::tcp_stream>> ws_;
    std::unique_ptr<boost::beast::websocket::stream<boost::asio::ssl::stream<boost::beast::tcp_stream>>> wss_;
}; 