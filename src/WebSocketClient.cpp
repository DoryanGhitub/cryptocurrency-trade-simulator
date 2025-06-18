#include "../include/WebSocketClient.h"
#include <iostream>
#include <regex>
#include <boost/beast/core.hpp>
#include <boost/beast/ssl.hpp>
#include <boost/asio/strand.hpp>

namespace beast = boost::beast;
namespace websocket = beast::websocket;
namespace net = boost::asio;
namespace ssl = boost::asio::ssl;
using tcp = boost::asio::ip::tcp;

WebSocketClient::WebSocketClient(const std::string& url)
    : url_(url),
      connected_(false),
      should_run_(false)
{
    // std::cout << "WebSocketClient: Initializing with URL: " << url << std::endl;
    parseUrl(url);
}

WebSocketClient::~WebSocketClient() {
    disconnect();
}

void WebSocketClient::parseUrl(const std::string& url) {
    std::regex url_regex("(wss?)://([^:/]+)(:([0-9]+))?(/.*)?");
    std::smatch match;
    
    if (std::regex_match(url, match, url_regex)) {
        std::string protocol = match[1].str();
        use_ssl_ = (protocol == "wss");
        host_ = match[2].str();
        port_ = match[4].str();
        target_ = match[5].str();
        
        if (port_.empty()) {
            port_ = use_ssl_ ? "443" : "80";
        }
        
        if (target_.empty()) {
            target_ = "/";
        }
        
        // std::cout << "WebSocketClient: Parsed URL - Host: " << host_ << ", Port: " << port_ 
        //          << ", Target: " << target_ << ", SSL: " << (use_ssl_ ? "Yes" : "No") << std::endl;
    } else {
        std::cerr << "Invalid WebSocket URL: " << url << std::endl;
    }
}

bool WebSocketClient::connect() {
    if (connected_) {
        return true;
    }
    
    try {
        // std::cout << "WebSocketClient: Connecting to " << host_ << ":" << port_ << target_ << std::endl;
        
        // Set up the io_context and resolver
        boost::asio::ip::tcp::resolver resolver(io_context_);
        // std::cout << "WebSocketClient: Resolving hostname..." << std::endl;
        auto const results = resolver.resolve(host_, port_);
        // std::cout << "WebSocketClient: Hostname resolved successfully" << std::endl;
        
        if (use_ssl_) {
            // std::cout << "WebSocketClient: Setting up SSL connection..." << std::endl;
            
            // Set up SSL context
            ssl::context ctx(ssl::context::tlsv12_client);
            ctx.set_default_verify_paths();
            ctx.set_verify_mode(ssl::verify_peer);
            
            // Create SSL WebSocket stream
            auto ssl_stream = std::make_unique<boost::asio::ssl::stream<beast::tcp_stream>>(io_context_, ctx);
            
            // Set SNI Hostname (many hosts need this to handshake successfully)
            if(!SSL_set_tlsext_host_name(ssl_stream->native_handle(), host_.c_str())) {
                beast::error_code ec{static_cast<int>(::ERR_get_error()), net::error::get_ssl_category()};
                throw beast::system_error{ec};
            }
            
            // std::cout << "WebSocketClient: Connecting to the server..." << std::endl;
            // Connect to the server
            beast::get_lowest_layer(*ssl_stream).connect(results);
            
            // std::cout << "WebSocketClient: Performing SSL handshake..." << std::endl;
            // Perform SSL handshake
            ssl_stream->handshake(ssl::stream_base::client);
            
            // Create the WebSocket stream - use the same type as in the header
            wss_ = std::make_unique<websocket::stream<boost::asio::ssl::stream<beast::tcp_stream>>>(std::move(*ssl_stream));
            
            // Set the handshake timeout
            beast::get_lowest_layer(*wss_).expires_never();
            wss_->set_option(websocket::stream_base::timeout::suggested(beast::role_type::client));
            
            // Set a decorator to add headers to the handshake
            wss_->set_option(websocket::stream_base::decorator(
                [](websocket::request_type& req) {
                    req.set(beast::http::field::user_agent, "Boost.Beast WebSocket Client");
                }));
            
            // std::cout << "WebSocketClient: Performing WebSocket handshake..." << std::endl;
            // Perform the WebSocket handshake
            wss_->handshake(host_, target_);
            // std::cout << "WebSocketClient: WebSocket handshake successful" << std::endl;
            
            // For the gomarket-cpp.goquant.io endpoint, we don't need to send a subscription message
            // as it's pre-configured to stream the specific instrument
            if (host_.find("gomarket-cpp.goquant.io") == std::string::npos) {
                // Only send subscription for direct OKX connections
                std::string subscribe_msg = R"({
                    "op": "subscribe",
                    "args": [
                        {
                            "channel": "books",
                            "instId": "BTC-USDT"
                        }
                    ]
                })";
                // std::cout << "WebSocketClient: Sending subscription message..." << std::endl;
                wss_->write(net::buffer(subscribe_msg));
            } else {
                // std::cout << "WebSocketClient: Using pre-configured endpoint, no subscription needed" << std::endl;
            }
            
        } else {
            // std::cout << "WebSocketClient: Setting up non-SSL connection..." << std::endl;
            
            // Create regular WebSocket stream
            ws_ = std::make_unique<websocket::stream<beast::tcp_stream>>(io_context_);
            
            // Set the handshake timeout
            beast::get_lowest_layer(*ws_).expires_never();
            ws_->set_option(websocket::stream_base::timeout::suggested(beast::role_type::client));
            
            // std::cout << "WebSocketClient: Connecting to the server..." << std::endl;
            // Connect to the server
            beast::get_lowest_layer(*ws_).connect(results);
            
            // Set a decorator to add headers to the handshake
            ws_->set_option(websocket::stream_base::decorator(
                [](websocket::request_type& req) {
                    req.set(beast::http::field::user_agent, "Boost.Beast WebSocket Client");
                }));
            
            // std::cout << "WebSocketClient: Performing WebSocket handshake..." << std::endl;
            // Perform the WebSocket handshake
            ws_->handshake(host_, target_);
            // std::cout << "WebSocketClient: WebSocket handshake successful" << std::endl;
            
            // For the gomarket-cpp.goquant.io endpoint, we don't need to send a subscription message
            // as it's pre-configured to stream the specific instrument
            if (host_.find("gomarket-cpp.goquant.io") == std::string::npos) {
                // Only send subscription for direct OKX connections
                std::string subscribe_msg = R"({
                    "op": "subscribe",
                    "args": [
                        {
                            "channel": "books",
                            "instId": "BTC-USDT"
                        }
                    ]
                })";
                // std::cout << "WebSocketClient: Sending subscription message..." << std::endl;
                ws_->write(net::buffer(subscribe_msg));
            } else {
                // std::cout << "WebSocketClient: Using pre-configured endpoint, no subscription needed" << std::endl;
            }
        }
        
        // Set connection status and start reading messages
        connected_ = true;
        should_run_ = true;
        // std::cout << "WebSocketClient: Starting read thread..." << std::endl;
        read_thread_ = std::make_unique<std::thread>(&WebSocketClient::readMessages, this);
        
        return true;
    } catch (std::exception const& e) {
        std::cerr << "WebSocketClient: Connection error: " << e.what() << std::endl;
        connected_ = false;
        return false;
    }
}

void WebSocketClient::disconnect() {
    should_run_ = false;
    
    if (connected_) {
        try {
            // std::cout << "WebSocketClient: Disconnecting..." << std::endl;
            
            // Only send unsubscribe message if using direct OKX endpoint
            if (host_.find("gomarket-cpp.goquant.io") == std::string::npos) {
                // Send unsubscribe message before closing
                std::string unsubscribe_msg = R"({
                    "op": "unsubscribe",
                    "args": [
                        {
                            "channel": "books",
                            "instId": "BTC-USDT"
                        }
                    ]
                })";
                
                if (use_ssl_ && wss_) {
                    // std::cout << "WebSocketClient: Sending unsubscribe message..." << std::endl;
                    wss_->write(net::buffer(unsubscribe_msg));
                } else if (ws_) {
                    // std::cout << "WebSocketClient: Sending unsubscribe message..." << std::endl;
                    ws_->write(net::buffer(unsubscribe_msg));
                }
            }
            
            // Close the connection
            if (use_ssl_ && wss_) {
                // std::cout << "WebSocketClient: Closing WebSocket connection..." << std::endl;
                wss_->close(websocket::close_code::normal);
            } else if (ws_) {
                // std::cout << "WebSocketClient: Closing WebSocket connection..." << std::endl;
                ws_->close(websocket::close_code::normal);
            }
        } catch (std::exception const& e) {
            std::cerr << "WebSocketClient: Disconnect error: " << e.what() << std::endl;
        }
        
        connected_ = false;
    }
    
    if (read_thread_ && read_thread_->joinable()) {
        // std::cout << "WebSocketClient: Joining read thread..." << std::endl;
        read_thread_->join();
        // std::cout << "WebSocketClient: Read thread joined" << std::endl;
    }
}

bool WebSocketClient::isConnected() const {
    return connected_;
}

void WebSocketClient::setMessageCallback(std::function<void(const nlohmann::json&)> callback) {
    message_callback_ = callback;
}

bool WebSocketClient::sendMessage(const std::string& message) {
    if (!connected_) {
        return false;
    }
    
    try {
        // std::cout << "WebSocketClient: Sending message: " << message << std::endl;
        if (use_ssl_ && wss_) {
            wss_->write(net::buffer(message));
        } else if (ws_) {
            ws_->write(net::buffer(message));
        }
        return true;
    } catch (std::exception const& e) {
        std::cerr << "WebSocketClient: Write error: " << e.what() << std::endl;
        connected_ = false;
        return false;
    }
}

void WebSocketClient::readMessages() {
    beast::flat_buffer buffer;
    
    // std::cout << "WebSocketClient: Read thread started" << std::endl;
    
    while (should_run_ && connected_) {
        try {
            std::string data;
            
            if (use_ssl_ && wss_) {
                // std::cout << "WebSocketClient: Waiting for data..." << std::endl;
                wss_->read(buffer);
                data = beast::buffers_to_string(buffer.data());
                buffer.consume(buffer.size());
                // std::cout << "WebSocketClient: Received data (" << data.size() << " bytes): " << data << std::endl;
            } else if (ws_) {
                // std::cout << "WebSocketClient: Waiting for data..." << std::endl;
                ws_->read(buffer);
                data = beast::buffers_to_string(buffer.data());
                buffer.consume(buffer.size());
                // std::cout << "WebSocketClient: Received data (" << data.size() << " bytes): " << data << std::endl;
            }
            
            if (!data.empty() && message_callback_) {
                try {
                    // std::cout << "WebSocketClient: Parsing JSON: " << data << std::endl;
                    nlohmann::json json_data = nlohmann::json::parse(data);
                    // std::cout << "WebSocketClient: JSON parsed successfully" << std::endl;
                    nlohmann::json orderbook_data;
                    
                    // Check if this is direct OKX format or gomarket format
                    if (json_data.contains("data") && json_data["data"].is_array() && !json_data["data"].empty()) {
                        // std::cout << "WebSocketClient: Detected OKX native format" << std::endl;
                        // Process the OKX native response format
                        // Extract timestamp - OKX uses numerical timestamps
                        std::string timestamp;
                        if (json_data["data"][0].contains("ts")) {
                            // Convert numerical timestamp to string
                            timestamp = std::to_string(json_data["data"][0]["ts"].get<int64_t>());
                            // std::cout << "WebSocketClient: Extracted timestamp: " << timestamp << std::endl;
                        }
                        
                        // Create a properly formatted orderbook update
                        orderbook_data = {
                            {"timestamp", timestamp},
                            {"exchange", "OKX"},
                            {"symbol", json_data.contains("arg") && json_data["arg"].contains("instId") ? 
                                      json_data["arg"]["instId"].get<std::string>() : "BTC-USDT"},
                            {"asks", json_data["data"][0].contains("asks") ? json_data["data"][0]["asks"] : nlohmann::json::array()},
                            {"bids", json_data["data"][0].contains("bids") ? json_data["data"][0]["bids"] : nlohmann::json::array()}
                        };
                        
                        // Print the first few price levels to verify data quality
                        // if (orderbook_data["asks"].size() > 0 && orderbook_data["bids"].size() > 0) {
                        //     auto& asks = orderbook_data["asks"];
                        //     auto& bids = orderbook_data["bids"];
                        //     std::cout << "WebSocketClient: First ask: " << asks[0][0] << " x " << asks[0][1] << std::endl;
                        //     std::cout << "WebSocketClient: First bid: " << bids[0][0] << " x " << bids[0][1] << std::endl;
                        // }
                    } else if (json_data.contains("timestamp") && json_data.contains("asks") && json_data.contains("bids")) {
                        // std::cout << "WebSocketClient: Detected gomarket format" << std::endl;
                        // This appears to be already in gomarket format, use it directly
                        orderbook_data = json_data;
                        // std::cout << "WebSocketClient: Timestamp: " << json_data["timestamp"].get<std::string>() << std::endl;
                        // std::cout << "WebSocketClient: Ask levels: " << json_data["asks"].size() << ", Bid levels: " << json_data["bids"].size() << std::endl;
                    } else if (json_data.contains("event") && json_data["event"] == "subscribe") {
                        // Subscription confirmation, just log it
                        // std::cout << "WebSocketClient: Successfully subscribed to OKX orderbook channel" << std::endl;
                        return;
                    } else if (json_data.contains("event") && json_data["event"] == "error") {
                        // Error message
                        std::cerr << "WebSocketClient: OKX WebSocket error: " 
                                  << (json_data.contains("msg") ? json_data["msg"].get<std::string>() : "Unknown error")
                                  << " Code: " << (json_data.contains("code") ? json_data["code"].get<std::string>() : "Unknown code")
                                  << std::endl;
                        return;
                    } else {
                        // Unknown message
                        // std::cout << "WebSocketClient: Received unknown message format: " << data.substr(0, 500) << (data.size() > 500 ? "..." : "") << std::endl;
                        return;
                    }
                    
                    // std::cout << "WebSocketClient: Received orderbook data with "
                    //           << (orderbook_data["asks"].is_array() ? orderbook_data["asks"].size() : 0) << " asks and "
                    //           << (orderbook_data["bids"].is_array() ? orderbook_data["bids"].size() : 0) << " bids" << std::endl;
                    
                    if (message_callback_) {
                        message_callback_(orderbook_data);
                    }
                } catch (const nlohmann::json::parse_error& e) {
                    std::cerr << "WebSocketClient: JSON parse error: " << e.what() << std::endl;
                    std::cerr << "WebSocketClient: Raw data: " << data.substr(0, 200) << (data.size() > 200 ? "..." : "") << std::endl;
                }
            }
        } catch (beast::system_error const& se) {
            if (se.code() != websocket::error::closed) {
                std::cerr << "WebSocketClient: Read error: " << se.code().message() << std::endl;
            }
            connected_ = false;
            break;
        } catch (std::exception const& e) {
            std::cerr << "WebSocketClient: Read error: " << e.what() << std::endl;
            connected_ = false;
            break;
        }
    }
    
    // std::cout << "WebSocketClient: Read thread exiting" << std::endl;
} 