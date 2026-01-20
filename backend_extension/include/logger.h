#ifndef LOGGER_H
#define LOGGER_H

#include <ctime>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

enum class LogLevel { DEBUG = 0, INFO = 1, WARN = 2, ERROR = 3 };

class Logger {
public:
  static void setLevel(LogLevel level) { get_level() = level; }

  static void log(LogLevel level, const std::string &component,
                  const std::string &message) {
    if (level < get_level())
      return;

    std::time_t now = std::time(nullptr);
    std::tm tm_struct;
#ifdef _WIN32
    localtime_s(&tm_struct, &now);
#else
    localtime_r(&now, &tm_struct);
#endif

    std::stringstream ss;
    ss << "[" << std::put_time(&tm_struct, "%Y-%m-%d %H:%M:%S") << "] ";

    switch (level) {
    case LogLevel::DEBUG:
      ss << "[DEBUG] ";
      break;
    case LogLevel::INFO:
      ss << "[INFO]  ";
      break;
    case LogLevel::WARN:
      ss << "[WARN]  ";
      break;
    case LogLevel::ERROR:
      ss << "[ERROR] ";
      break;
    }

    ss << "[" << component << "] " << message << std::endl;

    std::cout << ss.str();
  }

  static void debug(const std::string &component, const std::string &message) {
    log(LogLevel::DEBUG, component, message);
  }

  static void info(const std::string &component, const std::string &message) {
    log(LogLevel::INFO, component, message);
  }

  static void warn(const std::string &component, const std::string &message) {
    log(LogLevel::WARN, component, message);
  }

  static void error(const std::string &component, const std::string &message) {
    log(LogLevel::ERROR, component, message);
  }

  static void error(const std::string &component, const std::string &message,
                    int err) {
    std::stringstream ss;
    ss << message << " (errno: " << err << ")";
    log(LogLevel::ERROR, component, ss.str());
  }

private:
  static LogLevel &get_level() {
    static LogLevel level = LogLevel::INFO;
    return level;
  }
};

#define LOG_DEBUG(comp, msg) Logger::debug(comp, msg)
#define LOG_INFO(comp, msg) Logger::info(comp, msg)
#define LOG_WARN(comp, msg) Logger::warn(comp, msg)
#define LOG_ERROR(comp, msg) Logger::error(comp, msg)
#define LOG_ERROR_ERR(comp, msg, err) Logger::error(comp, msg, err)

#endif // LOGGER_H
