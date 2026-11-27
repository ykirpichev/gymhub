// gymkit C++ harness support: JSON in, typed arguments, timed call, JSON out.
//
// A harness (problems/<slug>/cpp/harness.cpp) includes this header and the user's solution.cpp,
// then calls gym::serve with a function that converts one case's arguments, times the user's
// call with gym::timed, and returns the result. Only the user's call is timed.
//
// Protocol: stdin holds {"cases": [[arg0, arg1, ...], ...]}. The original stdout gets one JSON line
// per case: {"ms": 1.23, "result": ..., "stdout": "..."} or {"error": "...", "stdout": "..."}.
// While a case runs, file descriptor 1 points at a temporary file, so whatever the solution prints
// (cout or printf) is captured per case and cannot corrupt the protocol.
#pragma once

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unistd.h>
#include <utility>
#include <vector>

namespace gym {

struct Json {
    enum class Type { Null, Bool, Number, String, Array, Object };
    Type type = Type::Null;
    bool boolean = false;
    double number = 0;
    bool integral = false;
    long long integer = 0;
    std::string text;
    std::vector<Json> items;
    std::vector<std::pair<std::string, Json>> fields;

    const Json& operator[](size_t index) const {
        if (type != Type::Array) throw std::runtime_error("expected a JSON array");
        if (index >= items.size()) throw std::runtime_error("argument index out of range");
        return items[index];
    }
    size_t size() const { return items.size(); }
    bool is_string() const { return type == Type::String; }
};

class Parser {
  public:
    explicit Parser(const std::string& source) : s(source) {}

    Json parse() {
        Json value = parse_value();
        skip();
        if (pos != s.size()) fail("trailing characters");
        return value;
    }

  private:
    const std::string& s;
    size_t pos = 0;

    [[noreturn]] void fail(const char* what) {
        throw std::runtime_error(std::string("bad input JSON: ") + what + " at " + std::to_string(pos));
    }
    void skip() {
        while (pos < s.size() && (s[pos] == ' ' || s[pos] == '\n' || s[pos] == '\r' || s[pos] == '\t')) pos++;
    }
    bool consume(const char* word) {
        size_t n = std::char_traits<char>::length(word);
        if (s.compare(pos, n, word) == 0) {
            pos += n;
            return true;
        }
        return false;
    }
    Json parse_value() {
        skip();
        if (pos >= s.size()) fail("unexpected end");
        Json value;
        char c = s[pos];
        if (c == '[') {
            value.type = Json::Type::Array;
            pos++;
            skip();
            if (pos < s.size() && s[pos] == ']') {
                pos++;
                return value;
            }
            while (true) {
                value.items.push_back(parse_value());
                skip();
                if (pos < s.size() && s[pos] == ',') {
                    pos++;
                    continue;
                }
                if (pos < s.size() && s[pos] == ']') {
                    pos++;
                    break;
                }
                fail("expected , or ]");
            }
        } else if (c == '{') {
            value.type = Json::Type::Object;
            pos++;
            skip();
            if (pos < s.size() && s[pos] == '}') {
                pos++;
                return value;
            }
            while (true) {
                skip();
                Json key = parse_value();
                if (key.type != Json::Type::String) fail("object key must be a string");
                skip();
                if (pos >= s.size() || s[pos] != ':') fail("expected :");
                pos++;
                value.fields.emplace_back(key.text, parse_value());
                skip();
                if (pos < s.size() && s[pos] == ',') {
                    pos++;
                    continue;
                }
                if (pos < s.size() && s[pos] == '}') {
                    pos++;
                    break;
                }
                fail("expected , or }");
            }
        } else if (c == '"') {
            value.type = Json::Type::String;
            pos++;
            while (pos < s.size() && s[pos] != '"') {
                char ch = s[pos++];
                if (ch != '\\') {
                    value.text += ch;
                    continue;
                }
                if (pos >= s.size()) fail("bad escape");
                char e = s[pos++];
                switch (e) {
                    case 'n': value.text += '\n'; break;
                    case 't': value.text += '\t'; break;
                    case 'r': value.text += '\r'; break;
                    case 'b': value.text += '\b'; break;
                    case 'f': value.text += '\f'; break;
                    case 'u': {
                        unsigned code = std::stoul(s.substr(pos, 4), nullptr, 16);
                        pos += 4;
                        if (code < 0x80) {
                            value.text += static_cast<char>(code);
                        } else if (code < 0x800) {
                            value.text += static_cast<char>(0xC0 | (code >> 6));
                            value.text += static_cast<char>(0x80 | (code & 0x3F));
                        } else {
                            value.text += static_cast<char>(0xE0 | (code >> 12));
                            value.text += static_cast<char>(0x80 | ((code >> 6) & 0x3F));
                            value.text += static_cast<char>(0x80 | (code & 0x3F));
                        }
                        break;
                    }
                    default: value.text += e;
                }
            }
            if (pos >= s.size()) fail("unterminated string");
            pos++;
        } else if (consume("true")) {
            value.type = Json::Type::Bool;
            value.boolean = true;
        } else if (consume("false")) {
            value.type = Json::Type::Bool;
        } else if (consume("null")) {
            value.type = Json::Type::Null;
        } else {
            size_t start = pos;
            if (s[pos] == '-') pos++;
            while (pos < s.size() && (isdigit(static_cast<unsigned char>(s[pos])) || s[pos] == '.' || s[pos] == 'e' ||
                                      s[pos] == 'E' || s[pos] == '+' || s[pos] == '-'))
                pos++;
            std::string token = s.substr(start, pos - start);
            if (token.empty()) fail("unexpected character");
            value.type = Json::Type::Number;
            value.number = std::stod(token);
            value.integral = token.find_first_of(".eE") == std::string::npos;
            if (value.integral) value.integer = std::stoll(token);
        }
        return value;
    }
};

// ---- JSON -> C++ ----------------------------------------------------------------------------

template <typename T>
struct From;

template <typename T>
T as(const Json& json) {
    return From<T>::get(json);
}

template <typename T>
struct From {
    static T get(const Json& json) {
        if constexpr (std::is_same_v<T, bool>) {
            if (json.type != Json::Type::Bool) throw std::runtime_error("expected a bool");
            return json.boolean;
        } else if constexpr (std::is_integral_v<T>) {
            if (json.type != Json::Type::Number) throw std::runtime_error("expected an integer");
            return static_cast<T>(json.integral ? json.integer : static_cast<long long>(json.number));
        } else if constexpr (std::is_floating_point_v<T>) {
            if (json.type != Json::Type::Number) throw std::runtime_error("expected a number");
            return static_cast<T>(json.integral ? static_cast<double>(json.integer) : json.number);
        } else if constexpr (std::is_same_v<T, std::string>) {
            if (json.type != Json::Type::String) throw std::runtime_error("expected a string");
            return json.text;
        } else if constexpr (std::is_same_v<T, Json>) {
            return json;
        } else {
            static_assert(sizeof(T) == 0, "gym::as: unsupported type");
        }
    }
};

template <typename T>
struct From<std::vector<T>> {
    static std::vector<T> get(const Json& json) {
        if (json.type != Json::Type::Array) throw std::runtime_error("expected a list");
        std::vector<T> out;
        out.reserve(json.items.size());
        for (const Json& item : json.items) out.push_back(as<T>(item));
        return out;
    }
};

template <typename A, typename B>
struct From<std::pair<A, B>> {
    static std::pair<A, B> get(const Json& json) {
        if (json.type != Json::Type::Array || json.items.size() != 2) throw std::runtime_error("expected a pair");
        return {as<A>(json.items[0]), as<B>(json.items[1])};
    }
};

// ---- C++ -> JSON text ------------------------------------------------------------------------

inline void write_string(std::string& out, const std::string& text) {
    out += '"';
    for (unsigned char c : text) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\t': out += "\\t"; break;
            case '\r': out += "\\r"; break;
            default:
                if (c < 0x20) {
                    char buffer[8];
                    std::snprintf(buffer, sizeof buffer, "\\u%04x", c);
                    out += buffer;
                } else {
                    out += static_cast<char>(c);
                }
        }
    }
    out += '"';
}

template <typename T>
void write(std::string& out, const T& value);

template <typename T>
struct To {
    static void put(std::string& out, const T& value) {
        if constexpr (std::is_same_v<T, bool>) {
            out += value ? "true" : "false";
        } else if constexpr (std::is_integral_v<T>) {
            out += std::to_string(value);
        } else if constexpr (std::is_floating_point_v<T>) {
            if (!std::isfinite(value)) {
                out += "null";
                return;
            }
            char buffer[40];
            std::snprintf(buffer, sizeof buffer, "%.17g", static_cast<double>(value));
            out += buffer;
        } else if constexpr (std::is_same_v<T, std::string>) {
            write_string(out, value);
        } else if constexpr (std::is_same_v<T, const char*>) {
            write_string(out, value);
        } else {
            static_assert(sizeof(T) == 0, "gym::write: unsupported type");
        }
    }
};

template <typename T>
struct To<std::vector<T>> {
    static void put(std::string& out, const std::vector<T>& value) {
        out += '[';
        for (size_t i = 0; i < value.size(); ++i) {
            if (i) out += ',';
            if constexpr (std::is_same_v<T, bool>) {
                out += value[i] ? "true" : "false";
            } else {
                write(out, static_cast<const T&>(value[i]));
            }
        }
        out += ']';
    }
};

template <typename A, typename B>
struct To<std::pair<A, B>> {
    static void put(std::string& out, const std::pair<A, B>& value) {
        out += '[';
        write(out, value.first);
        out += ',';
        write(out, value.second);
        out += ']';
    }
};

template <typename T>
void write(std::string& out, const T& value) {
    To<std::decay_t<T>>::put(out, value);
}

// ---- running cases ---------------------------------------------------------------------------

struct Result {
    std::string json;  // already-serialized result
    double ms = 0;
};

// Time `f()` and serialize what it returns. Serialization is not timed.
template <typename F>
Result timed(F&& f) {
    auto start = std::chrono::steady_clock::now();
    auto value = f();
    auto stop = std::chrono::steady_clock::now();
    Result result;
    result.ms = std::chrono::duration<double, std::milli>(stop - start).count();
    write(result.json, value);
    return result;
}

inline std::string read_capture(FILE* capture, size_t limit = 16000) {
    std::fflush(stdout);
    std::cout.flush();
    std::string text;
    std::rewind(capture);
    char buffer[4096];
    size_t n;
    while ((n = std::fread(buffer, 1, sizeof buffer, capture)) > 0) {
        text.append(buffer, n);
        if (text.size() > limit) {
            text.resize(limit);
            text += "\n... (output truncated)";
            break;
        }
    }
    return text;
}

template <typename Handler>
int serve(Handler&& handler) {
    std::ios::sync_with_stdio(true);
    std::string input((std::istreambuf_iterator<char>(std::cin)), std::istreambuf_iterator<char>());
    Json root = Parser(input).parse();
    const Json* cases = nullptr;
    for (const auto& [key, value] : root.fields)
        if (key == "cases") cases = &value;
    if (!cases || cases->type != Json::Type::Array) throw std::runtime_error("input has no cases");

    std::fflush(stdout);
    FILE* protocol = fdopen(dup(1), "w");
    if (!protocol) throw std::runtime_error("cannot open the result channel");
    // With GYMKIT_CAPTURE_DIR set, case i prints into <dir>/case-<i>.txt, which survives a crash.
    const char* capture_dir = std::getenv("GYMKIT_CAPTURE_DIR");
    size_t index = 0;
    for (const Json& args : cases->items) {
        FILE* capture = nullptr;
        if (capture_dir) {
            std::string name = std::string(capture_dir) + "/case-" + std::to_string(index) + ".txt";
            capture = std::fopen(name.c_str(), "w+");
        }
        if (!capture) capture = std::tmpfile();
        index++;
        if (capture) dup2(fileno(capture), 1);
        std::string line;
        try {
            Result result = handler(args);
            char ms[40];
            std::snprintf(ms, sizeof ms, "%.6f", result.ms);
            line = std::string("{\"ms\":") + ms + ",\"result\":" + result.json;
        } catch (const std::exception& error) {
            line = "{\"error\":";
            write_string(line, error.what());
        } catch (...) {
            line = "{\"error\":\"unknown exception\"";
        }
        if (capture) {
            line += ",\"stdout\":";
            write_string(line, read_capture(capture));
            std::fclose(capture);
        }
        line += "}\n";
        std::fputs(line.c_str(), protocol);
        std::fflush(protocol);
    }
    return 0;
}

}  // namespace gym
