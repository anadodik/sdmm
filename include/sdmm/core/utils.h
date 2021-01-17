#pragma once

#include <filesystem>
#include <fstream>
#include <iomanip>

#include <fmt/ostream.h>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <enoki/array.h>
#include <enoki/dynamic.h>

#include "sdmm/linalg/vector.h"
#include "sdmm/linalg/matrix.h"

#define VECTORIZE_WRAP(FUNC_NAME) [](auto&&... params) { FUNC_NAME(params...); }
#define VECTORIZE_WRAP_FWD(FUNC_NAME) [](auto&&... params) { FUNC_NAME(std::forward<decltype(params)>(params)...); }
#define VECTORIZE_WRAP_OUTPUT(FUNC_NAME) [](auto&& output, auto&&... params) { output = FUNC_NAME(params...); }

#define VECTORIZE_WRAP_MEMBER(FUNC_NAME) \
    [](auto&& obj, [[maybe_unused]] auto&&... params) {\
        return std::forward<decltype(obj)>(obj).FUNC_NAME(params...); \
    }

namespace nlohmann {

    template <typename T>
    struct adl_serializer<std::unique_ptr<T>> {
        static void to_json(json& j, const std::unique_ptr<T>& opt) {
            if (opt.get()) {
                j = *opt;
            } else {
                j = nullptr;
            }
        }

        static std::unique_ptr<T> from_json(const json& j) {
            if (j.is_null()) {
                return nullptr;
            } else {
                return std::make_unique<T>(std::move(j.get<T>()));
            }
        }
    };

    template <typename T>
    struct adl_serializer<enoki::DynamicArray<T>> {
        static void to_json(json& j, const enoki::DynamicArray<T>& array) {
            j["size"] = array.size();
            for(size_t i = 0; i < array.size(); ++i) {
                j["data"].push_back(enoki::slice(array, i));
            }
        }

        static void from_json(const json& j, enoki::DynamicArray<T>& array) {
            if(j["size"] > 0) {
                enoki::set_slices(array, j["size"]);
                for(size_t i = 0; i < array.size(); ++i) {
                    enoki::slice(array, i) = j["data"][i];
                }
            }
        }
    };

    template<typename T, size_t Size>
    struct adl_serializer<enoki::Array<T, Size>> {
        static void to_json(json& j, const enoki::Array<T, Size>& array) {
            for(size_t i = 0; i < Size; ++i) {
                auto coeff = array.coeff(i);
                j["data"].push_back(coeff);
            }
        }

        static void from_json(const json& j, enoki::Array<T, Size>& array) {
            for(size_t i = 0; i < Size; ++i) {
                array.coeff(i) = j["data"][i];
            }
        }
    };

    template<typename T, size_t Size>
    struct adl_serializer<sdmm::linalg::Vector<T, Size>> {
        static void to_json(json& j, const sdmm::linalg::Vector<T, Size>& array) {
            for(size_t i = 0; i < Size; ++i) {
                auto coeff = array.coeff(i);
                j["data"].push_back(coeff);
            }
        }

        static void from_json(const json& j, sdmm::linalg::Vector<T, Size>& array) {
            for(size_t i = 0; i < Size; ++i) {
                array.coeff(i) = j["data"][i].get<T>();
            }
        }
    };

    template<typename T, size_t Size>
    struct adl_serializer<sdmm::linalg::Matrix<T, Size>> {
        static void to_json(json& j, const sdmm::linalg::Matrix<T, Size>& array) {
            for(size_t c = 0; c < Size; ++c) {
                for(size_t r = 0; r < Size; ++r) {
                    auto coeff = array(c, r);
                    j["data"].push_back(coeff);
                }
            }
        }

        static void from_json(const json& j, sdmm::linalg::Matrix<T, Size>& array) {
            size_t i = 0;
            for(size_t c = 0; c < Size; ++c) {
                for(size_t r = 0; r < Size; ++r) {
                    array(c, r) = j["data"][i];
                    ++i;
                }
            }
        }
    };
}

namespace sdmm {

using namespace nlohmann;
namespace fs = std::filesystem;

template<typename...> struct Debug;

template<typename T>
auto save_json(const T& t, const fs::path& path) {
    json j;
    j = t;
    std::ofstream file(path);
    file << std::setw(4) << j << std::endl;
    // std::cerr << std::setw(4) << j << std::endl;
}

template<typename T>
auto load_json(T& t, const fs::path& path) {
    std::ifstream file(path);
    json j;
    file >> j;
    t = j.get<T>();
    // std::cerr << std::setw(4) << j << std::endl;
}

template<typename Func, typename... Args>
auto vectorize(const Func& func, Args&&... args) {
    std::tuple packet_sizes{enoki::packets(args)...};
    std::tuple slice_sizes{enoki::slices(args)...};
    spdlog::info("{}", packet_sizes);
}

template<typename Value, typename Enable=void>
struct nested_packet_size;

template<typename Value>
struct nested_packet_size<Value, std::enable_if_t<enoki::is_dynamic_v<Value>>> {
    static constexpr size_t value = Value::PacketSize;
};

template<typename Value>
struct nested_packet_size<Value, std::enable_if_t<!enoki::is_dynamic_v<Value>>> {
    static constexpr size_t value = Value::Size;
};

template<typename Value>
static constexpr size_t nested_packet_size_v = nested_packet_size<Value>::value;

template<typename Value, typename Enable=void>
struct nested_packet;

template<typename Value>
struct nested_packet<Value, std::enable_if_t<enoki::is_dynamic_v<Value>>> {
    using type = typename Value::Packet;
};

template<typename Value>
struct nested_packet<Value, std::enable_if_t<!enoki::is_dynamic_v<Value>>> {
    using type = Value;
};

template<typename Value>
using nested_packet_t = typename nested_packet<Value>::type;

template<typename Value, typename New, typename Enable=void>
struct outer_type {
    using type = New;
};

template<typename Value, typename New>
struct outer_type<
    Value,
    New,
    std::enable_if_t<
        enoki::is_array_v<Value> && (enoki::array_depth_v<Value> > 1)
    >
> {
    using type = std::remove_reference_t<typename Value::template ReplaceValue<New>>;
};

template<typename Value, typename New>
struct outer_type<
    Value,
    New,
    std::enable_if_t<
        !enoki::is_array_v<Value> ||
        (enoki::is_array_v<Value> && enoki::array_depth_v<Value> <= 1)
    >
> {
    using type = New;
};

template<typename Value, typename New>
using outer_type_t = typename outer_type<std::remove_reference_t<Value>, New>::type;

template<typename Value, std::enable_if_t<!enoki::is_array_v<Value>, int> = 0>
auto& coeff_safe(Value& value, [[maybe_unused]] size_t i) {
    assert(i == 0);
    return value;
};

template<typename Value, std::enable_if_t<enoki::is_array_v<Value>, int> = 0>
ENOKI_INLINE auto& coeff_safe(Value& value, size_t i) {
    return value.coeff(i);
};

template<typename Value, std::enable_if_t<enoki::is_array_v<Value>, int> = 0>
ENOKI_INLINE const auto& coeff_safe(const Value& value, size_t i) {
    return value.coeff(i);
};

template <typename Value_, size_t Size_>
using Vector = sdmm::linalg::Vector<Value_, Size_>;

template<typename Value, size_t Rows, size_t Cols=Rows>
using Matrix = sdmm::linalg::Matrix<Value, Rows, Cols>;

template<typename Type, typename Value, size_t Size, typename... Args>
auto full_inner(Args... args) {
    return Type(enoki::full<Value>(args, Size)...);
}

// https://stackoverflow.com/questions/81870/is-it-possible-to-print-a-variables-type-in-standard-c
template <class T>
constexpr std::string_view type_name() {
    using namespace std;
#ifdef __clang__
    string_view p = __PRETTY_FUNCTION__;
    return string_view(p.data() + 34, p.size() - 34 - 1);
#elif defined(__GNUC__)
    string_view p = __PRETTY_FUNCTION__;
#  if __cplusplus < 201402
    return string_view(p.data() + 36, p.size() - 36 - 1);
#  else
    return string_view(p.data() + 49, p.find(';', 49) - 49);
#  endif
#elif defined(_MSC_VER)
    string_view p = __FUNCSIG__;
    return string_view(p.data() + 84, p.size() - 84 - 7);
#endif
}

}

struct MutexWrapper {
    MutexWrapper() = default;
    ~MutexWrapper() = default;
    MutexWrapper([[maybe_unused]] const MutexWrapper& mutex_wrapper) { };
    MutexWrapper([[maybe_unused]] MutexWrapper&& mutex_wrapper) { };
    MutexWrapper& operator=([[maybe_unused]] const MutexWrapper& mutex_wrapper) { return *this; };
    MutexWrapper& operator=([[maybe_unused]] MutexWrapper&& mutex_wrapper) { return *this; };

    std::mutex mutex;
};
