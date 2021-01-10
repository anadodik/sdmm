#pragma once

#include <mutex>

#include <sdmm/core/utils.h>
#include <sdmm/opt/em.h>

struct MutexWrapper {
    MutexWrapper() = default;
    ~MutexWrapper() = default;
    MutexWrapper([[maybe_unused]] const MutexWrapper& mutex_wrapper) { };
    MutexWrapper([[maybe_unused]] MutexWrapper&& mutex_wrapper) { };
    MutexWrapper& operator=([[maybe_unused]] const MutexWrapper& mutex_wrapper) { return *this; };
    MutexWrapper& operator=([[maybe_unused]] MutexWrapper&& mutex_wrapper) { return *this; };

    std::mutex mutex;
};

namespace sdmm {

template<typename SDMM, typename DMM, typename RNG>
struct SDMMContext {
    SDMMContext() = default;
    SDMMContext(size_t data_size) {
        data.reserve(data_size);
        training_data.reserve(data_size);
    }
    // Copy constructor intentionally deleted.
    // This is a large data structure and should not be copied.
    SDMMContext(SDMMContext&& other) = default;
    SDMMContext& operator=(SDMMContext&& other) = default;
    ~SDMMContext() = default;

    SDMM sdmm;
    DMM dmm;

    sdmm::Data<SDMM> data;
    sdmm::EM<DMM> em;
    RNG rng;

    sdmm::Data<DMM> training_data;
    bool update_ready = false;
    bool initialized = false;

    MutexWrapper mutex_wrapper;

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(SDMMContext, dmm);
};

}
