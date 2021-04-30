#pragma once

#include <mutex>

#include <sdmm/core/utils.h>
#include <sdmm/opt/em.h>

namespace sdmm {

template <typename JointSDMM, typename Conditioner, typename RNG>
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

    JointSDMM sdmm;
    Conditioner conditioner;
    sdmm::Data<JointSDMM> data;
    sdmm::EM<JointSDMM> em;
    RNG rng;

    sdmm::Data<JointSDMM> training_data;
    bool update_ready = false;
    bool initialized = false;

    MutexWrapper mutex_wrapper;

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(SDMMContext, sdmm);
};

} // namespace sdmm
