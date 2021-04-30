#pragma once

#include <mutex>

#include <sdmm/core/utils.h>
#include <sdmm/opt/em.h>

namespace sdmm {

template <typename SDMM, typename DMM, typename RNG>
struct DMMContext {
    DMMContext() = default;
    DMMContext(size_t data_size) {
        data.reserve(data_size);
        training_data.reserve(data_size);
    }
    // Copy constructor intentionally deleted.
    // This is a large data structure and should not be copied.
    DMMContext(DMMContext&& other) = default;
    DMMContext& operator=(DMMContext&& other) = default;
    ~DMMContext() = default;

    SDMM sdmm;
    DMM dmm;

    sdmm::Data<SDMM> data;
    sdmm::EM<DMM> em;
    RNG rng;

    sdmm::Data<DMM> training_data;
    bool update_ready = false;
    bool initialized = false;

    MutexWrapper mutex_wrapper;

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(DMMContext, dmm);
};

} // namespace sdmm
