#pragma once

#include <mutex>

#include <sdmm/accelerators/spatial_stats.h>
#include <sdmm/core/utils.h>
#include <sdmm/opt/em.h>

namespace sdmm {

template <typename SDMM, typename DMM, typename RNG>
struct DMMContext {
    DMMContext() {
        data.clear();
        training_data.clear();
        stats.clear();
    };

    DMMContext(size_t data_size) {
        data.reserve(data_size);
        training_data.reserve(data_size);
        stats.clear();
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
    sdmm::SpatialStats stats;
    RNG rng;

    sdmm::Data<DMM> training_data;
    bool update_ready = false;
    bool initialized = false;

    MutexWrapper mutex_wrapper;

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(DMMContext, dmm);
};

} // namespace sdmm
