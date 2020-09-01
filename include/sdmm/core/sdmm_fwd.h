#pragma once

#include "sdmm/core/constants.h"
#include "sdmm/core/utils.h"

namespace sdmm {

template<typename Matrix_, typename TangentSpace_> struct SDMM;

template<typename Joint_, typename Marginal_, typename Conditional_> struct SDMMConditioner;

template<typename Embedded_, typename Tangent_> struct EuclidianTangentSpace;

template<typename Embedded_, typename Tangent_> struct DirectionalTangentSpace;

template<typename Embedded_, typename Tangent_> struct SpatioDirectionalTangentSpace;

}
