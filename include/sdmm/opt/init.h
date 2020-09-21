#pragma once

#include <cstdint>
#include <stdexcept>

#include <boost/math/distributions/chi_squared.hpp>

#include <enoki/array.h>

#include "sdmm/core/utils.h"
#include "sdmm/distributions/sdmm.h"
#include "sdmm/linalg/coordinate_system.h"
#include "sdmm/opt/data.h"

namespace sdmm {

template<typename SDMM, typename EM, typename Data, typename RNG>
void initialize(
    SDMM& sdmm,
    EM& em,
    Data& data,
    RNG rng,
    size_t n_spatial_components,
    float spatial_distance
) {
    using EmbeddedS = sdmm::embedded_s_t<SDMM>;
    using NormalS = sdmm::normal_s_t<Data>;
    using ScalarS = typename SDMM::ScalarS;
    using Vector3 = sdmm::Vector<float, 3>;
    using CoordinateSystem = linalg::CoordinateSystem<Vector3>;
    using Matrix3 = typename CoordinateSystem::Rotation;
    using Value = decltype(data.weight);

    ScalarS n_thetas = 2;
    ScalarS n_phis = 4;
    ScalarS n_components = n_spatial_components * n_thetas * n_phis;
    enoki::set_slices(sdmm, (size_t) n_components);
    em = enoki::zero<EM>((size_t) n_components);

    sdmm.cov = enoki::zero<decltype(sdmm.cov)>(n_components);
    static const boost::math::chi_squared chi_sqr(6); // TODO: try with 3
    static const ScalarS contained_mass = 0.90;
    static const ScalarS max_rad_sqr = (ScalarS) boost::math::quantile(chi_sqr, contained_mass);
    ScalarS width_var = 0.5 * spatial_distance * spatial_distance / max_rad_sqr;
    ScalarS depth_var = 3e-2 * 3e-2 / max_rad_sqr;

    ScalarS directional_var = 2.f * M_PI / 8.f;
    sdmm.cov(3, 3) = enoki::full<Value>(directional_var, (size_t) n_components);
    sdmm.cov(4, 4) = enoki::full<Value>(directional_var, (size_t) n_components);

    Value min_distances = enoki::full<Value>(std::numeric_limits<float>::infinity(), data.size);
    Value min_spatial_distances = enoki::full<Value>(std::numeric_limits<float>::infinity(), data.size);
    Value min_normal_distances = enoki::full<Value>(std::numeric_limits<float>::infinity(), data.size);
    sdmm::Categorical<Value> sampling_dist;
    enoki::set_slices(data.weight, data.size);
    sampling_dist.pmf = enoki::vectorize(
        [](auto&& w) {
            return enoki::min(enoki::max(w, 1e-3), 3);
        },
        data.weight
    );
    sampling_dist.prepare();
    const ScalarS SPATIAL_THRESHOLD = 2e-2 * 2e-2;
    const ScalarS NORMAL_THRESHOLD = 2e-1 * 2e-1;

    CoordinateSystem coordinates;
    size_t component_i = 0;
    for(size_t sc_i = 0; sc_i < n_spatial_components; ++sc_i) {
        // size_t point_i = enoki::min(rng.next_float32() * data.size, data.size - 1);
        size_t point_i = sdmm::sample(sampling_dist, rng);
        spdlog::info("Found point={}", point_i);

        EmbeddedS mean = enoki::slice(data.point, point_i);
        Vector3 position(mean.coeff(0), mean.coeff(1), mean.coeff(2));

        NormalS n = enoki::slice(data.normal, point_i);
        coordinates.prepare(n);
        Vector3 s = coordinates.from.col(0);
        Vector3 t = coordinates.from.col(1);

        Matrix3 manifold_cov =
            width_var * linalg::outer(s) +
            width_var * linalg::outer(t) +
            depth_var * linalg::outer(n);

        Matrix3 depth_prior = linalg::outer(n) * 1e-6;

        // spdlog::info("n={}, s={}, t={}, manifold_cov={}", n, s, t, manifold_cov);
        for(size_t sc_j = 0; sc_j < data.size; ++sc_j) {
            NormalS n_j = enoki::slice(data.normal, sc_j);
            ScalarS normal_dot = enoki::dot(n, n_j);
            ScalarS normal_d = enoki::sqr(enoki::safe_acos(normal_dot) / M_PI);

            EmbeddedS mean_j = enoki::slice(data.point, sc_j);
            Vector3 position_j(mean_j.coeff(0), mean_j.coeff(1), mean_j.coeff(2));
            ScalarS position_d = enoki::squared_norm(position - position_j);

            ScalarS d = normal_d + position_d;

            if(d < min_distances.coeff(sc_j)) {
                min_distances.coeff(sc_j) = d;
                ScalarS metric = enoki::min(enoki::max(enoki::slice(data.weight, sc_j), 1e-3), 3);
                enoki::slice(sampling_dist.pmf, sc_j) = metric * enoki::pow(d, 5);
            }

            if(normal_d < NORMAL_THRESHOLD && position_d < min_spatial_distances.coeff(sc_j)) {
                min_spatial_distances.coeff(sc_j) = position_d;
                min_normal_distances.coeff(sc_j) = normal_d;
            }

            if(
                min_normal_distances.coeff(sc_j) < NORMAL_THRESHOLD &&
                min_spatial_distances.coeff(sc_j) < SPATIAL_THRESHOLD
            ) {
                min_distances.coeff(sc_j) = 0;
                enoki::slice(sampling_dist.pmf, sc_j) = 0;
            }
        }
        if(!sampling_dist.prepare()) {
            std::cerr <<
                "Could not create discrete CDF for initialization, "
                "using uniform CDF.\n";
            // std::cerr << fmt::format("Could not create CDF={}\n", sampling_dist.pmf);
            sampling_dist.pmf = enoki::full<Value>(1.f / enoki::slices(sampling_dist.pmf), enoki::slices(data.weight));
            // std::cerr << fmt::format("CDF after update={}\n", sampling_dist.pmf);
            sampling_dist.prepare();
        }

        ScalarS theta = 0;
        for(size_t theta_i = 0; theta_i < n_thetas; ++theta_i) {
            theta += 0.5 * M_PI / (n_thetas + 1.f);
            ScalarS cos_theta = enoki::cos(theta);
            ScalarS sin_theta = enoki::safe_sqrt(1.f - cos_theta * cos_theta);
            ScalarS phi = 0;
            for(size_t phi_i = 0; phi_i < n_phis; ++phi_i) {
                phi += 2 * M_PI / n_phis;
                auto [sin_phi, cos_phi] = enoki::sincos(phi);

                sdmm::Vector<float, 3> direction(
                    sin_theta * cos_phi, sin_theta * sin_phi, cos_theta
                );
                direction = coordinates.from * direction;
                mean = EmbeddedS(
                    mean.coeff(0),
                    mean.coeff(1),
                    mean.coeff(2),
                    direction.coeff(0),
                    direction.coeff(1),
                    direction.coeff(2)
                );
                enoki::slice(sdmm.tangent_space, component_i).set_mean(mean);

                for(size_t r = 0; r < 3; ++r) {
                    for(size_t c = 0; c < 3; ++c) {
                        enoki::slice(em.depth_prior, component_i)(r, c) = depth_prior(r, c);
                        enoki::slice(sdmm.cov, component_i)(r, c) = manifold_cov(r, c);
                    }
                }
                // spdlog::info("n={}, s={}, t={}, cov=\n{}", n, s, t, enoki::slice(sdmm.cov, component_i));
                ++component_i;
            }
        }
    }

    sdmm.weight.pmf = enoki::full<Value>(1.f / n_components, (size_t) n_components);
    sdmm::prepare_vectorized(sdmm);

    ScalarS weight_prior = 1.f / n_components;
    ScalarS cov_prior_strength = 5.f / n_components;
    sdmm::matrix_t<SDMM> cov_prior = enoki::zero<sdmm::matrix_t<SDMM>>((size_t) n_components);
    for(size_t slice_i = 0; slice_i < n_components; ++slice_i) {
        enoki::slice(cov_prior, slice_i) = sdmm::matrix_s_t<SDMM>(
            2e-3, 0, 0, 0, 0,
            0, 2e-3, 0, 0, 0,
            0, 0, 2e-3, 0, 0,
            0, 0, 0, 2e-4, 0,
            0, 0, 0, 0, 2e-4
        );
    }
    em.set_priors(weight_prior, cov_prior_strength, cov_prior);
}

}
