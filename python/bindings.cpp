#include <enoki/python.h>
#include <enoki/random.h>

#include <fmt/format.h>

#include "sdmm/spaces/euclidian.h"
#include "sdmm/spaces/directional.h"
#include "sdmm/spaces/spatio_directional.h"

#include "sdmm/distributions/categorical.h"
#include "sdmm/distributions/sdmm.h"
#include "sdmm/distributions/sdmm_conditioner.h"

#include "sdmm/opt/em.h"

namespace py = pybind11;
using namespace py::literals;

constexpr static size_t PacketSize = 4;
using Scalar = float;
using Packet = enoki::Packet<Scalar, PacketSize>;
using Value = enoki::DynamicArray<Packet>;
using RNG = enoki::PCG32<float>;


using JointEmbedded = sdmm::Vector<Value, 4 + 1>;
using JointTangent = sdmm::Vector<Value, 4>;
using TangentSpace = sdmm::SpatioDirectionalTangentSpace<JointEmbedded, JointTangent>;
using SDMM = sdmm::SDMM<sdmm::Matrix<Value, 4>, TangentSpace>;
using Stats = sdmm::Stats<SDMM>;

using Tangent = sdmm::tangent_t<TangentSpace>;
using Embedded = sdmm::embedded_t<TangentSpace>;
using Scalar_ = enoki::value_t<Tangent>;

static_assert(std::is_same_v<
    decltype(enoki::packet(Stats(), 0))::ScalarExpr,
    Packet
>);

template<size_t Size>
auto add_euclidian(py::module& m) {
    using namespace sdmm;

    auto name = fmt::format("EuclidianTangentSpace{}", Size);
	using Embedded = sdmm::Vector<Value, Size>;
	using Tangent = sdmm::Vector<Value, Size>;
	using EmbeddedS = sdmm::Vector<Scalar, Size>;
	using TangentS = sdmm::Vector<Scalar, Size>;
    using TangentSpace = sdmm::EuclidianTangentSpace<Embedded, Tangent>;

    py::class_<TangentSpace>(m, name.c_str())
        .def(py::init<>())
        .def_readwrite("mean", &TangentSpace::mean)
        .def("set_mean", py::overload_cast<const Embedded&>(&TangentSpace::set_mean), "mean"_a)
        .def("to_tangent", &TangentSpace::template to<EmbeddedS>, "embedded"_a, "inv_jacobian"_a)
        .def("from_tangent", &TangentSpace::template from<TangentS>, "tangent"_a, "inv_jacobian"_a)
	;
}

auto add_directional(py::module& m) {
    using namespace sdmm;

    auto name = fmt::format("DirectionalTangentSpace");
	using Embedded = sdmm::Vector<Value, 3>;
	using Tangent = sdmm::Vector<Value, 2>;
	using EmbeddedS = sdmm::Vector<Scalar, 3>;
	using TangentS = sdmm::Vector<Scalar, 2>;
    using TangentSpace = sdmm::DirectionalTangentSpace<Embedded, Tangent>;

    using Matrix = typename TangentSpace::Matrix;

    py::class_<TangentSpace>(m, name.c_str())
        .def(py::init<>())
        .def_readwrite("mean", &TangentSpace::mean)
        .def("get_to", [](TangentSpace& ts) { return ts.coordinate_system.to; })
        .def("get_from", [](TangentSpace& ts) { return ts.coordinate_system.from; })
        .def("set_to", [](TangentSpace& ts, Matrix& m) { ts.coordinate_system.to = m; })
        .def("set_from", [](TangentSpace& ts, Matrix& m) { ts.coordinate_system.from = m; })
        .def("set_mean", py::overload_cast<const Embedded&>(&TangentSpace::set_mean), "mean"_a)
        .def("rotate_to_wo", &TangentSpace::rotate_to_wo, "wi"_a)
        .def("to_tangent", &TangentSpace::template to<EmbeddedS>, "embedded"_a, "inv_jacobian"_a)
        .def("from_tangent", &TangentSpace::template from<TangentS>, "tangent"_a, "inv_jacobian"_a)
        .def("to_center_jacobian", &TangentSpace::to_center_jacobian)
        .def("from_jacobian", &TangentSpace::template from_jacobian<TangentS>, "tangent"_a)
        .def("from_tangent_many", [](TangentSpace& ts, Tangent& tangent) -> std::pair<Embedded, Value> {
            if(enoki::slices(ts) != 1) {
                throw std::runtime_error("tangent_space can only have one slice for from_tangent_many");
            }
            size_t n_slices = enoki::slices(tangent);
            Value inv_jacobian = enoki::empty<Value>(1);

            Embedded embedded = enoki::empty<Embedded>(n_slices);
            Value inv_jacobians = enoki::empty<Value>(n_slices);
            for(size_t slice_i = 0; slice_i < n_slices; ++slice_i) {
                auto result = 
                    ts.from(enoki::slice(tangent, slice_i), inv_jacobian);
                enoki::slice(embedded, slice_i) = enoki::slice(result, 0);
                inv_jacobians.coeff(slice_i) = inv_jacobian.coeff(0);
            }
            return {embedded, inv_jacobians};
        })
	;
}

template<size_t Size>
auto add_spatio_directional(py::module& m) {
    using namespace sdmm;

    auto name = fmt::format("SpatioDirectionalTangentSpace{}", Size);
	using Embedded = sdmm::Vector<Value, Size + 1>;
	using Tangent = sdmm::Vector<Value, Size>;
	using EmbeddedS = sdmm::Vector<Scalar, Size + 1>;
	using TangentS = sdmm::Vector<Scalar, Size>;
    using TangentSpace = sdmm::SpatioDirectionalTangentSpace<Embedded, Tangent>;

    py::class_<TangentSpace>(m, name.c_str())
        .def(py::init<>())
        .def_readwrite("mean", &TangentSpace::mean)
        .def("set_mean", py::overload_cast<const Embedded&>(&TangentSpace::set_mean), "mean"_a)
        .def("to_tangent", &TangentSpace::template to<EmbeddedS>, "embedded"_a, "inv_jacobian"_a)
        .def("from_tangent", &TangentSpace::template from<TangentS>, "tangent"_a, "inv_jacobian"_a)
	;
}

template<typename SDMM>
auto add_mm_data(py::module& m, const std::string& name) {
    using Data = sdmm::Data<SDMM>; 
    py::class_<Data>(m, name.c_str())
        .def(py::init<>())
        .def("reserve", &Data::reserve, "new_capacity"_a)
        .def("clear", &Data::clear)
        .def_readwrite("size", &Data::size)
        .def_readwrite("capacity", &Data::capacity)
        .def_readwrite("point", &Data::point)
        .def_readwrite("normal", &Data::normal)
        .def_readwrite("weight", &Data::weight)
        .def_readwrite("heuristic_pdf", &Data::heuristic_pdf)
    ;
}

template<typename SDMM>
auto add_mm_em(py::module& m, const std::string& name) {
    using EM = sdmm::EM<SDMM>; 

    py::class_<EM>(m, name.c_str())
        .def(py::init<>([](size_t n_components) {
            EM em = enoki::zero<EM>(n_components);
            return em;
        }))
        .def(
            "step",
            [](EM& em, SDMM& sdmm, sdmm::Data<SDMM>& data) {
                sdmm::em_step(sdmm, em, data);
                return sdmm;
            }
        )
    ;
}

template<typename SDMM>
auto add_mm(py::module& m, const std::string& name) {
	using Embedded = sdmm::embedded_t<SDMM>;
	using EmbeddedS = sdmm::embedded_s_t<SDMM>;
	using TangentS = sdmm::tangent_s_t<SDMM>;

    py::class_<SDMM>(m, name.c_str())
        .def(py::init<>())
		.def_readwrite("weight", &SDMM::weight)
		.def_readwrite("tangent_space", &SDMM::tangent_space)
		.def_readwrite("cov", &SDMM::cov)
		.def_readwrite("compute_inverse", &SDMM::compute_inverse)
		.def("prepare", &SDMM::prepare)
		.def(
            "save",
            [](SDMM& sdmm, const std::string& path) {
                sdmm::save_json<SDMM>(sdmm, path);
            },
            "path"_a
        )
        .def(
            "load",
            [](SDMM& sdmm, const std::string& path) {
                sdmm::load_json<SDMM>(sdmm, path);
            },
            "path"_a
        )
		.def(
            "product",
            [](SDMM& first, SDMM& second) {
                SDMM result;
                sdmm::product(first, second, result);
                return result;
            }
        )
		.def(
            "product_approximate",
            [](SDMM& first, SDMM& second) {
                SDMM result;
                sdmm::product_approximate(first, second, result);
                return result;
            }
        )
		.def(
			"pdf",
			[](SDMM& sdmm, const Embedded& embedded) {
                Value pdf_single = enoki::empty<Value>(enoki::slices(sdmm)); 
                Value pdfs = enoki::empty<Value>(enoki::slices(embedded)); 
                for(size_t i = 0; i < enoki::slices(embedded); ++i) {
                    sdmm.posterior(enoki::slice(embedded, i), pdf_single);
                    pdfs.coeff(i) = enoki::hsum(pdf_single);
                }
                return pdfs;
			},
			"embedded"_a
		)
		.def(
            "sample", 
            [](SDMM& sdmm, RNG& rng, size_t n_samples) -> std::pair<Embedded, Value> {
                EmbeddedS sample;
                TangentS tangent;
                Scalar inv_jacobian;

                Embedded samples = enoki::empty<Embedded>(n_samples);
                Value inv_jacobians = enoki::empty<Value>(n_samples);
                for(size_t sample_i = 0; sample_i < n_samples; ++sample_i) {
                    sdmm.sample(rng, sample, inv_jacobian, tangent);
                    enoki::slice(samples, sample_i) = sample;
                    inv_jacobians.coeff(sample_i) = inv_jacobian;
                }
                return {samples, inv_jacobians};
            },
            "rng"_a,
            "n_samples"_a
        )
	;
}

template<typename Conditioner>
auto add_conditioner(py::module& m, const std::string& name) {
    using Joint = typename Conditioner::Joint;
    using Conditional = typename Conditioner::Conditional;
    using Point = typename Conditioner::MarginalEmbeddedS;

    py::class_<Conditioner>(m, name.c_str())
        .def(py::init<>())
		.def("prepare", [](Conditioner& conditioner, Joint& joint) {
            if(enoki::slices(conditioner) != enoki::slices(joint)) {
                enoki::set_slices(conditioner, enoki::slices(joint));
            }
            sdmm::prepare(conditioner, joint); 
        })
		.def("condition", [](Conditioner& conditioner, Point& point) {
            Conditional conditional;
            enoki::set_slices(conditional, enoki::slices(conditioner));
            sdmm::create_conditional(conditioner, point, conditional); 
            return conditional;
        })
		.def("condition_pruned", [](Conditioner& conditioner, Point& point, size_t max_components, size_t preserve_idx=-1) {
            Conditional conditional;
            enoki::set_slices(conditional, enoki::slices(conditioner));
            sdmm::create_conditional_pruned(conditioner, point, conditional, max_components, preserve_idx); 
            return conditional;
        })
    ;
}

template<size_t Size>
auto add_sdmm(py::module& dist_m, py::module& opt_m) {
	constexpr static size_t JointSize = Size;
	constexpr static size_t ConditionalSize = 2;
	constexpr static size_t MarginalSize = Size - ConditionalSize;

	using JointEmbedded = sdmm::Vector<Value, JointSize + 1>;
	using JointTangent = sdmm::Vector<Value, JointSize>;
    using JointTangentSpace = sdmm::SpatioDirectionalTangentSpace<JointEmbedded, JointTangent>;
    using JointSDMM = sdmm::SDMM<sdmm::Matrix<Value, JointSize>, JointTangentSpace>;

	using MarginalEmbedded = sdmm::Vector<Value, MarginalSize>;
	using MarginalTangent = sdmm::Vector<Value, MarginalSize>;
    using MarginalTangentSpace = sdmm::EuclidianTangentSpace<MarginalEmbedded, MarginalTangent>;
    using MarginalSDMM = sdmm::SDMM<sdmm::Matrix<Value, MarginalSize>, MarginalTangentSpace>;

	using ConditionalEmbedded = sdmm::Vector<Value, ConditionalSize + 1>;
	using ConditionalTangent = sdmm::Vector<Value, ConditionalSize>;
    using ConditionalTangentSpace = sdmm::DirectionalTangentSpace<ConditionalEmbedded, ConditionalTangent>;
    using ConditionalSDMM = sdmm::SDMM<sdmm::Matrix<Value, ConditionalSize>, ConditionalTangentSpace>;

    using Conditioner = sdmm::SDMMConditioner<JointSDMM, MarginalSDMM, ConditionalSDMM>;

    auto dist_name = fmt::format("SDMM{}", JointSize);
	add_mm<JointSDMM>(dist_m, dist_name);

    auto data_name = fmt::format("SDMMData{}", JointSize);
	add_mm_data<JointSDMM>(opt_m, data_name);

    auto em_name = fmt::format("SDMMEM{}", JointSize);
	add_mm_em<JointSDMM>(opt_m, em_name);

    auto conditioner_name = fmt::format("SDMMConditioner{}", Size);
	add_conditioner<Conditioner>(dist_m, conditioner_name);
}

template<size_t Size>
auto add_gmm(py::module& dist_m, py::module& opt_m) {
	using Embedded = sdmm::Vector<Value, Size>;
	using Tangent = sdmm::Vector<Value, Size>;
	using EmbeddedS = sdmm::Vector<Scalar, Size>;
	using TangentS = sdmm::Vector<Scalar, Size>;
    using TangentSpace = sdmm::EuclidianTangentSpace<Embedded, Tangent>;

    using SDMM = sdmm::SDMM<
        sdmm::Matrix<Value, Size>, TangentSpace
    >;

    auto dist_name = fmt::format("GMM{}", Size);
	add_mm<SDMM>(dist_m, dist_name);

    auto data_name = fmt::format("GMMData{}", Size);
	add_mm_data<SDMM>(opt_m, data_name);

    auto em_name = fmt::format("GMMEM{}", Size);
    add_mm_em<SDMM>(opt_m, em_name);
}

auto add_dmm(py::module& dist_m, py::module& opt_m) {
	using Embedded = sdmm::Vector<Value, 3>;
	using Tangent = sdmm::Vector<Value, 2>;
	using EmbeddedS = sdmm::Vector<Scalar, 3>;
	using TangentS = sdmm::Vector<Scalar, 2>;
    using TangentSpace = sdmm::DirectionalTangentSpace<Embedded, Tangent>;

    using SDMM = sdmm::SDMM<
        sdmm::Matrix<Value, 2>, TangentSpace
    >;

    auto dist_name = std::string("DMM");
	add_mm<SDMM>(dist_m, dist_name);

    auto data_name = fmt::format("DMMData");
	add_mm_data<SDMM>(opt_m, data_name);

    auto em_name = fmt::format("DMMEM");
	add_mm_em<SDMM>(opt_m, em_name);
}

PYBIND11_MODULE(pysdmm, m) {
    using namespace sdmm;

    m.doc() = "SDMM Mixture Model Fitting Library.";

    py::class_<RNG>(m, "RNG")
        .def(py::init<>())
    ;

    auto spaces_m = m.def_submodule("spaces", "SDMM spaces.");

    add_euclidian<1>(spaces_m);
    add_euclidian<2>(spaces_m);
    add_euclidian<3>(spaces_m);

    add_directional(spaces_m);

    add_spatio_directional<3>(spaces_m);
    add_spatio_directional<4>(spaces_m);
    add_spatio_directional<5>(spaces_m);

    auto dist_m = m.def_submodule("distributions", "SDMM distributions.");
    py::class_<Categorical<Value>>(dist_m, "Categorical")
        .def(py::init<>())
        .def_readwrite("pmf", &Categorical<Value>::pmf)
        .def_readwrite("cdf", &Categorical<Value>::cdf)
        .def("prepare", &Categorical<Value>::prepare)
    ;

	auto opt_m = m.def_submodule("opt", "Optimizers for SDMM distributions.");

    // add_sdmm<3>(dist_m, opt_m);
    add_sdmm<4>(dist_m, opt_m);
    add_sdmm<5>(dist_m, opt_m);

    add_gmm<2>(dist_m, opt_m);
    add_gmm<3>(dist_m, opt_m);
    add_gmm<4>(dist_m, opt_m);
    add_gmm<5>(dist_m, opt_m);

    add_dmm(dist_m, opt_m);
}
