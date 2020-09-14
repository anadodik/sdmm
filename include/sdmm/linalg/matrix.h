#pragma once

#include <fmt/ostream.h>
#include <spdlog/spdlog.h>

#include "enoki/array.h"
#include "enoki/matrix.h"

#include "sdmm/linalg/vector.h"

namespace sdmm::linalg {

template <typename Value_, size_t Rows_, size_t Cols_=Rows_>
struct Matrix : enoki::StaticArrayImpl<enoki::Array<Value_, Rows_>, Cols_, false, Matrix<Value_, Rows_, Cols_>> {
    static constexpr size_t Rows = Rows_;
    static constexpr size_t Cols = Cols_;
    static_assert(Rows > 1 && Cols > 1);

    using Entry = Value_;
    using Column = enoki::Array<Entry, Rows_>;
    using MatrixTranspose = Matrix<Value_, Cols_, Rows_>;

    using Base = enoki::StaticArrayImpl<Column, Cols_, false, Matrix<Value_, Rows_, Cols_>>;
    using Base::coeff;

    ENOKI_ARRAY_IMPORT_BASIC(Base, Matrix);
    using Base::operator=;

    static constexpr bool IsMatrix = true;
    static constexpr bool IsVector = false;

    using ArrayType = Matrix;
    using MaskType = enoki::Mask<enoki::mask_t<Column>, Cols_>;

    template <typename T> using ReplaceValue = Matrix<enoki::value_t<T>, Rows_, Cols_>;
    template <size_t R, size_t C> using ReplaceSize = Matrix<enoki::value_t<Value>, R, C>;

    Matrix() = default;

    /// Initialize from a compatible matrix
    template <
        typename Value2,
        size_t Rows2,
        size_t Cols2,
        enoki::enable_if_t<Rows2 == Rows_ && Cols2 == Cols_> = 0
    >
    ENOKI_INLINE Matrix(const Matrix<Value2, Rows2, Cols2> &m)
     : Base(m) { }

    template <typename T, enoki::enable_if_t<(enoki::array_depth_v<T> <= Base::Depth - 2)> = 0,
                          enoki::enable_if_not_matrix_t<T> = 0>
    ENOKI_INLINE Matrix(T&& v) {
        for (size_t i = 0; i < Size; ++i) {
            coeff(i) = enoki::zero<Column>();
            coeff(i, i) = v;
        }
    }

    template <typename T, enoki::enable_if_t<(enoki::array_depth_v<T> == Base::Depth)> = 0,
                          enoki::enable_if_not_matrix_t<T> = 0>
    ENOKI_INLINE Matrix(T&& v) : Base(std::forward<T>(v)) { }

    /// Initialize the matrix from a list of columns
    template <typename... Args, enoki::enable_if_t<sizeof...(Args) == Cols_ &&
              std::conjunction_v<std::is_constructible<Column, Args>...>> = 0>
    ENOKI_INLINE Matrix(Args&&... args) : Base(std::forward<Args>(args)...) { }

    /// Initialize the matrix from a list of entries in row-major order
    template <typename... Args, enoki::enable_if_t<sizeof...(Args) == Cols_ * Rows_ &&
              sizeof...(Args) != Cols_  &&
              std::conjunction_v<std::is_constructible<Entry, Args>...>> = 0>
    ENOKI_INLINE Matrix(const Args&... args) {
        alignas(alignof(Column)) Entry values[sizeof...(Args)] = { Entry(args)... };
        for (size_t j = 0; j < Size; ++j)
            for (size_t i = 0; i < Size; ++i)
                coeff(j, i) = values[i * Size + j];
    }

    template <typename... Column>
    ENOKI_INLINE static Matrix from_cols(const Column&... cols) {
        return Matrix(cols...);
    }

    template <typename... Row>
    ENOKI_INLINE static Matrix from_rows(const Row&... rows) {
        return transpose(Matrix(rows...));
    }

    static ENOKI_INLINE Derived zero_(size_t size) {
        Derived result;
        for (size_t i = 0; i < Size; ++i)
            result.coeff(i) = enoki::zero<Column>(size);
        return result;
    }

    ENOKI_INLINE decltype(auto) operator()(size_t i, size_t j) { return coeff(j, i); }
    ENOKI_INLINE decltype(auto) operator()(size_t i, size_t j) const { return coeff(j, i); }

    ENOKI_INLINE Column& col(size_t index) { return coeff(index); }
    ENOKI_INLINE const Column& col(size_t index) const { return coeff(index); }
};

template <typename Scalar, size_t Rows>
ENOKI_INLINE auto outer(const Vector<Scalar, Rows> &s) {
    using EValue  = enoki::expr_t<Scalar>;
    using EMatrix = Matrix<EValue, Rows>;
    EMatrix sum;
    for (size_t c = 0; c < Rows; ++c) {
        for (size_t r = 0; r < Rows; ++r) {
            sum(r, c) = s.coeff(r) * s.coeff(c);
        }
    }
    return sum;
}

template <typename T0, typename T1, size_t Rows1, size_t Rows2,
          size_t Cols1, size_t Cols2,
          typename Result = Matrix<enoki::expr_t<T0, T1>, Rows1, Cols2>,
          typename Column = enoki::column_t<Result>>
ENOKI_INLINE Result operator*(const Matrix<T0, Rows1, Cols1> &m0,
                              const Matrix<T1, Rows2, Cols2> &m1) {
    static_assert(Rows2 == Cols1);
    Result result;
    for (size_t j = 0; j < Cols2; ++j) {
        Column sum = m0.coeff(0) * Column::full_(m1(0, j), 1);
        for (size_t i = 1; i < Cols1; ++i)
            sum = enoki::fmadd(m0.coeff(i), Column::full_(m1(i, j), 1), sum);
        result.coeff(j) = sum;
    }

    return result;
}

template <typename T0, typename T1, size_t Rows, size_t Cols, enoki::enable_if_t<!T1::IsMatrix> = 0>
ENOKI_INLINE auto operator*(const sdmm::linalg::Matrix<T0, Rows, Cols> &m, const T1 &s) {
    if constexpr (enoki::array_size_v<T1> == Cols && T1::IsVector && !std::is_same_v<T1, T0>) {
        using EValue  = enoki::expr_t<T0, enoki::value_t<T1>>;
        using EVector = enoki::Array<EValue, Rows>;
        EVector sum = enoki::zero<EVector>();
        for (size_t c = 0; c < Cols; ++c) {
            for (size_t r = 0; r < Rows; ++r) {
                sum.coeff(r) = enoki::fmadd(m(r, c), s.coeff(c), sum.coeff(r));
            }
        }
        return sum;
        // EVector sum = m.coeff(0) * EVector::full_(s.coeff(0), 1);
        // for (size_t i = 1; i < Cols; ++i) {
        //     sum = enoki::fmadd(m.coeff(i), EVector::full_(s.coeff(i), 1), sum);
        // }
        // return sum;
    } else {
        using EValue  = enoki::expr_t<T0, T1>;
        using EArray  = enoki::Array<enoki::Array<EValue, Rows>, Cols>;
        using EMatrix = sdmm::linalg::Matrix<EValue, Rows, Cols>;

        return EMatrix(EArray(m) * EArray::full_(EValue(s), 1));
    }
}

template <typename Value_, size_t Rows_, size_t Cols_>
ENOKI_INLINE auto transpose(const Matrix<Value_, Rows_, Cols_>& matrix) {
    using Matrix = Matrix<Value_, Rows_, Cols_>;
    enoki::expr_t<typename Matrix::MatrixTranspose> result;
    for (size_t i = 0; i < Cols_; ++i)
        for (size_t j = 0; j < Rows_; ++j)
            result.coeff(j, i) = matrix.derived().coeff(i, j);
    return result;
}

template <typename T, enoki::enable_if_matrix_t<T> = 0>
ENOKI_INLINE T identity(size_t size = 1) {
    T result = enoki::zero<T>(size);
    for (size_t i = 0; i < T::Cols; ++i)
        result(i, i) = enoki::full<typename T::Entry>(enoki::scalar_t<T>(1.f), size);
    return result;
}

}

NAMESPACE_BEGIN(enoki)

template <typename T, size_t Rows, size_t Cols>
struct struct_support<sdmm::linalg::Matrix<T, Rows, Cols>,
                      enable_if_static_array_t<sdmm::linalg::Matrix<T, Rows, Cols>>> {
    static constexpr bool IsDynamic = enoki::is_dynamic_v<T>;
    using Dynamic = sdmm::linalg::Matrix<enoki::make_dynamic_t<T>, Rows, Cols>;
    using Value = sdmm::linalg::Matrix<T, Rows, Cols>;
    using Column = column_t<Value>;

    static ENOKI_INLINE size_t slices(const Value &value) {
        return enoki::slices(value.coeff(0, 0));
    }

    static ENOKI_INLINE size_t packets(const Value &value) {
        return enoki::packets(value.coeff(0, 0));
    }

    static ENOKI_INLINE void set_slices(Value &value, size_t size) {
        for (size_t i = 0; i < Cols; ++i)
            enoki::set_slices(value.coeff(i), size);
    }

    template <typename T2>
    static ENOKI_INLINE auto packet(T2&& value, size_t i) {
        return packet(value, i, std::make_index_sequence<Cols>());
    }

    template <typename T2>
    static ENOKI_INLINE auto slice(T2&& value, size_t i) {
        return slice(value, i, std::make_index_sequence<Cols>());
    }

    template <typename T2>
    static ENOKI_INLINE auto slice_ptr(T2&& value, size_t i) {
        return slice_ptr(value, i, std::make_index_sequence<Cols>());
    }

    template <typename T2>
    static ENOKI_INLINE auto ref_wrap(T2&& value) {
        return ref_wrap(value, std::make_index_sequence<Cols>());
    }

    template <typename T2>
    static ENOKI_INLINE auto detach(T2&& value) {
        return detach(value, std::make_index_sequence<Cols>());
    }

    template <typename T2>
    static ENOKI_INLINE auto gradient(T2&& value) {
        return gradient(value, std::make_index_sequence<Cols>());
    }

    static ENOKI_INLINE Value zero(size_t size) {
        return Value::zero_(size);
    }

    static ENOKI_INLINE Value empty(size_t size) {
        return Value::empty_(size);
    }

    template <typename T2, typename Mask,
              enable_if_t<array_size<T2>::value == array_size<Mask>::value> = 0>
    static ENOKI_INLINE auto masked(T2 &value, const Mask &mask) {
        return detail::MaskedArray<T2>{ value, mask_t<T2>(mask) };
    }

    template <typename T2, typename Mask,
              enable_if_t<array_size<T2>::value != array_size<Mask>::value> = 0>
    static ENOKI_INLINE auto masked(T2 &value, const Mask &mask) {
        using Arr = Array<Array<T, Rows>, Cols>;
        return enoki::masked((Arr&) value, mask_t<Arr>(mask));
    }

private:
    template <typename T2, size_t... Index>
    static ENOKI_INLINE auto packet(T2&& value, size_t i, std::index_sequence<Index...>) {
        return sdmm::linalg::Matrix<decltype(enoki::packet(value.coeff(0, 0), i)), Rows, Cols>(
            enoki::packet(value.coeff(Index), i)...);
    }

    template <typename T2, size_t... Index>
    static ENOKI_INLINE auto slice(T2&& value, size_t i, std::index_sequence<Index...>) {
        return sdmm::linalg::Matrix<decltype(enoki::slice(value.coeff(0, 0), i)), Rows, Cols>(
            enoki::slice(value.coeff(Index), i)...);
    }

    template <typename T2, size_t... Index>
    static ENOKI_INLINE auto slice_ptr(T2&& value, size_t i, std::index_sequence<Index...>) {
        return sdmm::linalg::Matrix<decltype(enoki::slice_ptr(value.coeff(0, 0), i)), Rows, Cols>(
            enoki::slice_ptr(value.coeff(Index), i)...);
    }

    template <typename T2, size_t... Index>
    static ENOKI_INLINE auto ref_wrap(T2&& value, std::index_sequence<Index...>) {
        return sdmm::linalg::Matrix<decltype(enoki::ref_wrap(value.coeff(0, 0))), Rows, Cols>(
            enoki::ref_wrap(value.coeff(Index))...);
    }

    template <typename T2, size_t... Index>
    static ENOKI_INLINE auto detach(T2&& value, std::index_sequence<Index...>) {
        return sdmm::linalg::Matrix<decltype(enoki::detach(value.coeff(0, 0))), Rows, Cols>(
            enoki::detach(value.coeff(Index))...);
    }

    template <typename T2, size_t... Index>
    static ENOKI_INLINE auto gradient(T2&& value, std::index_sequence<Index...>) {
        return sdmm::linalg::Matrix<decltype(enoki::gradient(value.coeff(0, 0))), Rows, Cols>(
            enoki::gradient(value.coeff(Index))...);
    }
};

NAMESPACE_END(enoki)
