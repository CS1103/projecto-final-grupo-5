#ifndef TENSOR_H
#define TENSOR_H

#include <array>
#include <vector>
#include <stdexcept>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <initializer_list>

namespace utec::algebra {

template <typename T, std::size_t Rank>
class Tensor {
private:
    std::array<std::size_t, Rank> shape_;
    std::vector<T> data_;

    std::size_t compute_size(const std::array<std::size_t, Rank>& shape) const {
        return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
    }

    template <typename... Idxs>
    std::array<std::size_t, Rank> make_index_array(Idxs... idxs) const {
        return {static_cast<std::size_t>(idxs)...};
    }

    std::size_t compute_linear_index(const std::array<std::size_t, Rank>& indices) const {
        std::size_t index = 0;
        std::size_t stride = 1;
        for (int i = Rank - 1; i >= 0; --i) {
            if (indices[i] >= shape_[i])
                throw std::out_of_range("Index out of range");
            index += indices[i] * stride;
            stride *= shape_[i];
        }
        return index;
    }

public:
    // Constructor por defecto
    Tensor() {
        for (auto& dim : shape_) dim = 0;
        data_ = {};
    }

    explicit Tensor(const std::array<std::size_t, Rank>& shape)
        : shape_(shape), data_(compute_size(shape)) {}

    template <typename... Dims>
    Tensor(Dims... dims) : shape_{static_cast<std::size_t>(dims)...} {
        static_assert(sizeof...(Dims) == Rank, "Incorrect number of dimensions");
        data_.resize(compute_size(shape_));
    }

    // Asignación desde lista de inicialización
    Tensor& operator=(std::initializer_list<T> list) {
        if (list.size() != data_.size()) {
            throw std::invalid_argument("Initializer list size does not match tensor size");
        }
        std::copy(list.begin(), list.end(), data_.begin());
        return *this;
    }

    // Acceso a elementos
    template <typename... Idxs>
    T& operator()(Idxs... idxs) {
        auto indices = make_index_array(idxs...);
        return data_[compute_linear_index(indices)];
    }

    template <typename... Idxs>
    const T& operator()(Idxs... idxs) const {
        auto indices = make_index_array(idxs...);
        return data_[compute_linear_index(indices)];
    }

    T& operator[](std::size_t index) { return data_[index]; }
    const T& operator[](std::size_t index) const { return data_[index]; }

    const std::array<std::size_t, Rank>& shape() const noexcept { return shape_; }

    void reshape(const std::array<std::size_t, Rank>& new_shape) {
        if (compute_size(new_shape) != data_.size())
            throw std::invalid_argument("Reshape must preserve total elements");
        shape_ = new_shape;
    }

    void fill(const T& value) noexcept {
        std::fill(data_.begin(), data_.end(), value);
    }

    // Función slice para tensores 2D
    Tensor<T, Rank> slice(std::size_t start, std::size_t end) const {
        static_assert(Rank == 2, "Slice only for 2D tensors");
        if (end > shape_[0]) end = shape_[0];

        // Usar constructor explícito para evitar ambigüedad
        std::array<std::size_t, 2> new_shape = {end - start, shape_[1]};
        Tensor<T, Rank> result(new_shape);

        for (std::size_t i = start; i < end; ++i) {
            for (std::size_t j = 0; j < shape_[1]; ++j) {
                result(i - start, j) = (*this)(i, j);
            }
        }
        return result;
    }

    auto begin() { return data_.begin(); }
    auto end() { return data_.end(); }
    auto begin() const { return data_.begin(); }
    auto end() const { return data_.end(); }

    friend std::ostream& operator<<(std::ostream& os, const Tensor<T, Rank>& tensor) {
        if constexpr (Rank == 1) {
            os << "{";
            for (size_t i = 0; i < tensor.shape()[0]; ++i) {
                os << tensor(i);
                if (i < tensor.shape()[0]-1) os << " ";
            }
            os << "}";
        } else if constexpr (Rank == 2) {
            os << "{\n";
            for (size_t i = 0; i < tensor.shape()[0]; ++i) {
                for (size_t j = 0; j < tensor.shape()[1]; ++j) {
                    os << tensor(i, j);
                    if (j < tensor.shape()[1]-1) os << " ";
                }
                if (i < tensor.shape()[0]-1) os << "\n";
            }
            os << "\n}";
        } else {
            os << "Tensor<Rank=" << Rank << ">";
        }
        return os;
    }

    std::size_t size() const noexcept {
        return data_.size();
    }
    Tensor operator/(const T& scalar) const {
        static_assert(!std::is_integral<T>::value || !std::is_floating_point<decltype(scalar)>::value,
                      "Division of integer tensor by floating point requires explicit casting");

        Tensor result(shape_);
        for (std::size_t i = 0; i < data_.size(); ++i) {
            result.data_[i] = data_[i] / scalar;
        }
        return result;
    }

    // Operador de división por escalar (versión no miembro para permitir escalar / tensor)
    friend Tensor operator/(const T& scalar, const Tensor& tensor) {
        Tensor result(tensor.shape_);
        for (std::size_t i = 0; i < tensor.data_.size(); ++i) {
            result.data_[i] = scalar / tensor.data_[i];
        }
        return result;
    }
};

// Apply function to each element of the tensor
template <typename T, std::size_t Rank, typename F>
Tensor<T, Rank> apply(const Tensor<T, Rank>& tensor, F func) {
    Tensor<T, Rank> result(tensor.shape());
    for (std::size_t i = 0; i < tensor.size(); ++i) {
        result[i] = func(tensor[i]);
    }
    return result;
}

} // namespace utec::algebra

using utec::algebra::Tensor;
#endif // TENSOR_H