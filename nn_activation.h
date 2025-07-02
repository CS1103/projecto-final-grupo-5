#ifndef NN_ACTIVATION_H
#define NN_ACTIVATION_H

#include <valarray>

#include "nn_interfaces.h"

namespace utec::neural_network {

template <typename T>
class ReLU final : public ILayer<T> {
public:
    Tensor<T, 2> forward(const Tensor<T, 2>& input) override {
        mask_ = input;
        Tensor<T, 2> output(input.shape()[0], input.shape()[1]);
        for (size_t i = 0; i < input.shape()[0]; ++i) {
            for (size_t j = 0; j < input.shape()[1]; ++j) {
                output(i, j) = input(i, j) > 0 ? input(i, j) : 0;
            }
        }
        return output;
    }

    Tensor<T, 2> backward(const Tensor<T, 2>& grad) override {
        Tensor<T, 2> output(grad.shape()[0], grad.shape()[1]);
        for (size_t i = 0; i < grad.shape()[0]; ++i) {
            for (size_t j = 0; j < grad.shape()[1]; ++j) {
                output(i, j) = grad(i, j) * (mask_(i, j) > 0 ? 1 : 0);
            }
        }
        return output;
    }

    void update_params(IOptimizer<T>& optimizer) override {}

private:
    Tensor<T, 2> mask_;
};

template <typename T>
class Sigmoid final : public ILayer<T> {
public:
    Tensor<T, 2> forward(const Tensor<T, 2>& input) override {
        output_ = input;
        for (size_t i = 0; i < input.shape()[0]; ++i) {
            for (size_t j = 0; j < input.shape()[1]; ++j) {
                T x = input(i, j);
                output_(i, j) = 1.0 / (1.0 + std::exp(-x));
            }
        }
        return output_;
    }

    Tensor<T, 2> backward(const Tensor<T, 2>& grad) override {
        Tensor<T, 2> result(grad.shape()[0], grad.shape()[1]);
        for (size_t i = 0; i < grad.shape()[0]; ++i) {
            for (size_t j = 0; j < grad.shape()[1]; ++j) {
                T s = output_(i, j);
                result(i, j) = grad(i, j) * s * (1 - s);
            }
        }
        return result;
    }

    void update_params(IOptimizer<T>& optimizer) override {}

private:
    Tensor<T, 2> output_;
};

} // namespace utec::neural_network

#endif // NN_ACTIVATION_H