#ifndef NN_OPTIMIZER_H
#define NN_OPTIMIZER_H

#include "nn_interfaces.h"
#include <cmath>

namespace utec::neural_network {

    template <typename T>
    class SGD final : public IOptimizer<T> {
    private:
        T learning_rate_;
    public:
        explicit SGD(T learning_rate = 0.01) : learning_rate_(learning_rate) {}

        void update(Tensor<T, 2>& params, const Tensor<T, 2>& grads) override {
            for (size_t i = 0; i < params.shape()[0]; ++i) {
                for (size_t j = 0; j < params.shape()[1]; ++j) {
                    params(i, j) -= learning_rate_ * grads(i, j);
                }
            }
        }
    };

    template <typename T>
    class Adam final : public IOptimizer<T> {
    private:
        T learning_rate_;
        T beta1_;
        T beta2_;
        T epsilon_;
        std::size_t t_;
        Tensor<T,2> m_;
        Tensor<T,2> v_;

    public:
        explicit Adam(T learning_rate = 0.001, T beta1 = 0.9, T beta2 = 0.999, T epsilon = 1e-8)
            : learning_rate_(learning_rate), beta1_(beta1), beta2_(beta2), epsilon_(epsilon), t_(0) {}

        void update(Tensor<T, 2>& params, const Tensor<T, 2>& grads) override {
            if (t_ == 0) {
                m_ = Tensor<T,2>(params.shape()[0], params.shape()[1]);
                v_ = Tensor<T,2>(params.shape()[0], params.shape()[1]);
                m_.fill(0);
                v_.fill(0);
            }
            t_++;

            for (size_t i = 0; i < params.shape()[0]; ++i) {
                for (size_t j = 0; j < params.shape()[1]; ++j) {
                    T g = grads(i, j);
                    m_(i, j) = beta1_ * m_(i, j) + (1 - beta1_) * g;
                    v_(i, j) = beta2_ * v_(i, j) + (1 - beta2_) * g * g;

                    T m_hat = m_(i, j) / (1 - std::pow(beta1_, t_));
                    T v_hat = v_(i, j) / (1 - std::pow(beta2_, t_));

                    params(i, j) -= learning_rate_ * m_hat / (std::sqrt(v_hat) + epsilon_);
                }
            }
        }

        void step() override {}
    };

} // namespace utec::neural_network

#endif // NN_OPTIMIZER_H