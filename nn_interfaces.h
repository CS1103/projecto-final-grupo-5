#ifndef NN_INTERFACES_H
#define NN_INTERFACES_H

#include "tensor.h"

namespace utec::neural_network {

    template <typename T>
    class IOptimizer;

    template <typename T>
    class ILayer {
    public:
        virtual ~ILayer() = default;
        virtual Tensor<T, 2> forward(const Tensor<T, 2>& input) = 0;
        virtual Tensor<T, 2> backward(const Tensor<T, 2>& grad) = 0;
        virtual void update_params(IOptimizer<T>& optimizer) = 0;
    };

    template <typename T, std::size_t Rank>
    class ILoss {
    public:
        virtual ~ILoss() = default;
        virtual T loss() const = 0;
        virtual Tensor<T, Rank> loss_gradient() const = 0;
    };

    template <typename T>
    class IOptimizer {
    public:
        virtual ~IOptimizer() = default;
        virtual void update(Tensor<T, 2>& params, const Tensor<T, 2>& grads) = 0;
        virtual void step() {}
    };

} // namespace utec::neural_network

#endif // NN_INTERFACES_H