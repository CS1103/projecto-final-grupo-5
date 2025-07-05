#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include "interfaces.h"
#include "loss.h"
#include <memory>
#include <vector>
#include <iostream>
#include "optimizer.h"

namespace utec::neural_network {

template <typename T>
class NeuralNetwork {
private:
    std::vector<std::unique_ptr<ILayer<T>>> layers_;
    bool verbose_ = false;  // Add verbose flag

public:
    void add_layer(std::unique_ptr<ILayer<T>> layer) {
        layers_.push_back(std::move(layer));
    }

    void set_verbose(bool verbose) { verbose_ = verbose; }  // Add setter

    Tensor<T, 2> forward(const Tensor<T, 2>& x) {
        Tensor<T, 2> output = x;
        for (auto& layer : layers_) {
            output = layer->forward(output);
        }
        return output;
    }

    void backward(const Tensor<T, 2>& grad) {
        Tensor<T, 2> current_grad = grad;
        for (auto it = layers_.rbegin(); it != layers_.rend(); ++it) {
            current_grad = (*it)->backward(current_grad);
        }
    }

    void update_params(IOptimizer<T>& optimizer) {
        for (auto& layer : layers_) {
            layer->update_params(optimizer);
        }
    }

    template <template <typename> class LossType,
              template <typename> class OptimizerType = SGD>
    void train(const Tensor<T,2>& X, const Tensor<T,2>& Y,
              const size_t epochs, const size_t batch_size, T learning_rate) {

        OptimizerType<T> optimizer(learning_rate);
        const size_t num_batches = (X.shape()[0] + batch_size - 1) / batch_size;

        for (size_t epoch = 0; epoch < epochs; ++epoch) {
            T total_loss = 0;

            for (size_t batch = 0; batch < num_batches; ++batch) {
                const size_t start = batch * batch_size;
                const size_t end = std::min(start + batch_size, X.shape()[0]);

                auto X_batch = X.slice(start, end);
                auto Y_batch = Y.slice(start, end);

                auto Y_pred = forward(X_batch);

                LossType<T> loss_func(Y_pred, Y_batch);
                T loss = loss_func.loss();
                total_loss += loss;

                auto grad = loss_func.loss_gradient();
                backward(grad);
                update_params(optimizer);
            }

            // Only print if verbose mode is enabled
            if (verbose_ && epoch % 100 == 0) {
                std::cout << "Epoch " << epoch << ", Loss: "
                          << total_loss / num_batches << std::endl;
            }
        }
    }

    Tensor<T,2> predict(const Tensor<T,2>& X) {
        return forward(X);
    }
};

} // namespace utec::neural_network

#endif // NEURAL_NETWORK_H