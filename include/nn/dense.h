#ifndef DENSE_H
#define DENSE_H

#include "interfaces.h"
#include <functional>
#include <type_traits>
/// @file dense.h
/// @brief Implementación de una capa densa (fully connected) para redes neuronales
namespace utec::neural_network {
/// Capa densa (fully connected).
/// Realiza la operación: output = input * W + b
/// Guarda el input para su uso en backpropagation.
template <typename T>
class Dense final : public ILayer<T> {
private:
    Tensor<T, 2> W_, dW_;  ///< Pesos y su gradiente
    Tensor<T, 1> b_, db_;  ///< Bias y su gradiente
    Tensor<T, 2> last_x_;  ///< Cache del input para usar en backward

public:
    using Initializer = std::function<void(Tensor<T, 2>&)>;
    using InitializerBias = std::function<void(Tensor<T, 1>&)>;
    /// Constructor principal de la capa densa
    /// @param in_f Número de entradas
    /// @param out_f Número de salidas
    /// @param init_w_fun Función para inicializar los pesos
    /// @param init_b_fun Función para inicializar los bias
    template <typename InitW, typename InitB>
    Dense(size_t in_f, size_t out_f, InitW&& init_w_fun, InitB&& init_b_fun) {
        W_ = Tensor<T, 2>(in_f, out_f);
        dW_ = Tensor<T, 2>(in_f, out_f);
        init_w_fun(W_);

        b_ = Tensor<T, 1>(out_f);
        db_ = Tensor<T, 1>(out_f);

        // Create a 2D view for 1D bias to match initializer expectation
        Tensor<T, 2> b_view(1, out_f);
        init_b_fun(b_view);
        for (size_t j = 0; j < out_f; ++j) {
            b_(j) = b_view(0, j);
        }
    }
    /// Constructor alternativo que usa una sola función de inicialización
    /// para pesos y bias
    template <typename Init,
              typename = std::enable_if_t<
                  std::is_invocable_r_v<void, Init, Tensor<T, 2>&>
              >>
    Dense(size_t in_f, size_t out_f, Init&& init_fun)
        : Dense(in_f, out_f,
                [&](auto& tensor) { init_fun(tensor); },
                [&](auto& tensor) {
                    Tensor<T, 2> temp(1, tensor.size());
                    init_fun(temp);
                    for (size_t j = 0; j < tensor.size(); ++j) {
                        tensor[j] = temp(0, j);
                    }
                }) {}
    /// Propagación hacia adelante (forward pass)
    /// @param x Tensor de entrada
    /// @return Salida de la capa (x * W + b)
    Tensor<T, 2> forward(const Tensor<T, 2>& x) override {
        last_x_ = x;
        Tensor<T, 2> output(x.shape()[0], W_.shape()[1]);
        for (size_t i = 0; i < x.shape()[0]; ++i) {
            for (size_t j = 0; j < W_.shape()[1]; ++j) {
                T sum = 0;
                for (size_t k = 0; k < W_.shape()[0]; ++k) {
                    sum += x(i, k) * W_(k, j);
                }
                output(i, j) = sum + b_(j);
            }
        }
        return output;
    }
    /// Retropropagación del error (backward pass)
    /// Calcula los gradientes y devuelve el gradiente respecto a la entrada
    /// @param dZ Gradiente recibido desde la capa superior
    /// @return Gradiente respecto a la entrada (para capa anterior)
    Tensor<T, 2> backward(const Tensor<T, 2>& dZ) override {
        dW_.fill(0);
        for (size_t i = 0; i < last_x_.shape()[0]; ++i) {
            for (size_t k = 0; k < last_x_.shape()[1]; ++k) {
                for (size_t j = 0; j < dZ.shape()[1]; ++j) {
                    dW_(k, j) += last_x_(i, k) * dZ(i, j);
                }
            }
        }

        db_.fill(0);
        for (size_t i = 0; i < dZ.shape()[0]; ++i) {
            for (size_t j = 0; j < dZ.shape()[1]; ++j) {
                db_(j) += dZ(i, j);
            }
        }

        algebra::Tensor<T, 2> dX(last_x_.shape()[0], last_x_.shape()[1]);
        for (size_t i = 0; i < dX.shape()[0]; ++i) {
            for (size_t k = 0; k < dX.shape()[1]; ++k) {
                T sum = 0;
                for (size_t j = 0; j < W_.shape()[1]; ++j) {
                    sum += dZ(i, j) * W_(k, j);
                }
                dX(i, k) = sum;
            }
        }
        return dX;
    }
    /// Actualiza los parámetros W y b usando el optimizador proporcionado
    void update_params(IOptimizer<T>& optimizer) override {
        optimizer.update(W_, dW_);

        algebra::Tensor<T, 2> b2d(1, b_.shape()[0]);
        algebra::Tensor<T, 2> db2d(1, db_.shape()[0]);
        for (size_t j = 0; j < b_.shape()[0]; ++j) {
            b2d(0, j) = b_(j);
            db2d(0, j) = db_(j);
        }

        optimizer.update(b2d, db2d);

        for (size_t j = 0; j < b_.shape()[0]; ++j) {
            b_(j) = b2d(0, j);
        }
    }
};

} // namespace utec::neural_network

#endif // NN_DENSE_H