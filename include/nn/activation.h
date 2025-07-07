#ifndef NN_ACTIVATION_H
#define NN_ACTIVATION_H

#include <valarray>

#include "interfaces.h"

namespace utec::neural_network {
/// @brief Función de activación ReLU (Rectified Linear Unit).
///
/// Define f(x) = max(0, x). Deja pasar valores positivos y anula los negativos.
/// Esta clase implementa `forward()` y `backward()` para su uso en redes neuronales.
template <typename T>
class ReLU final : public ILayer<T> {
public:
    /// @brief Aplica la función ReLU elemento a elemento.
    /// @param input Tensor de entrada
    /// @return Tensor activado
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
    /// @brief Derivada de ReLU para retropropagación.
    /// Multiplica el gradiente solo donde la entrada original fue positiva.
    /// @param grad Gradiente de salida
    /// @return Gradiente de entrada
    Tensor<T, 2> backward(const Tensor<T, 2>& grad) override {
        Tensor<T, 2> output(grad.shape()[0], grad.shape()[1]);
        for (size_t i = 0; i < grad.shape()[0]; ++i) {
            for (size_t j = 0; j < grad.shape()[1]; ++j) {
                output(i, j) = grad(i, j) * (mask_(i, j) > 0 ? 1 : 0);
            }
        }
        return output;
    }
    /// @brief ReLU no tiene parámetros entrenables.
    void update_params(IOptimizer<T>& optimizer) override {}

private:
    Tensor<T, 2> mask_;///< Guarda la entrada original para usar en el backward
};
/// @brief Función de activación Sigmoid.
///
/// Define f(x) = 1 / (1 + e^(-x)). Comprime el rango a (0,1).
/// Usada comúnmente en tareas de clasificación binaria.
template <typename T>
class Sigmoid final : public ILayer<T> {
public:
    /// @brief Aplica la función Sigmoid elemento a elemento.
    /// @param input Tensor de entrada
    /// @return Tensor activado
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
    /// @brief Derivada de Sigmoid para retropropagación.
    /// Usa la fórmula s(x) * (1 - s(x)) con s = sigmoid(x)
    /// @param grad Gradiente de salida
    /// @return Gradiente de entrada
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
    /// @brief Sigmoid no tiene parámetros entrenables.
    void update_params(IOptimizer<T>& optimizer) override {}

private:
    Tensor<T, 2> output_; ///< Guarda la salida para calcular la derivada
};

} // namespace utec::neural_network

#endif // NN_ACTIVATION_H