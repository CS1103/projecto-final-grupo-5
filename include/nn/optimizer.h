#ifndef NN_OPTIMIZER_H
#define NN_OPTIMIZER_H

#include "interfaces.h"
#include <cmath>

namespace utec::neural_network {

/// @brief Optimizador Stochastic Gradient Descent (SGD).
///
/// Realiza una actualización directa de los parámetros utilizando una tasa de aprendizaje fija.
/// Es simple y ampliamente usado para entrenamiento de modelos.
template <typename T>
class SGD final : public IOptimizer<T> {
private:
    T learning_rate_; ///< Tasa de aprendizaje

public:
    /// @brief Constructor con tasa de aprendizaje opcional
    /// @param learning_rate Valor que controla la magnitud de las actualizaciones
    explicit SGD(T learning_rate = 0.01) : learning_rate_(learning_rate) {}

    /// @brief Aplica una actualización de los parámetros con descenso de gradiente
    /// @param params Parámetros actuales del modelo
    /// @param grads Gradientes calculados respecto a los parámetros
    void update(Tensor<T, 2>& params, const Tensor<T, 2>& grads) override {
        for (size_t i = 0; i < params.shape()[0]; ++i) {
            for (size_t j = 0; j < params.shape()[1]; ++j) {
                params(i, j) -= learning_rate_ * grads(i, j);
            }
        }
    }
};


/// @brief Optimizador Adam (Adaptive Moment Estimation).
/// Combina ventajas de AdaGrad y RMSProp. Utiliza promedios móviles de primer y segundo orden
/// para adaptar la tasa de aprendizaje por parámetro.
template <typename T>
class Adam final : public IOptimizer<T> {
private:
    T learning_rate_; ///< Tasa de aprendizaje
    T beta1_;         ///< Coeficiente para el promedio móvil de primer orden (momento)
    T beta2_;         ///< Coeficiente para el promedio móvil de segundo orden (aceleración)
    T epsilon_;       ///< Pequeño valor para evitar división por cero
    std::size_t t_;   ///< Contador de iteraciones
    Tensor<T,2> m_;   ///< Promedio móvil de primer orden (momentum)
    Tensor<T,2> v_;   ///< Promedio móvil de segundo orden (varianza)

public:
    /// @brief Constructor con hiperparámetros configurables
    Adam(T learning_rate = 0.001, T beta1 = 0.9, T beta2 = 0.999, T epsilon = 1e-8)
        : learning_rate_(learning_rate), beta1_(beta1), beta2_(beta2),
          epsilon_(epsilon), t_(0) {}

    /// @brief Aplica una actualización de parámetros con el algoritmo Adam
    /// @param params Parámetros actuales del modelo
    /// @param grads Gradientes calculados
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

                // Cálculo de momentos
                m_(i, j) = beta1_ * m_(i, j) + (1 - beta1_) * g;
                v_(i, j) = beta2_ * v_(i, j) + (1 - beta2_) * g * g;

                // Corrección de sesgo
                T m_hat = m_(i, j) / (1 - std::pow(beta1_, t_));
                T v_hat = v_(i, j) / (1 - std::pow(beta2_, t_));

                // Actualización de parámetros
                params(i, j) -= learning_rate_ * m_hat / (std::sqrt(v_hat) + epsilon_);
            }
        }
    }

    /// @brief Método adicional opcional para optimizadores con estado (no usado aquí)
    void step() override {}
};

} // namespace utec::neural_network

#endif // NN_OPTIMIZER_H
