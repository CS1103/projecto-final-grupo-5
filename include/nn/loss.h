#ifndef NN_LOSS_H
#define NN_LOSS_H

#include "interfaces.h"
#include <cmath>

namespace utec::neural_network {

/// @brief Función de pérdida de error cuadrático medio (MSE).
/// Se utiliza comúnmente para tareas de regresión. Calcula la media del cuadrado
/// de las diferencias entre las predicciones y los valores verdaderos.
template <typename T>
class MSELoss : public ILoss<T, 2> {
private:
    algebra::Tensor<T, 2> y_pred_;  ///< Predicciones del modelo
    algebra::Tensor<T, 2> y_true_;  ///< Valores verdaderos

public:
    /// @brief Constructor que recibe las predicciones y etiquetas verdaderas
    MSELoss(const algebra::Tensor<T,2>& y_prediction, const algebra::Tensor<T,2>& y_true)
        : y_pred_(y_prediction), y_true_(y_true) {}

    /// @brief Calcula el valor de la pérdida MSE
    /// @return Escalar con el error cuadrático medio
    T loss() const override {
        T total_loss = 0;
        size_t elements = y_pred_.shape()[0] * y_pred_.shape()[1];
        for (size_t i = 0; i < y_pred_.shape()[0]; ++i) {
            for (size_t j = 0; j < y_pred_.shape()[1]; ++j) {
                T diff = y_pred_(i, j) - y_true_(i, j);
                total_loss += diff * diff;
            }
        }
        return total_loss / elements;
    }

    /// @brief Calcula el gradiente de la pérdida MSE con respecto a la predicción
    /// @return Tensor con el gradiente
    algebra::Tensor<T,2> loss_gradient() const override {
        algebra::Tensor<T,2> grad(y_pred_.shape()[0], y_pred_.shape()[1]);
        size_t elements = y_pred_.shape()[0] * y_pred_.shape()[1];
        for (size_t i = 0; i < y_pred_.shape()[0]; ++i) {
            for (size_t j = 0; j < y_pred_.shape()[1]; ++j) {
                grad(i, j) = 2 * (y_pred_(i, j) - y_true_(i, j)) / elements;
            }
        }
        return grad;
    }
};


/// @brief Función de pérdida binaria de entropía cruzada (Binary Cross Entropy).
///
/// Utilizada para clasificación binaria. Evalúa la diferencia entre las probabilidades
/// predichas y las verdaderas etiquetas binarias.
template <typename T>
class BCELoss : public ILoss<T, 2> {
private:
    algebra::Tensor<T, 2> y_pred_;  ///< Predicciones del modelo
    algebra::Tensor<T, 2> y_true_;  ///< Etiquetas verdaderas
    T epsilon = 1e-12;              ///< Término de seguridad para evitar log(0)

public:
    /// @brief Constructor que recibe las predicciones y etiquetas verdaderas
    BCELoss(const algebra::Tensor<T,2>& y_prediction, const algebra::Tensor<T,2>& y_true)
        : y_pred_(y_prediction), y_true_(y_true) {}

    /// @brief Calcula el valor de la pérdida BCE
    /// @return Escalar con la pérdida binaria
    T loss() const override {
        T total_loss = 0;
        size_t elements = y_pred_.shape()[0] * y_pred_.shape()[1];
        for (size_t i = 0; i < y_pred_.shape()[0]; ++i) {
            for (size_t j = 0; j < y_pred_.shape()[1]; ++j) {
                T y_p = std::max(epsilon, std::min(1 - epsilon, y_pred_(i, j)));
                T y_t = y_true_(i, j);
                total_loss += - (y_t * std::log(y_p) + (1 - y_t) * std::log(1 - y_p));
            }
        }
        return total_loss / elements;
    }

    /// @brief Calcula el gradiente de la pérdida BCE con respecto a la predicción
    /// @return Tensor con el gradiente
    algebra::Tensor<T,2> loss_gradient() const override {
        algebra::Tensor<T,2> grad(y_pred_.shape()[0], y_pred_.shape()[1]);
        size_t elements = y_pred_.shape()[0] * y_pred_.shape()[1];
        for (size_t i = 0; i < y_pred_.shape()[0]; ++i) {
            for (size_t j = 0; j < y_pred_.shape()[1]; ++j) {
                T y_p = std::max(epsilon, std::min(1 - epsilon, y_pred_(i, j)));
                T y_t = y_true_(i, j);
                grad(i, j) = (y_p - y_t) / (y_p * (1 - y_p) * elements);
            }
        }
        return grad;
    }
};

} // namespace utec::neural_network

#endif // NN_LOSS_H
