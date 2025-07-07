#ifndef NN_INTERFACES_H
#define NN_INTERFACES_H

#include "../algebra/tensor.h"

namespace utec::neural_network {

/// @brief Interfaz para optimizadores (SGD, Adam, etc.)
/// Define cómo se actualizan los parámetros durante el entrenamiento.
template <typename T>
class IOptimizer;


/// @brief Interfaz para capas de una red neuronal (e.g. Dense, ReLU).
/// Toda capa debe implementar `forward`, `backward` y `update_params`.
template <typename T>
class ILayer {
public:
    virtual ~ILayer() = default;

    /// @brief Propagación hacia adelante
    /// @param input Tensor de entrada
    /// @return Tensor transformado por la capa
    virtual Tensor<T, 2> forward(const Tensor<T, 2>& input) = 0;

    /// @brief Retropropagación del error (gradientes)
    /// @param grad Gradiente desde la capa superior
    /// @return Gradiente respecto a la entrada
    virtual Tensor<T, 2> backward(const Tensor<T, 2>& grad) = 0;

    /// @brief Actualiza los parámetros de la capa usando un optimizador
    /// @param optimizer Optimizer que aplica la actualización
    virtual void update_params(IOptimizer<T>& optimizer) = 0;
};


/// @brief Interfaz para funciones de pérdida (loss functions).
/// Define la forma de calcular el error y su gradiente.
template <typename T, std::size_t Rank>
class ILoss {
public:
    virtual ~ILoss() = default;

    /// @brief Calcula el valor actual de la pérdida
    /// @return Escalar representando el error
    virtual T loss() const = 0;

    /// @brief Calcula el gradiente de la pérdida respecto a la salida
    /// @return Tensor del mismo rango que la salida
    virtual Tensor<T, Rank> loss_gradient() const = 0;
};


/// @brief Interfaz para optimizadores (e.g. SGD, Adam)
/// Define cómo aplicar el gradiente a los parámetros.
template <typename T>
class IOptimizer {
public:
    virtual ~IOptimizer() = default;

    /// @brief Aplica actualización a los parámetros usando los gradientes
    /// @param params Parámetros a actualizar
    /// @param grads Gradientes calculados
    virtual void update(Tensor<T, 2>& params, const Tensor<T, 2>& grads) = 0;

    /// @brief Metodo opcional para optimizadores con estado (ej. Adam)
    virtual void step() {}
};

} // namespace utec::neural_network

#endif // NN_INTERFACES_H
