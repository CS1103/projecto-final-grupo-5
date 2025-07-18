#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include "interfaces.h"
#include "loss.h"
#include <memory>
#include <vector>
#include <iostream>
#include "optimizer.h"

namespace utec::neural_network {

/// @brief Clase principal para construir, entrenar y usar una red neuronal.
/// Permite añadir capas, ejecutar el paso forward, backpropagation y entrenamiento completo.
template <typename T>
class NeuralNetwork {
private:
    std::vector<std::unique_ptr<ILayer<T>>> layers_;  ///< Capas de la red
    bool verbose_ = false;                            ///< Imprimir progreso de entrenamiento

public:
    /// @brief Añade una nueva capa a la red
    /// @param layer Puntero a la capa a añadir
    void add_layer(std::unique_ptr<ILayer<T>> layer) {
        layers_.push_back(std::move(layer));
    }

    /// @brief Activa o desactiva la salida en consola durante el entrenamiento
    /// @param verbose Si es true, se imprime la pérdida cada 100 épocas
    void set_verbose(bool verbose) { verbose_ = verbose; }

    /// @brief Propagación hacia adelante de la red completa
    /// @param x Entrada inicial a la red
    /// @return Salida final después de pasar por todas las capas
    Tensor<T, 2> forward(const Tensor<T, 2>& x) {
        Tensor<T, 2> output = x;
        for (auto& layer : layers_) {
            output = layer->forward(output);
        }
        return output;
    }

    /// @brief Propagación hacia atrás de los gradientes
    /// @param grad Gradiente desde la función de pérdida
    void backward(const Tensor<T, 2>& grad) {
        Tensor<T, 2> current_grad = grad;
        for (auto it = layers_.rbegin(); it != layers_.rend(); ++it) {
            current_grad = (*it)->backward(current_grad);
        }
    }

    /// @brief Actualiza los parámetros de todas las capas usando un optimizador
    /// @param optimizer Optimizador que aplica la actualización
    void update_params(IOptimizer<T>& optimizer) {
        for (auto& layer : layers_) {
            layer->update_params(optimizer);
        }
    }

    /// @brief Entrena la red usando un dataset dado, función de pérdida y optimizador
    /// Usa mini-batch training y retropropagación. Muestra la pérdida si verbose está activo.
    /// @tparam LossType Tipo de función de pérdida (por ejemplo, MSELoss)
    /// @tparam OptimizerType Tipo de optimizador (por ejemplo, SGD o Adam)
    /// @param X Datos de entrada
    /// @param Y Etiquetas verdaderas
    /// @param epochs Número de épocas
    /// @param batch_size Tamaño de cada mini-batch
    /// @param learning_rate Tasa de aprendizaje
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

            // Imprime la pérdida si verbose está activo y es una época múltiplo de 100
            if (verbose_ && epoch % 100 == 0) {
                std::cout << "Epoch " << epoch << ", Loss: "
                          << total_loss / num_batches << std::endl;
            }
        }
    }

    /// @brief Realiza predicción (forward pass) sin modificar parámetros
    /// @param X Entrada a la red
    /// @return Salida producida por la red
    Tensor<T,2> predict(const Tensor<T,2>& X) {
        return forward(X);
    }
};

} // namespace utec::neural_network

#endif // NEURAL_NETWORK_H
