//
// Created by karol on 5/07/2025.
//

#ifndef PONG_AGENT_H
#define PONG_AGENT_H

#include "../nn/interfaces.h"
#include "EnvGym.h"
#include <algorithm> // Para std::max_element
#include <memory>

namespace utec::nn {

    template <typename T>
    class PongAgent {
    private:
        // El modelo es una capa o una red neuronal completa.
        std::unique_ptr<utec::neural_network::ILayer<T>> model_;

    public:
        // El constructor toma posesión del modelo entrenado.
        PongAgent(std::unique_ptr<utec::neural_network::ILayer<T>> m)
            : model_(std::move(m)) {}

        // Convierte el estado del juego en una acción (-1, 0, o 1).
        int act(const State& s) {
            // 1. Convertir el estado (struct) a un Tensor de entrada (1x3).
            // Usamos el Tensor de tu proyecto.
            utec::algebra::Tensor<T, 2> input(1, 3);
            input(0, 0) = s.ball_x;
            input(0, 1) = s.ball_y;
            input(0, 2) = s.paddle_y;

            // 2. Obtener la predicción de la red (propagación hacia adelante).
            utec::algebra::Tensor<T, 2> output = model_->forward(input);

            // 3. Encontrar la acción con el valor más alto.
            // La salida es un tensor (1x3) con las puntuaciones para [abajo, quieto,
            // arriba].
            T max_val = output(0, 0);
            int max_idx = 0;
            for (size_t j = 1; j < output.shape()[1]; ++j) {
                if (output(0, j) > max_val) {
                    max_val = output(0, j);
                    max_idx = j;
                }
            }

            // 4. Mapear el índice de la acción al valor de acción requerido.
            // Índice 0 -> Acción -1 (bajar)
            // Índice 1 -> Acción  0 (quieto)
            // Índice 2 -> Acción +1 (subir)
            return max_idx - 1;
        }
    };

} // namespace utec::nn

#endif // PONG_AGENT_H
