
#ifndef PONG_AGENT_H
#define PONG_AGENT_H
/**
 * @class PongAgent
 * @brief Agente inteligente para jugar Pong usando un modelo neuronal
 *
 * Funcionamiento:
 * 1. Recibe el estado del juego (ball_x, ball_y, paddle_y)
 * 2. Convierte el estado a tensor de entrada (1x3)
 * 3. Realiza predicción con el modelo neuronal
 * 4. Selecciona acción con mayor valor Q
 * 5. Maneja empates seleccionando acción neutral (0)
 *
 * @tparam T Tipo de datos para cálculos (float/double)
 */
#include "../nn/interfaces.h"
#include "EnvGym.h"
#include <algorithm>
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
            // La salida es un tensor (1x3) con las puntuaciones para [abajo, quieto, arriba].
            T max_val = output(0, 0);
            int max_idx = 0;
            bool tie = false;  // Bandera para detectar empates

            // Buscar el valor máximo y verificar empates
            for (size_t j = 1; j < output.shape()[1]; ++j) {
                if (output(0, j) > max_val) {
                    max_val = output(0, j);
                    max_idx = j;
                    tie = false;  // Nuevo máximo encontrado, reinicia bandera de empate
                } else if (output(0, j) == max_val) {
                    tie = true;   // Se encontró un empate
                }
            }

            // 4. Mapear el índice de la acción al valor de acción requerido.
            // Si hay empate, seleccionar la acción "quieto" (índice 1)
            if (tie) {
                return 0;  // Acción quieto
            }

            // Índice 0 -> Acción -1 (bajar)
            // Índice 1 -> Acción  0 (quieto)
            // Índice 2 -> Acción +1 (subir)
            return max_idx - 1;
        }
    };

} // namespace utec::nn

#endif // PONG_AGENT_H