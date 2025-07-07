
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
        std::unique_ptr<utec::neural_network::ILayer<T>> model_;

    public:
        PongAgent(std::unique_ptr<utec::neural_network::ILayer<T>> m)
            : model_(std::move(m)) {}

        int act(const State& s) {
            utec::algebra::Tensor<T, 2> input(1, 3);
            input(0, 0) = s.ball_x;
            input(0, 1) = s.ball_y;
            input(0, 2) = s.paddle_y;

            utec::algebra::Tensor<T, 2> output = model_->forward(input);

            T max_val = output(0, 0);
            int max_idx = 0;
            int count_max = 1;

            for (size_t j = 1; j < output.shape()[1]; ++j) {
                T val = output(0, j);
                if (val > max_val) {
                    max_val = val;
                    max_idx = j;
                    count_max = 1;
                }
                else if (val == max_val) {
                    count_max++;
                }
            }

            // Manejar empates inteligentemente
            if (count_max > 1) {
                float diff = s.ball_y - s.paddle_y;
                if (std::abs(diff) < 0.1f) return 0;
                return (diff > 0) ? 1 : -1;
            }

            return max_idx - 1;
        }
    };

} // namespace utec::nn

#endif // PONG_AGENT_H