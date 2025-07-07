#ifndef PONG_AGENT_H
#define PONG_AGENT_H

/**
 * @file PongAgent.h
 * @brief Define el agente que toma decisiones en el entorno Pong usando un modelo neuronal.
 *
 * El agente convierte el estado del juego en un tensor de entrada, ejecuta un forward pass
 * y toma la acción correspondiente al valor máximo predicho (con estrategia de desempate).
 */

#include "../nn/interfaces.h"
#include "EnvGym.h"
#include <algorithm>
#include <memory>

namespace utec::nn {

/// @brief Agente inteligente para jugar Pong usando una red neuronal o capa densa.
///
/// ### Funcionamiento:
/// 1. Recibe el estado del entorno (`State` con posición de bola y paleta)
/// 2. Convierte ese estado en un tensor de entrada (1x3)
/// 3. Ejecuta una predicción `forward()`
/// 4. Elige la acción con mayor valor Q:
///    - Índice 0 → acción -1 (bajar)
///    - Índice 1 → acción  0 (quieto)
///    - Índice 2 → acción +1 (subir)
/// 5. Si hay empate en las predicciones, elige siempre la acción neutral (quieto).
///
/// @tparam T Tipo de datos usados (float, double)
template <typename T>
class PongAgent {
private:
    std::unique_ptr<utec::neural_network::ILayer<T>> model_; ///< Modelo neuronal usado por el agente

public:
    /// @brief Constructor que recibe un modelo neuronal entrenado
    /// @param m Puntero único al modelo (puede ser una capa o red)
    PongAgent(std::unique_ptr<utec::neural_network::ILayer<T>> m)
        : model_(std::move(m)) {}

    /// @brief Toma una acción en base al estado actual del entorno.
    ///
    /// @param s Estado actual del juego (posición de bola y paleta)
    /// @return Acción seleccionada: -1 (bajar), 0 (quieto), +1 (subir)
    int act(const State& s) {
        // 1. Convertir el estado en tensor de entrada (1x3)
        utec::algebra::Tensor<T, 2> input(1, 3);
        input(0, 0) = s.ball_x;
        input(0, 1) = s.ball_y;
        input(0, 2) = s.paddle_y;

        // 2. Ejecutar predicción con el modelo
        utec::algebra::Tensor<T, 2> output = model_->forward(input);

        // 3. Determinar el índice con el mayor valor
        T max_val = output(0, 0);
        int max_idx = 0;
        bool tie = false;

        for (size_t j = 1; j < output.shape()[1]; ++j) {
            if (output(0, j) > max_val) {
                max_val = output(0, j);
                max_idx = j;
                tie = false;
            } else if (output(0, j) == max_val) {
                tie = true;
            }
        }

        // 4. Si hay empate, retornar acción neutral
        if (tie)
            return 0;

        // Mapear índice a acción: 0 → -1, 1 → 0, 2 → +1
        return max_idx - 1;
    }
};

} // namespace utec::nn

#endif // PONG_AGENT_H
