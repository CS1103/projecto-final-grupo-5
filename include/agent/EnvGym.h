//
// Created by karol on 5/07/2025.
//
#ifndef ENV_GYM_H
#define ENV_GYM_H

/**
 * @file EnvGym.h
 * @brief Define un entorno de simulación simplificado del juego Pong para pruebas de IA.
 *
 * Este entorno permite probar agentes que toman decisiones con acciones discretas:
 * -1 (mover abajo), 0 (no mover), +1 (mover arriba).
 *
 * Se utiliza en conjunto con modelos de redes neuronales y agentes como PongAgent.
 */

#include <algorithm>  // Para std::min, std::max
#include <random>

namespace utec::nn {

/// @brief Representa el estado actual del juego Pong.
/// Incluye la posición de la bola (x, y) y la posición de la paleta controlada por el agente.
struct State {
    float ball_x;    ///< Posición X de la bola [0,1]
    float ball_y;    ///< Posición Y de la bola [0,1]
    float paddle_y;  ///< Posición Y del centro de la paleta [0,1]
};

/// @brief Entorno mínimo de simulación para el juego Pong.
///
/// Características principales:
/// - Movimiento simple de bola y paleta.
/// - Colisiones con paredes y paleta.
/// - Recompensas: +1 por golpear, -1 por fallar.
/// - Coordenadas normalizadas entre 0 y 1.
class EnvGym {
private:
    State current_state_;   ///< Estado actual del juego
    float ball_vx_ = 0.03f; ///< Velocidad X de la bola
    float ball_vy_ = 0.01f; ///< Velocidad Y de la bola
    const float PADDLE_SPEED = 0.04f;   ///< Velocidad de movimiento de la paleta
    const float PADDLE_HEIGHT = 0.2f;   ///< Altura total de la paleta

public:
    /// @brief Reinicia el entorno a un estado inicial predeterminado.
    ///
    /// Bola en el centro, paleta centrada, velocidad hacia la izquierda.
    /// @return Estado inicial del entorno
    State reset() {
        current_state_ = {0.5f, 0.5f, 0.5f};
        ball_vx_ = -0.05f;  // Hacia la izquierda
        ball_vy_ = 0.02f;   // Diagonal
        return current_state_;
    }

    /// @brief Avanza un paso en la simulación en base a la acción del agente.
    ///
    /// @param action Acción del agente: -1 (abajo), 0 (sin mover), 1 (arriba)
    /// @param reward Valor de recompensa (referencia de salida)
    /// @param done Indica si el juego terminó (referencia de salida)
    /// @return Nuevo estado del entorno luego de aplicar acción y física
    State step(int action, float& reward, bool& done) {
        done = false;
        reward = 0.0f;

        // 1. Mover paleta
        current_state_.paddle_y += action * PADDLE_SPEED;
        current_state_.paddle_y =
            std::max(0.0f, std::min(1.0f, current_state_.paddle_y));

        // 2. Mover bola
        current_state_.ball_x += ball_vx_;
        current_state_.ball_y += ball_vy_;

        // 3. Rebote con techo y piso
        if (current_state_.ball_y < 0.0f || current_state_.ball_y > 1.0f) {
            ball_vy_ = -ball_vy_;
        }

        // 4. Rebote con pared derecha (oponente)
        if (current_state_.ball_x > 1.0f) {
            ball_vx_ = -ball_vx_;
        }

        // 5. Chequear colisión con la paleta (izquierda)
        if (current_state_.ball_x < 0.0f) {
            if (current_state_.ball_y > current_state_.paddle_y - PADDLE_HEIGHT / 2 &&
                current_state_.ball_y < current_state_.paddle_y + PADDLE_HEIGHT / 2) {
                ball_vx_ = -ball_vx_;  // Rebote
                reward = 1.0f;         // Golpe exitoso
            } else {
                done = true;           // Fin del juego
                reward = -1.0f;        // Fallo
            }
        }

        return current_state_;
    }
};

} // namespace utec::nn

#endif // ENV_GYM_H
