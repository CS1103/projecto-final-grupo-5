//
// Created by karol on 5/07/2025.
//
#ifndef ENV_GYM_H
#define ENV_GYM_H
/**
 * @class EnvGym
 * @brief Entorno de simulación del juego Pong simplificado
 *
 * Características principales:
 * - Estado: Posición de la bola (x,y) y posición de la paleta (y)
 * - Acciones: -1 (mover abajo), 0 (no mover), +1 (mover arriba)
 * - Física básica: Rebotes en paredes y paleta
 * - Sistema de recompensas: +1 por golpear bola, -1 por fallar
 *
 * @note La simulación usa coordenadas normalizadas [0,1] en ambos ejes
 */
#include <algorithm> // Para std::min, std::max
#include <random>

namespace utec::nn {

// Define el estado del juego.
struct State {
    float ball_x, ball_y;
    float paddle_y;
};

// Entorno mínimo de simulación.
class EnvGym {
private:
    State current_state_;
    float ball_vx_;
    float ball_vy_;
    const float PADDLE_SPEED = 0.04f;
    const float PADDLE_HEIGHT = 0.2f;
    const float BALL_RADIUS = 0.02f;

public:
    // Inicia o reinicia el juego a un estado inicial.
    State reset() {
        // Posición inicial aleatoria
        current_state_ = {0.5f, static_cast<float>(rand()) / RAND_MAX, 0.5f};

        // Velocidad inicial aleatoria
        ball_vx_ = -0.03f - (static_cast<float>(rand()) / RAND_MAX * 0.02f);
        ball_vy_ = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 0.04f;

        return current_state_;
    }

    State step(int action, float& reward, bool& done) {
        done = false;
        reward = 0.0f;

        // Mover paleta
        current_state_.paddle_y += action * PADDLE_SPEED;
        current_state_.paddle_y =
            std::max(0.0f, std::min(1.0f, current_state_.paddle_y));

        // Mover bola
        current_state_.ball_x += ball_vx_;
        current_state_.ball_y += ball_vy_;

        // Rebotar en paredes superior/inferior
        if (current_state_.ball_y < 0.0f || current_state_.ball_y > 1.0f) {
            ball_vy_ = -ball_vy_;
            current_state_.ball_y =
                std::max(0.0f, std::min(1.0f, current_state_.ball_y));
        }

        // Rebotar en pared derecha
        if (current_state_.ball_x > 1.0f) {
            ball_vx_ = -ball_vx_;
            current_state_.ball_x = 1.0f - (current_state_.ball_x - 1.0f);
        }

        // Verificar colisión con paleta
        const bool in_paddle_x = current_state_.ball_x < 0.05f;
        const bool in_paddle_y =
            (current_state_.ball_y > current_state_.paddle_y - PADDLE_HEIGHT/2) &&
            (current_state_.ball_y < current_state_.paddle_y + PADDLE_HEIGHT/2);

        if (in_paddle_x && in_paddle_y) {
            ball_vx_ = -ball_vx_ * 1.1f;  // Aumentar velocidad
            reward = 1.0f;
        }
        else if (current_state_.ball_x < 0.0f) {
            done = true;
            reward = -1.0f;
        }

        return current_state_;
    }
};

} // namespace utec::nn

#endif // ENV_GYM_H