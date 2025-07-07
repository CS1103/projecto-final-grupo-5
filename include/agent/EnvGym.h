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
    float ball_vx_ = 0.03f;
    float ball_vy_ = 0.01f;
    const float PADDLE_SPEED = 0.04f;
    const float PADDLE_HEIGHT = 0.2f;

public:
    // Inicia o reinicia el juego a un estado inicial.
    State reset() {
        current_state_ = {0.5f, 0.5f, 0.5f};
        // Cambiado: Velocidad X negativa (hacia izquierda) y mayor magnitud
        ball_vx_ = -0.05f;
        ball_vy_ = 0.02f;
        return current_state_;
    }

    // Avanza un paso en la simulación.
    State step(int action, float& reward, bool& done) {
        done = false;
        reward = 0.0f;

        // 1. Mover la paleta según la acción.
        current_state_.paddle_y += action * PADDLE_SPEED;
        // Mantener la paleta dentro de los límites [0, 1]
        current_state_.paddle_y =
            std::max(0.0f, std::min(1.0f, current_state_.paddle_y));

        // 2. Mover la bola.
        current_state_.ball_x += ball_vx_;
        current_state_.ball_y += ball_vy_;

        // 3. Manejar colisiones con las paredes superior e inferior.
        if (current_state_.ball_y < 0.0f || current_state_.ball_y > 1.0f) {
            ball_vy_ = -ball_vy_;
        }

        // 4. Manejar colisión con la pared derecha (el oponente imaginario).
        if (current_state_.ball_x > 1.0f) {
            ball_vx_ = -ball_vx_;
        }

        // 5. Manejar colisión con la paleta del agente o fin del juego.
        if (current_state_.ball_x < 0.0f) {
            // ¿La bola golpeó la paleta?
            if (current_state_.ball_y > current_state_.paddle_y - PADDLE_HEIGHT / 2 &&
                current_state_.ball_y < current_state_.paddle_y + PADDLE_HEIGHT / 2) {
                ball_vx_ = -ball_vx_;
                reward = 1.0f; // Recompensa positiva por golpear la bola.
            } else {
                // La bola se fue, el juego termina.
                done = true;
                reward = -1.0f; // Recompensa negativa por fallar.
            }
        }

        return current_state_;
    }
};

} // namespace utec::nn

#endif // ENV_GYM_H