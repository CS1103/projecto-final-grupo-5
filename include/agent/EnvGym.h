#pragma once
#ifndef ENV_GYM_H
#define ENV_GYM_H

#include <algorithm>
#include <cmath>
#include <cstdlib>  // para rand()
#include <iostream>

namespace utec::nn {

/// @brief Estado del entorno Pong.
/// Contiene la posición de la bola (x, y) y la posición vertical de la paleta.
    struct State {
        float ball_x, ball_y;
        float paddle_y;
        float ball_vx;      ///< Velocidad de la bola en X
        float ball_vy;
    };

/// @brief Entorno de simulación tipo Pong.
/// Permite probar agentes en un entorno simple de paddle controlado por IA.
class EnvGym {
private:
    State current_state_;          ///< Estado actual del juego
    float ball_vx_ = 0.06f;        ///< Velocidad de la bola en X
    float ball_vy_ = 0.03f;        ///< Velocidad de la bola en Y
    const float PADDLE_SPEED = 0.05f;   ///< Velocidad de la paleta
    const float PADDLE_HEIGHT = 0.2f;   ///< Altura de la paleta

public:
    /// @brief Constructor. Llama automáticamente a reset().
    EnvGym() {
        reset();
    }

    /// @brief Reinicia el estado del entorno.
    /// La bola se ubica en el centro con una velocidad aleatoria.
    /// @return Estado inicial del juego
    State reset() {
        current_state_ = {0.5f, 0.5f, 0.5f, ball_vx_, ball_vy_};

        // Velocidad aleatoria y rápida (comentada alternativa fija)
        ball_vx_ = (0.04f + 0.06f * (rand() % 100 / 100.0f)) * (rand() % 2 ? 1 : -1);
        ball_vy_ = (0.02f + 0.04f * (rand() % 100 / 100.0f)) * (rand() % 2 ? 1 : -1);
        current_state_.ball_vx = ball_vx_;
        current_state_.ball_vy = ball_vy_;
        return current_state_;
    }

    /// @brief Ejecuta un paso en la simulación.
    /// @param action Acción del agente: -1 (bajar), 0 (quedarse), 1 (subir)
    /// @param reward Recompensa obtenida
    /// @param done Indica si el juego terminó
    /// @return Nuevo estado después del paso
    State step(int action, float& reward, bool& done) {
        done = false;
        reward = 0.0f;

        // Penalización leve por moverse (para promover estabilidad)
        if (action != 0) reward -= 0.01f;

        // Movimiento de la paleta (corrigiendo la dirección)
        current_state_.paddle_y += (-action) * PADDLE_SPEED;

        // Limitar la paleta a los bordes del entorno
        current_state_.paddle_y = std::clamp(
            current_state_.paddle_y,
            PADDLE_HEIGHT / 2,
            1.0f - PADDLE_HEIGHT / 2
        );

        // Movimiento de la bola
        current_state_.ball_x += ball_vx_;
        current_state_.ball_y += ball_vy_;

        // Rebote contra el borde superior
        if (current_state_.ball_y <= 0.0f) {
            current_state_.ball_y = 0.0f;
            ball_vy_ = -ball_vy_;
        }

        // Rebote contra el borde inferior
        if (current_state_.ball_y >= 1.0f) {
            current_state_.ball_y = 1.0f;
            ball_vy_ = -ball_vy_;
        }

        // Rebote contra la pared derecha (enemigo imaginario)
        if (current_state_.ball_x >= 1.0f) {
            current_state_.ball_x = 1.0f;
            ball_vx_ = -ball_vx_;
        }

        // Colisión con la paleta (o fallo)
        if (current_state_.ball_x <= 0.0f) {
            float paddle_top = current_state_.paddle_y + PADDLE_HEIGHT / 2;
            float paddle_bottom = current_state_.paddle_y - PADDLE_HEIGHT / 2;

            if (current_state_.ball_y >= paddle_bottom && current_state_.ball_y <= paddle_top) {
                // Rebote exitoso
                current_state_.ball_x = 0.0f;
                ball_vx_ = std::abs(ball_vx_);
                reward += 1.0f;

                // Recompensa adicional por interceptar cerca del centro
                float center_dist = std::abs(current_state_.ball_y - current_state_.paddle_y);
                if (center_dist < 0.05f) {
                    reward += 1.0f;
                }
            } else {
                // Fallo: termina el episodio
                done = true;
                reward -= 5.0f;
            }
        }

        return current_state_;
    }
};

} // namespace utec::nn

#endif // ENV_GYM_H
