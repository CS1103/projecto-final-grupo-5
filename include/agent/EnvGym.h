#pragma once
#ifndef ENV_GYM_H
#define ENV_GYM_H

#include <algorithm>
#include <cmath>
#include <cstdlib>  // para rand()
#include <iostream>

namespace utec::nn {

struct State {
    float ball_x, ball_y;
    float paddle_y;
};

class EnvGym {
private:
    State current_state_;
    float ball_vx_ = 0.06f;   // más rápido que antes
    float ball_vy_ = 0.03f;   // más rápido que antes
    const float PADDLE_SPEED = 0.05f;
    const float PADDLE_HEIGHT = 0.2f;

public:
    EnvGym() {
        reset();
    }

    State reset() {
        current_state_ = {0.5f, 0.5f, 0.5f};

        // velocidad fija más rápida:
        ball_vx_ = 0.06f;
        ball_vy_ = 0.03f;

        // O si quieres que sea aleatoria y rápida, descomenta esto:

        ball_vx_ = (0.04f + 0.06f * (rand() % 100 / 100.0f)) * (rand() % 2 ? 1 : -1);
        ball_vy_ = (0.02f + 0.04f * (rand() % 100 / 100.0f)) * (rand() % 2 ? 1 : -1);


        return current_state_;
    }

    State step(int action, float &reward, bool &done) {
        done = false;
        reward = 0.0f;

        if (action != 0) reward -= 0.01f;

        // Corregido: 1 = subir, -1 = bajar
        current_state_.paddle_y += action * PADDLE_SPEED;

        // Limitar paddle dentro de los límites
        current_state_.paddle_y = std::clamp(
            current_state_.paddle_y,
            PADDLE_HEIGHT / 2,
            1.0f - PADDLE_HEIGHT / 2
        );

        // Mover bola
        current_state_.ball_x += ball_vx_;
        current_state_.ball_y += ball_vy_;

        // Rebote en paredes superior/inferior
        if (current_state_.ball_y <= 0.0f) {
            current_state_.ball_y = 0.0f;
            ball_vy_ = -ball_vy_;
        }
        if (current_state_.ball_y >= 1.0f) {
            current_state_.ball_y = 1.0f;
            ball_vy_ = -ball_vy_;
        }

        // Rebote en la pared derecha
        if (current_state_.ball_x >= 1.0f) {
            current_state_.ball_x = 1.0f;
            ball_vx_ = -ball_vx_;
        }

        // Colisión con la paleta o pérdida
        if (current_state_.ball_x <= 0.0f) {
            float paddle_top = current_state_.paddle_y + PADDLE_HEIGHT / 2;
            float paddle_bottom = current_state_.paddle_y - PADDLE_HEIGHT / 2;

            if (current_state_.ball_y >= paddle_bottom && current_state_.ball_y <= paddle_top) {
                // rebote
                current_state_.ball_x = 0.0f; // aseguramos que no salga
                ball_vx_ = std::abs(ball_vx_); // siempre hacia la derecha
                reward += 1.0f;

                float center_dist = std::abs(current_state_.ball_y - current_state_.paddle_y);
                if (center_dist < 0.05f) {
                    reward += 1.0f;
                }
            } else {
                done = true;
                reward -= 5.0f;
            }
        }

        return current_state_;
    }

};

} // namespace utec::nn

#endif // ENV_GYM_H
