//
// Created by karol on 5/07/2025.
//
#include "../include/agent/EnvGym.h"
#include "../include/agent/PongAgent.h"
#include "../include/nn/dense.h"
#include "../include/algebra/tensor.h"


 // Para crear un modelo de prueba
#include <iostream>
#include <memory>

// Para simplificar, usamos los namespaces
using namespace utec::nn;
using namespace utec::neural_network;
using namespace utec::algebra;

// Función para crear un modelo de prueba con pesos predecibles.
// Este modelo simple aprende a mover la paleta hacia la bola.
std::unique_ptr<Dense<float>> create_test_model() {
    // Un inicializador que establece pesos específicos.
    auto init_weights = [](Tensor<float, 2>& W) {
        // La lógica es:
        // Salida 0 (bajar): se activa si paddle_y > ball_y
        // Salida 1 (quieto): tiene una puntuación base de 0.
        // Salida 2 (subir): se activa si ball_y > paddle_y
        // Input: [ball_x, ball_y, paddle_y]
        W(0, 0) = 0; W(0, 1) = 0; W(0, 2) = 0; // No nos importa ball_x
        W(1, 0) = -10; W(1, 1) = 0; W(1, 2) = 10; // Coeficientes para ball_y
        W(2, 0) = 10; W(2, 1) = 0; W(2, 2) = -10; // Coeficientes para paddle_y
    };
    auto init_bias = [](Tensor<float, 2>& b) { b.fill(0); };

    return std::make_unique<Dense<float>>(3, 3, init_weights, init_bias);
}

int main(){
    std::cout << "--- Use Case #1: Basic Instantiation ---\n";
    auto agent = PongAgent<float>(create_test_model());
    // Estado: la bola está muy por encima de la paleta, se espera que suba (+1).
    State s_up{0.5f, 0.8f, 0.3f};
    int a_up = agent.act(s_up);
    std::cout << "Action for ball above paddle: " << a_up << " (Expected: 1)\n\n";

    std::cout << "--- Use Case #2: Single Step Simulation ---\n";
    EnvGym env;
    float reward;
    bool done;
    auto s0 = env.reset();
    int a0 = agent.act(s0);
    auto s1 = env.step(a0, reward, done);
    std::cout << "New state ball_x: " << s1.ball_x << ", Reward: " << reward
              << ", Done: " << done << "\n\n";

    std::cout << "--- Use Case #3: Multi-step Integration ---\n";
    s0 = env.reset();
    for (int t = 0; t < 5; ++t) {
        int a = agent.act(s0);
        s0 = env.step(a, reward, done);
        std::cout << "Step " << t << ", action=" << a << ", reward=" << reward
                  << "\n";
        if (done) {
            std::cout << "Game Over.\n";
            break;
        }
    }
    std::cout << "\n";

    std::cout << "--- Use Case #4: Boundary Test ---\n";
}



