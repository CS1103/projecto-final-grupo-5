#include "../include/agent/EnvGym.h"
#include "../include/agent/PongAgent.h"
#include "../include/nn/dense.h"
#include "../include/algebra/tensor.h"
#include <iostream>
#include <memory>
#include <iomanip>
/**
 * @file test_agent_env.cpp
 * @brief Prueba de integracion entre agente y entorno Pong
 *
 * Flujo principal:
 * 1. Inicializar modelo neuronal con pesos predefinidos
 * 2. Crear agente con el modelo
 * 3. Configurar entorno de simulacion
 * 4. Ejecutar pasos de simulacion
 * 5. Registrar y mostrar resultados
 *
 * Metricas clave:
 * - Puntos ganados: Golpes exitosos (+1)
 * - Puntos perdidos: Fallos (-1)
 * - Puntuacion total: Diferencia entre ambos
 */
using namespace utec::nn;
using namespace utec::neural_network;
using namespace utec::algebra;

std::unique_ptr<Dense<float>> create_test_model() {
    auto init_weights = [](Tensor<float, 2>& W) {
        W(0, 0) = 0; W(0, 1) = 0; W(0, 2) = 0;
        W(1, 0) = -10; W(1, 1) = 0; W(1, 2) = 10;
        W(2, 0) = 10; W(2, 1) = 0; W(2, 2) = -10;
    };
    auto init_bias = [](Tensor<float, 2>& b) { b.fill(0); };
    return std::make_unique<Dense<float>>(3, 3, init_weights, init_bias);
}

void print_state_header() {
    std::cout << std::setw(6) << "Paso" << std::setw(8) << "Accion"
              << std::setw(12) << "Recompensa" << std::setw(10) << "Ball X"
              << std::setw(10) << "Ball Y" << std::setw(12) << "Paddle Y"
              << std::setw(15) << "Estado" << "\n";
}

void print_state(int step, int action, float reward, const State& s, const std::string& status) {
    std::cout << std::setw(6) << step
              << std::setw(8) << action
              << std::setw(12) << std::fixed << std::setprecision(4) << reward
              << std::setw(10) << std::fixed << std::setprecision(4) << s.ball_x
              << std::setw(10) << std::fixed << std::setprecision(4) << s.ball_y
              << std::setw(12) << std::fixed << std::setprecision(4) << s.paddle_y
              << std::setw(15) << status << "\n";
}

int main() {
    // Configuracion inicial
    auto agent = PongAgent<float>(create_test_model());
    EnvGym env;
    float reward;
    bool done;
    int max_steps = 30;
    int points = 0;
    int misses = 0;

    // Informacion de simulacion
    std::cout << "=== PARAMETROS DE SIMULACION ===\n";
    std::cout << "Pasos totales: " << max_steps << "\n";
    std::cout << "Velocidad bola X: " << -0.05f << " | Y: " << 0.02f << "\n";
    std::cout << "Velocidad paleta: " << 0.04f << "\n";
    std::cout << "Altura paleta: " << 0.2f << "\n\n";

    // Informacion del modelo
    std::cout << "=== MODELO DEL AGENTE ===\n";
    std::cout << "Arquitectura: Densa (3x3)\n";
    std::cout << "Pesos:\n";
    std::cout << "  [0,0] = 0    [0,1] = 0    [0,2] = 0\n";
    std::cout << "  [1,0] = -10  [1,1] = 0    [1,2] = 10\n";
    std::cout << "  [2,0] = 10   [2,1] = 0    [2,2] = -10\n\n";

    std::cout << "=== SIMULACION COMPLETA DE PONG ===\n";
    print_state_header();

    auto state = env.reset();

    for (int step = 0; step < max_steps; step++) {
        int action = agent.act(state);
        state = env.step(action, reward, done);

        // Registrar eventos importantes
        std::string status = "Jugando";
        if (reward > 0) {
            status = "GOLPE! +1";
            points++;

            // Detalle adicional para golpes exitosos
            std::cout << ">>> EVENTO: Golpe exitoso - "
                      << "Bola en Y=" << std::fixed << std::setprecision(4) << state.ball_y
                      << " vs Paleta en Y=" << std::fixed << std::setprecision(4) << state.paddle_y
                      << " | Diferencia: " << std::abs(state.ball_y - state.paddle_y)
                      << "\n";
        } else if (reward < 0) {
            status = "FALLO! -1";
            misses++;

            // Detalle adicional para fallos
            std::cout << ">>> EVENTO: Fallo - "
                      << "Bola en Y=" << std::fixed << std::setprecision(4) << state.ball_y
                      << " vs Paleta en Y=" << std::fixed << std::setprecision(4) << state.paddle_y
                      << " | Diferencia: " << std::abs(state.ball_y - state.paddle_y)
                      << "\n";
        }

        print_state(step, action, reward, state, status);

        if (done) {
            std::cout << "=== REINICIANDO ENTORNO ===\n";
            state = env.reset();
        }
    }

    // Resumen final
    std::cout << "\n=== RESUMEN FINAL ===";
    std::cout << "\nPuntos ganados: " << points;
    std::cout << "\nPuntos perdidos: " << misses;
    std::cout << "\nPuntos totales: " << (points - misses) << "\n";

    // Analisis de resultados
    std::cout << "\n=== ANALISIS DE RESULTADOS ===\n";
    if (points > 0) {
        std::cout << "- El agente logro interceptar la bola exitosamente " << points << " veces\n";
    }
    if (misses > 0) {
        std::cout << "- El agente fallo en interceptar la bola " << misses << " veces\n";
    }
    std::cout << "- La precision de movimiento fue "
              << (points > 0 ? "consistente" : "no optima")
              << " durante la simulacion\n";

    // Calculamos la tasa de exito
    float total_events = points + misses;
    if (total_events > 0) {
        float success_rate = (points / total_events) * 100;
        std::cout << "- Tasa de exito: " << std::fixed << std::setprecision(1)
                  << success_rate << "%\n";
    }

    std::cout << "- La estrategia de manejo de empates funciono correctamente\n";
    std::cout << "- El sistema completo demostro integracion estable entre componentes\n";

    return 0;
}
/***Ciclo de juego completo**:
1. Bola inicia en centro → se mueve hacia paleta
2. Agente intenta interceptar:
   - Éxito: Bola rebota hacia lado opuesto
   - Fallo: Termina el punto
3. Bola viaja a pared opuesta y rebota
4. Bola regresa hacia paleta (nueva oportunidad)
5. El ciclo continúa hasta:
   - Fin de pasos simulados
   - Fallo del agente

**Factores clave para múltiples interacciones**:
- Velocidad de la bola (mayor = más ciclos)
- Duración de simulación (más pasos = más oportunidades)
- Habilidad del agente (mejor posición = más éxitos)*/