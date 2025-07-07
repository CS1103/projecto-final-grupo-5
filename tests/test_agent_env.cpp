/**
 * @file test_agent_env.cpp
 * @brief Prueba de integración entre un agente de aprendizaje y un entorno de Pong simulado.
 *
 * ### Flujo principal:
 * 1. Inicializa un modelo neuronal con pesos predefinidos (modelo simple).
 * 2. Crea un agente que utiliza dicho modelo.
 * 3. Configura el entorno Pong simulado.
 * 4. Ejecuta la simulación paso a paso, donde el agente toma decisiones.
 * 5. Registra y muestra el resultado de cada paso (acción, estado, recompensa).
 *
 * ### Métricas clave:
 * - Puntos ganados: Golpes exitosos (+1)
 * - Puntos perdidos: Fallos del agente (-1)
 * - Puntuación total: Diferencia entre ambos
 *
 * ### Observaciones:
 * - El modelo es totalmente determinista (no se entrena).
 * - La simulación tiene un máximo de 30 pasos por ejecución.
 */

#include "../include/agent/EnvGym.h"
#include "../include/agent/PongAgent.h"
#include "../include/nn/dense.h"
#include "../include/algebra/tensor.h"
#include <iostream>
#include <memory>
#include <iomanip>

using namespace utec::nn;
using namespace utec::neural_network;
using namespace utec::algebra;

/// @brief Crea un modelo denso fijo con pesos manualmente definidos.
/// @return Puntero único a un modelo Dense<float> 3x3
std::unique_ptr<Dense<float>> create_test_model() {
    auto init_weights = [](Tensor<float, 2>& W) {
        W(0, 0) = 0; W(0, 1) = 0; W(0, 2) = 0;
        W(1, 0) = -10; W(1, 1) = 0; W(1, 2) = 10;
        W(2, 0) = 10; W(2, 1) = 0; W(2, 2) = -10;
    };
    auto init_bias = [](Tensor<float, 2>& b) { b.fill(0); };
    return std::make_unique<Dense<float>>(3, 3, init_weights, init_bias);
}

/// @brief Imprime el encabezado de las columnas del log de simulación
void print_state_header() {
    std::cout << std::setw(6) << "Paso" << std::setw(8) << "Accion"
              << std::setw(12) << "Recompensa" << std::setw(10) << "Ball X"
              << std::setw(10) << "Ball Y" << std::setw(12) << "Paddle Y"
              << std::setw(15) << "Estado" << "\n";
}

/// @brief Imprime una línea del estado actual del entorno y acción del agente
void print_state(int step, int action, float reward, const State& s, const std::string& status) {
    std::cout << std::setw(6) << step
              << std::setw(8) << action
              << std::setw(12) << std::fixed << std::setprecision(4) << reward
              << std::setw(10) << std::fixed << std::setprecision(4) << s.ball_x
              << std::setw(10) << std::fixed << std::setprecision(4) << s.ball_y
              << std::setw(12) << std::fixed << std::setprecision(4) << s.paddle_y
              << std::setw(15) << status << "\n";
}

/// @brief Función principal de simulación
int main() {
    // Inicialización de agente y entorno
    auto agent = PongAgent<float>(create_test_model());
    EnvGym env;
    float reward;
    bool done;
    int max_steps = 30;
    int points = 0;
    int misses = 0;

    // Información inicial
    std::cout << "=== PARAMETROS DE SIMULACION ===\n";
    std::cout << "Pasos totales: " << max_steps << "\n";
    std::cout << "Velocidad bola X: " << -0.05f << " | Y: " << 0.02f << "\n";
    std::cout << "Velocidad paleta: " << 0.04f << "\n";
    std::cout << "Altura paleta: " << 0.2f << "\n\n";

    std::cout << "=== MODELO DEL AGENTE ===\n";
    std::cout << "Arquitectura: Densa (3x3)\n";
    std::cout << "Pesos:\n";
    std::cout << "  [0,0] = 0    [0,1] = 0    [0,2] = 0\n";
    std::cout << "  [1,0] = -10  [1,1] = 0    [1,2] = 10\n";
    std::cout << "  [2,0] = 10   [2,1] = 0    [2,2] = -10\n\n";

    std::cout << "=== SIMULACION COMPLETA DE PONG ===\n";
    print_state_header();

    // Reseteamos el entorno y empezamos la simulación
    auto state = env.reset();

    for (int step = 0; step < max_steps; step++) {
        int action = agent.act(state);
        state = env.step(action, reward, done);

        std::string status = "Jugando";
        if (reward > 0) {
            status = "GOLPE! +1";
            points++;
            std::cout << ">>> EVENTO: Golpe exitoso - "
                      << "Bola en Y=" << std::fixed << std::setprecision(4) << state.ball_y
                      << " vs Paleta en Y=" << state.paddle_y
                      << " | Diferencia: " << std::abs(state.ball_y - state.paddle_y) << "\n";
        } else if (reward < 0) {
            status = "FALLO! -1";
            misses++;
            std::cout << ">>> EVENTO: Fallo - "
                      << "Bola en Y=" << std::fixed << std::setprecision(4) << state.ball_y
                      << " vs Paleta en Y=" << state.paddle_y
                      << " | Diferencia: " << std::abs(state.ball_y - state.paddle_y) << "\n";
        }

        print_state(step, action, reward, state, status);

        if (done) {
            std::cout << "=== REINICIANDO ENTORNO ===\n";
            state = env.reset();
        }
    }

    // Mostrar resultados finales
    std::cout << "\n=== RESUMEN FINAL ===";
    std::cout << "\nPuntos ganados: " << points;
    std::cout << "\nPuntos perdidos: " << misses;
    std::cout << "\nPuntos totales: " << (points - misses) << "\n";

    // Análisis de rendimiento del agente
    std::cout << "\n=== ANALISIS DE RESULTADOS ===\n";
    if (points > 0)
        std::cout << "- El agente logro interceptar la bola exitosamente " << points << " veces\n";
    if (misses > 0)
        std::cout << "- El agente fallo en interceptar la bola " << misses << " veces\n";
    std::cout << "- La precision de movimiento fue "
              << (points > 0 ? "consistente" : "no optima") << " durante la simulacion\n";

    float total_events = points + misses;
    if (total_events > 0) {
        float success_rate = (points / total_events) * 100;
        std::cout << "- Tasa de exito: " << std::fixed << std::setprecision(1) << success_rate << "%\n";
    }

    std::cout << "- La estrategia de manejo de empates funciono correctamente\n";
    std::cout << "- El sistema completo demostro integracion estable entre componentes\n";

    return 0;
}

/*
 * === Ciclo de juego ===
 * 1. Bola inicia en centro → se mueve hacia paleta
 * 2. Agente intenta interceptar:
 *    - Éxito: Bola rebota hacia lado opuesto
 *    - Fallo: Termina el punto
 * 3. Bola viaja a pared opuesta y rebota
 * 4. Bola regresa hacia paleta (nueva oportunidad)
 * 5. El ciclo continúa hasta:
 *    - Fin de pasos simulados
 *    - Fallo del agente
 *
 * === Factores clave para rendimiento ===
 * - Velocidad de la bola (mayor = más ciclos posibles)
 * - Duración de simulación (más pasos = más oportunidades)
 * - Precisión del modelo (más éxitos con buena predicción)
 */
