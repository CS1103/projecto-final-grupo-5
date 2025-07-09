/// @file DataGenerator.cpp
/// @brief Genera datos sintéticos de entrenamiento para el agente de Pong.
///
/// Este programa escribe un archivo CSV llamado `pong_train.csv` en el directorio `Data/`.
/// Cada fila representa una muestra del entorno de juego con sus respectivas características,
/// acción ideal y recompensa asociada.
///
/// Se generan combinaciones de posiciones de bola, velocidades y posiciones de paleta,
/// con acciones correctas (acción ideal según posición) y acciones incorrectas (para entrenar el castigo).

#include <iostream>
#include <fstream>

int main() {
    std::ofstream file("Data/pong_train.csv");

    if (!file.is_open()) {
        std::cerr << "Error al abrir el archivo." << std::endl;
        return 1;
    }

    file << "ball_x,ball_y,ball_vx,ball_vy,paddle_y,action,reward\n";

    for (float bx = 0.0f; bx <= 1.0f; bx += 0.1f) {
        for (float by = 0.0f; by <= 1.0f; by += 0.1f) {
            for (float vx : {-0.02f, 0.02f}) {
                for (float vy : {-0.01f, 0.01f}) {
                    for (float py = 0.0f; py <= 1.0f; py += 0.1f) {

                        int action = 0;
                        if (by > py + 0.05f) action = -1;       // SUBIR
                        else if (by < py - 0.05f) action = 1;   // BAJAR
                        else action = 0;                        // QUIETO

                        int reward = 1;

                        file << bx << "," << by << "," << vx << "," << vy << "," << py << "," << action << "," << reward << "\n";

                        int wrong_action = (action == 1) ? -1 : (action == -1 ? 1 : 1);
                        int wrong_reward = -1;

                        file << bx << "," << by << "," << vx << "," << vy << "," << py << "," << wrong_action << "," << wrong_reward << "\n";
                    }
                }
            }
        }
    }

    file.close();
    std::cout << "Datos generados en Data/pong_train.csv\n";
    return 0;
}

