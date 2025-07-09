#include "../../../include/nn/dense.h"
#include "../../../include/nn/loss.h"
#include "../../../include/nn/optimizer.h"
#include "../../../include/agent/PongAgent.h"

#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <iostream>

/// @file PonAgent.cpp
/// @brief Implementación de utilidad para cargar datos de entrenamiento para el agente Pong.
/// Este archivo contiene una función que permite cargar datos desde un archivo CSV
/// y convertirlos en estructuras `PongSample`, listas para entrenar un modelo de red neuronal.

/// @brief Carga los datos de entrenamiento desde un archivo CSV.
/// Cada fila del archivo debe contener los siguientes valores en orden:
/// ball_x, ball_y, ball_vx, ball_vy, paddle_y, action, reward
///
/// @param filename Ruta del archivo CSV.
/// @return Vector de muestras de entrenamiento (PongSample).
std::vector<utec::nn::PongSample> load_training_data(const std::string& filename) {
    std::vector<utec::nn::PongSample> data;
    std::ifstream file(filename);
    std::string line;

    // Verifica que el archivo se haya abierto correctamente
    if (!file.is_open()) {
        std::cerr << "Error: No se pudo abrir el archivo " << filename << std::endl;
        return data;
    }

    // Saltar la primera línea (encabezado CSV)
    std::getline(file, line);

    // Leer línea por línea
    while (std::getline(file, line)) {
        try {
            std::stringstream ss(line);
            utec::nn::PongSample sample;
            char comma;

            // Parseo de valores separados por comas
            ss >> sample.ball_x >> comma
               >> sample.ball_y >> comma
               >> sample.ball_vx >> comma
               >> sample.ball_vy >> comma
               >> sample.paddle_y >> comma
               >> sample.action >> comma
               >> sample.reward;

            if (ss.fail()) {
                throw std::runtime_error("Formato incorrecto en la línea: " + line);
            }

            // Agregar muestra a la colección
            data.push_back(sample);

        } catch (const std::exception& e) {
            std::cerr << "Error al procesar la línea: " << e.what() << std::endl;
        }
    }

    return data;
}
