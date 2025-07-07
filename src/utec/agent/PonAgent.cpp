#include "../../../include/nn/dense.h"
#include "../../../include/nn/loss.h"
#include "../../../include/nn/optimizer.h"
#include "../../../include/agent/PongAgent.h" // Asegúrate de que aquí esté la definición de PongSample
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <iostream>

std::vector<utec::nn::PongSample> load_training_data(const std::string& filename) {
    std::vector<utec::nn::PongSample> data;
    std::ifstream file(filename);
    std::string line;

    if (!file.is_open()) {
        std::cerr << "Error: No se pudo abrir el archivo " << filename << std::endl;
        return data;
    }

    // Saltar encabezado
    std::getline(file, line);
    while (std::getline(file, line)) {
        try {
            std::stringstream ss(line);
            utec::nn::PongSample sample;
            char comma;
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
            data.push_back(sample);
        } catch (const std::exception& e) {
            std::cerr << "Error al procesar la línea: " << e.what() << std::endl;
        }
    }
    return data;
}
