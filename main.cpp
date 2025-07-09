#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <limits>
#include <fstream>

#include "include/agent/PongAgent.h"
#include "include/agent/EnvGym.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

using namespace utec::nn;

void limpiar_pantalla() {
#ifdef _WIN32
    system("cls");
#else
    system("clear");
#endif
}

void pausa(int ms = 1500) {
#ifdef _WIN32
    Sleep(ms);
#else
    usleep(ms * 1000);
#endif
}

void esperar_enter() {
    std::cin.ignore();
    std::cin.get();
}

void mostrar_menu() {
    std::cout <<
        "+==============================================+\n"
        "|                  PANEL PONG                 |\n"
        "+==============================================+\n"
        "| 1. Entrenar y cargar modelo desde CSV (IA)  |\n"
        "| 2. Ejecutar simulacion                      |\n"
        "| 3. Jugar manualmente y guardar datos        |\n"
        "| 4. Entrenar y cargar modelo con datos manual|\n"
        "| 5. Guardar modelo entrenado                 |\n"
        "| 6. Cargar modelo desde archivo              |\n"
        "| 7. Salir                                    |\n"
        "+==============================================+\n"
        "Seleccione una opcion: ";
}

void simular(PongAgent<float>& agente) {
    EnvGym env;
    auto estado = env.reset();
    float recompensa = 0, recompensa_total = 0;
    bool terminado = false;

    const int pasos_totales = 500;
    const int delay_ms = 20;

    for (int paso = 0; paso < pasos_totales; ++paso) {
        if (terminado) {
            std::cout << "\n🎮 ¡Perdiste la bola! Reiniciando...\n";
            estado = env.reset();
            terminado = false;
        }

        int accion = agente.act(estado);
        estado = env.step(accion, recompensa, terminado);
        recompensa_total += recompensa;

        const int ancho = 30, alto = 15; // 🔷 tabla más pequeña
        std::vector<std::string> pantalla(alto, std::string(ancho, ' '));

        // Bordes
        for (int i = 0; i < alto; ++i) {
            pantalla[i][0] = '|';
            pantalla[i][ancho - 1] = '|';
        }
        for (int j = 0; j < ancho; ++j) {
            pantalla[0][j] = '-';
            pantalla[alto - 1][j] = '-';
        }

        // Paleta
        int paddle_pos = static_cast<int>(estado.paddle_y * (alto - 3)) + 1;
        for (int i = -1; i <= 1; ++i) {
            int y = paddle_pos + i;
            if (y > 0 && y < alto - 1) pantalla[y][2] = '#';
        }

        // Bola
        int ball_x = static_cast<int>(estado.ball_x * (ancho - 4)) + 2;
        int ball_y = static_cast<int>(estado.ball_y * (alto - 2)) + 1;
        if (ball_x > 1 && ball_x < ancho - 1 && ball_y > 0 && ball_y < alto - 1)
            pantalla[ball_y][ball_x] = 'O';

        limpiar_pantalla();
        std::cout <<
            "+----------------------------------------+\n"
            "| Paso: " << paso <<
            " | Accion: " << (accion == -1 ? "ABAJO" : (accion == 1 ? "ARRIBA" : "QUIETO")) <<
            " | Recompensa: " << recompensa << "\n" <<
            "+----------------------------------------+\n";

        for (const auto& linea : pantalla)
            std::cout << "|" << linea << "|\n";
        std::cout << std::flush;
        pausa(50);  // <-- dar tiempo a la terminal

        std::cout << "+----------------------------------------+\n";
        std::cout << "Ball X: " << estado.ball_x
                  << " | Ball Y: " << estado.ball_y
                  << " | Paddle Y: " << estado.paddle_y << "\n";
        std::cout << "Recompensa total: " << recompensa_total << "\n";

        pausa(delay_ms);
    }

    std::cout <<
        "\n=== RESUMEN DE SIMULACION ===\n"
        "Pasos totales: " << pasos_totales << "\n"
        "Recompensa acumulada: " << recompensa_total << "\n"
        "Estado final: Bola (" << estado.ball_x << ", " << estado.ball_y
        << ") | Paleta: " << estado.paddle_y << "\n"
        "Presione ENTER para volver al menu...";
    esperar_enter();
}

void jugar_manual() {
    EnvGym env;
    auto estado = env.reset();
    bool terminado = false;
    float recompensa = 0;
    std::ofstream archivo("Data/pong_train_manual.csv", std::ios::app);

    if (!archivo) {
        std::cerr << "No se pudo abrir Data/pong_train_manual.csv para escribir\n";
        return;
    }

    const int ancho = 30, alto = 15; // 🔷 tabla más pequeña

    std::cout << "Jugando manualmente. Usa:\n";
    std::cout << "W = Subir | S = Bajar | D = Quieto | Q = Salir\n";

    while (true) {
        std::vector<std::string> pantalla(alto, std::string(ancho, ' '));

        for (int i = 0; i < alto; ++i) {
            pantalla[i][0] = '|';
            pantalla[i][ancho - 1] = '|';
        }
        for (int j = 0; j < ancho; ++j) {
            pantalla[0][j] = '-';
            pantalla[alto - 1][j] = '-';
        }

        int paddle_pos = static_cast<int>(estado.paddle_y * (alto - 3)) + 1;
        for (int i = -1; i <= 1; ++i) {
            int y = paddle_pos + i;
            if (y > 0 && y < alto - 1) pantalla[y][2] = '#';
        }

        int ball_x = static_cast<int>(estado.ball_x * (ancho - 4)) + 2;
        int ball_y = static_cast<int>(estado.ball_y * (alto - 2)) + 1;
        if (ball_x > 1 && ball_x < ancho - 1 && ball_y > 0 && ball_y < alto - 1)
            pantalla[ball_y][ball_x] = 'O';

        limpiar_pantalla();

        for (const auto &linea : pantalla)
            std::cout << linea << "\n";
        std::cout << std::flush;

        std::cout << "\nBall(" << estado.ball_x << ", " << estado.ball_y
                  << ") Paddle(" << estado.paddle_y << ") Recompensa: " << recompensa << "\n";
        std::cout << "Accion (W=Subir, S=Bajar, D=Quieto, Q=Salir): ";

        std::string entrada;
        std::cin >> entrada;

        int accion = 0;
        if (entrada == "W" || entrada == "w")
            accion = -1;
        else if (entrada == "S" || entrada == "s")
            accion = 1;
        else if (entrada == "D" || entrada == "d")
            accion = 0;
        else if (entrada == "Q" || entrada == "q")
            break;
        else {
            std::cout << "Comando invalido. Usa W, S, D o Q.\n";
            pausa(1000);
            continue;
        }

        estado = env.step(accion, recompensa, terminado);

        archivo << estado.ball_x << ","
                << estado.ball_y << ","
                << estado.paddle_y << ","
                << accion << ","
                << recompensa << "\n";

        if (terminado) {
            std::cout << "\n🎮 ¡Perdiste la bola! Reiniciando...\n";
            estado = env.reset();
            pausa(1000);
        }

        pausa(50); // 🔷 para dar tiempo a redibujar
    }

    archivo.close();
    std::cout << "\nDatos manuales guardados en Data/pong_train_manual.csv\n";
    pausa();
}

int main() {
    std::unique_ptr<PongAgent<float>> agente;
    bool modelo_cargado = false;
    int opcion = 0;
    bool salir = false;

    while (!salir) {
        limpiar_pantalla();
        mostrar_menu();
        std::cin >> opcion;

        if (std::cin.fail()) {
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            std::cout << "Opcion no valida.\n";
            pausa();
            continue;
        }

        limpiar_pantalla();

        switch (opcion) {
            case 1: {
                std::cout << "Entrenando el modelo desde CSV IA...\n";
                auto modelo = PongAgent<float>::train_from_csv("../Data/pong_train.csv",
                                                    2000, 0.001f);
                agente = std::make_unique<PongAgent<float>>(std::move(modelo));
                modelo_cargado = true;
                std::cout << "Entrenamiento completado y modelo cargado.\n";
                pausa();
                break;
            }
            case 2: {
                if (!modelo_cargado) {
                    std::cout << "Primero debe entrenar o cargar un modelo.\n";
                    pausa();
                    continue;
                }
                simular(*agente);
                break;
            }
            case 3: {
                jugar_manual();
                break;
            }
            case 4: {
                std::cout << "Entrenando el modelo con datos manuales...\n";
                auto modelo = PongAgent<float>::train_from_csv("../Data/pong_train_manual.csv",
                                                    2000, 0.001f);
                agente = std::make_unique<PongAgent<float>>(std::move(modelo));
                modelo_cargado = true;
                std::cout << "Entrenamiento con datos manuales completado.\n";
                pausa();
                break;
            }
            case 5: {
                if (!modelo_cargado || !agente) {
                    std::cout << "Primero debe entrenar o cargar un modelo antes de guardar.\n";
                } else {
                    auto* dense1 = agente->get_dense1();
                    auto* dense2 = agente->get_dense2();
                    if (dense1 && dense2) {
                        dense1->save_weights("../Data/pong_model_dense1.weights");
                        dense2->save_weights("../Data/pong_model_dense2.weights");
                        std::cout << "Modelo guardado.\n";
                    } else {
                        std::cout << "No se pudo acceder a las capas Dense para guardar.\n";
                    }
                }
                pausa();
                break;
            }
            case 6: {
                std::cout << "Cargando modelo desde archivos de pesos...\n";
                auto modelo = PongAgent<float>::create_sequential_with_weights(
                    "../Data/pong_model_dense1.weights",
                    "../Data/pong_model_dense2.weights"
                );
                agente = std::make_unique<PongAgent<float>>(std::move(modelo));
                modelo_cargado = true;
                std::cout << "Modelo cargado desde archivos de pesos.\n";
                pausa();
                break;
            }
            case 7: {
                salir = true;
                break;
            }
            default: {
                std::cout << "Opcion no valida.\n";
                pausa();
            }
        }
    }

    return 0;
}

/*
====================================================
🎮 Pong IA — Proyecto Final
====================================================

Este programa implementa una versión del clásico juego Pong,
con un agente de IA capaz de entrenarse y jugar automáticamente,
o con la posibilidad de jugar manualmente y generar datos para el entrenamiento.

El programa utiliza un menú interactivo en la consola
para ejecutar diferentes acciones relacionadas con el entrenamiento,
simulación y juego manual.

----------------------------------------------------
📋 Opciones del menú
----------------------------------------------------

Al ejecutar el programa (main.exe), se muestra un menú como este:

+==============================================+
|                  PANEL PONG                 |
+==============================================+
| 1. Entrenar y cargar modelo desde CSV (IA)  |
| 2. Ejecutar simulación                      |
| 3. Jugar manualmente y guardar datos        |
| 4. Entrenar y cargar modelo con datos manual|
| 5. Guardar modelo entrenado                 |
| 6. Cargar modelo desde archivo              |
| 7. Salir                                    |
+==============================================+

----------------------------------------------------
📄 Descripción de cada opción
----------------------------------------------------

✅ Opción 1: Entrenar y cargar modelo desde CSV (IA)
Entrena un modelo de IA leyendo datos desde el archivo
Data/pong_train.csv y lo carga en memoria.
Ideal para entrenar desde datos previamente recolectados automáticamente.

✅ Opción 2: Ejecutar simulación
Simula un juego de Pong controlado completamente por la IA entrenada.
Requiere haber cargado un modelo previamente (con la opción 1, 4 o 6).
Muestra la partida en la consola, incluyendo el estado de la bola y la paleta.

✅ Opción 3: Jugar manualmente y guardar datos
Permite al usuario jugar manualmente al Pong usando las teclas:
- W = Subir paleta
- S = Bajar paleta
- D = Mantener quieta
- Q = Salir

Mientras juegas, los datos de estado (posición de bola y paleta, acción y recompensa)
se guardan en el archivo Data/pong_train_manual.csv
para usarlos como datos de entrenamiento posteriormente.

✅ Opción 4: Entrenar y cargar modelo con datos manuales
Entrena la IA con los datos que el usuario haya generado manualmente en
Data/pong_train_manual.csv y carga el modelo resultante en memoria.

✅ Opción 5: Guardar modelo entrenado
Guarda el modelo entrenado actualmente en dos archivos de pesos:
- Data/pong_model_dense1.weights
- Data/pong_model_dense2.weights

✅ Opción 6: Cargar modelo desde archivo
Carga un modelo previamente guardado desde los archivos de pesos.

✅ Opción 7: Salir
Cierra el programa.

----------------------------------------------------
📁 Archivos importantes
----------------------------------------------------

- Data/pong_train.csv — Datos para entrenamiento automático.
- Data/pong_train_manual.csv — Datos generados por el usuario al jugar manualmente.
- Data/pong_model_dense1.weights y Data/pong_model_dense2.weights — Pesos del modelo entrenado.

----------------------------------------------------
🎯 Objetivo
----------------------------------------------------

Este programa está diseñado para:
- Recolectar datos de juego manual para entrenar a la IA.
- Entrenar una IA capaz de jugar Pong con aprendizaje supervisado.
- Visualizar el comportamiento del agente en una simulación.

====================================================
*/
