[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/Lj3YlzJp)
# Proyecto Final 2025-1: AI Neural Network
## **CS2013 Programación III** · Informe Final

### **Descripción**

> Ejemplo: Implementación de una red neuronal multicapa en C++ para clasificación de dígitos manuscritos.

### Contenidos

1. [Datos generales](#datos-generales)
2. [Requisitos e instalación](#requisitos-e-instalación)
3. [Investigación teórica](#1-investigación-teórica)
4. [Diseño e implementación](#2-diseño-e-implementación)
5. [Ejecución](#3-ejecución)
6. [Análisis del rendimiento](#4-análisis-del-rendimiento)
7. [Trabajo en equipo](#5-trabajo-en-equipo)
8. [Conclusiones](#6-conclusiones)
9. [Bibliografía](#7-bibliografía)
10. [Licencia](#licencia)
---

### Datos generales

* **Tema**: Redes Neuronales en AI
* **Grupo**: `group_3_custom_name`
* **Integrantes**:

  * Alumno A – 209900001 (Responsable de investigación teórica)
  * Alumno B – 209900002 (Desarrollo de la arquitectura)
  * Alumno C – 209900003 (Implementación del modelo)
  * Alumno D – 209900004 (Pruebas y benchmarking)
  * Alumno E – 209900005 (Documentación y demo)

> *Nota: Reemplazar nombres y roles reales.*

---

### Requisitos e instalación

1. **Compilador**: GCC 11 o superior
2. **Dependencias**:

   * CMake 3.18+
   * Eigen 3.4
   * \[Otra librería opcional]
3. **Instalación**:

   ```bash
   git clone https://github.com/EJEMPLO/proyecto-final.git
   cd proyecto-final
   mkdir build && cd build
   cmake ..
   make
   ```>

> *Ejemplo de repositorio y comandos, ajustar según proyecto.*

---

### 1. Investigación teórica

* **Objetivo**: Explorar fundamentos y arquitecturas de redes neuronales.
* **Contenido de ejemplo**:

  1. Historia y evolución de las NNs.
  2. Principales arquitecturas: MLP, CNN, RNN.
  3. Algoritmos de entrenamiento: backpropagation, optimizadores.
     Fundamentos de redes neuronales
     Una red neuronal artificial (ANN) es un modelo inspirado en el cerebro humano que aprende a realizar tareas como clasificación o regresión mediante un proceso de entrenamiento. En este proyecto se implementa una red neuronal desde cero en C++, sin usar frameworks externos, lo que permite entender en detalle cada componente del modelo.

1.1 Estructura de una red neuronal
Una red neuronal está compuesta por:

Capas densas (fully connected): Cada neurona de una capa está conectada con todas las de la siguiente.

Funciones de activación: Como ReLU, permiten introducir no linealidad y mejorar la capacidad de representación.

Función de pérdida: Se usa MSE (Mean Squared Error) para medir el error entre predicciones y valores esperados.

Algoritmos de optimización: Se usaron SGD (descenso de gradiente estocástico) y Adam para ajustar los pesos y minimizar la pérdida.

1.2 Aprendizaje supervisado
El modelo se entrena con datos de entrada y salidas esperadas. Se realiza un ciclo:

forward: cálculo de predicción.

loss: comparación con el valor real.

backward: retropropagación del error para ajustar pesos.

optimize: aplicación del optimizador.

Este ciclo se repite durante muchas épocas para que la red aprenda.

1.3 Aplicación al juego Pong
En este proyecto, la red neuronal se entrena para que un agente llamado PongAgent decida cómo mover la paleta (arriba, abajo, quedarse quieto) en función del estado del juego (posición de la bola y la paleta).

La red aprende una política que maximiza la recompensa, usando una forma simplificada de aprendizaje por refuerzo.
---

## **2. Diseño e implementación**

### **2.1 Arquitectura de la solución**

El proyecto está construido con una arquitectura modular y escalable, separando responsabilidades entre álgebra tensorial, capas de red neuronal, funciones de pérdida, optimizadores, y el agente que interactúa con el entorno Pong.

#### **Patrones de diseño aplicados**

- **Strategy Pattern**: Implementado en el módulo de optimización. Las clases `SGD` y `Adam` heredan de una interfaz común `IOptimizer`, permitiendo intercambiar estrategias de actualización sin alterar la lógica de entrenamiento.
- **Template Method**: El método `train()` en la clase `NeuralNetwork` define el flujo fijo del entrenamiento (`forward → loss → backward → update`), pero permite variar las funciones de pérdida y optimización.
- **Factory Pattern (uso moderno)**: Las capas como `Dense` y `ReLU` son instanciadas dinámicamente usando `std::make_unique` y agregadas al modelo como punteros polimórficos, lo que permite encapsular fácilmente la creación de nuevas capas.

#### **Estructura del proyecto**

```
projecto-final-grupo-5/
├── cmake-build-debug/          # Carpeta de compilación
├── docs/                       # Documentación del proyecto
│   ├── BIBLIOGRAFIA.md
│   └── README.md
├── include/                    # Archivos de cabecera
│   ├── agent/
│   │   ├── EnvGym.h
│   │   └── PongAgent.h
│   ├── algebra/
│   │   └── tensor.h
│   └── nn/
│       ├── activation.h
│       ├── dense.h
│       ├── interfaces.h
│       ├── loss.h
│       ├── neural_network.h
│       └── optimizer.h
├── src/                        # Implementaciones fuente
│   └── utec/
│       └── agent/
│           ├── EnvGym.cpp
│           └── PongAgent.cpp
└── tests/                      # Casos de prueba
    └── test_agent_env.cpp
```

---

### **2.2 Manual de uso y casos de prueba**

#### **Cómo ejecutar**

Después de compilar el proyecto con CMake:

```bash
./build/neural_net_demo input.csv output.csv
```

Para probar el agente de Pong en el entorno simulado:

```bash
./build/pong_agent_demo
```

> Asegúrate de tener los archivos `input.csv` y `output.csv` listos, con datos en formato adecuado.

#### **Casos de prueba implementados**

- ✅ **Capa `Dense`**: Verifica propagación hacia adelante y retropropagación de gradientes.
- ✅ **Activación `ReLU`**: Asegura su correcto comportamiento en `forward` y `backward`.
- ✅ **Función de pérdida `MSELoss`**: Calcula la pérdida y deriva correctamente respecto a la predicción.
- ✅ **Entrenamiento de XOR**: La red converge con pérdida < 0.1 en 1000 épocas.
- ✅ **Agente `PongAgent`**: Dado un estado del juego, decide correctamente la acción (-1, 0, +1).
- ✅ **Entorno `EnvGym`**: Simulación funcional paso a paso, evaluando la interacción con el agente.

#### **Para correr los tests**

Si usas CMake:

```bash
cd build
ctest
```


### 3. Ejecución

> **Demo de ejemplo**: Video/demo alojado en `docs/demo.mp4`.
> Pasos:
>
> 1. Preparar datos de entrenamiento (formato CSV).
> 2. Ejecutar comando de entrenamiento.
> 3. Evaluar resultados con script de validación.

---

### 4. Análisis del rendimiento

* **Métricas de ejemplo**:

  * Iteraciones: 1000 épocas.
  * Tiempo total de entrenamiento: 2m30s.
  * Precisión final: 92.5%.
* **Ventajas/Desventajas**:

  * * Código ligero y dependencias mínimas.
  * – Sin paralelización, rendimiento limitado.
* **Mejoras futuras**:

  * Uso de BLAS para multiplicaciones (Justificación).
  * Paralelizar entrenamiento por lotes (Justificación).

---

### 5. Trabajo en equipo

| Tarea                     | Miembro  | Rol                       |
| ------------------------- | -------- | ------------------------- |
| Investigación teórica     | Alumno A | Documentar bases teóricas |
| Diseño de la arquitectura | Alumno B | UML y esquemas de clases  |
| Implementación del modelo | Alumno C | Código C++ de la NN       |
| Pruebas y benchmarking    | Alumno D | Generación de métricas    |
| Documentación y demo      | Alumno E | Tutorial y video demo     |

> *Actualizar con tareas y nombres reales.*

---

### 6. Conclusiones

* **Logros**: Implementar NN desde cero, validar en dataset de ejemplo.
* **Evaluación**: Calidad y rendimiento adecuados para propósito académico.
* **Aprendizajes**: Profundización en backpropagation y optimización.
* **Recomendaciones**: Escalar a datasets más grandes y optimizar memoria.

---

### 7. Bibliografía

> *Actualizar con bibliografia utilizada, al menos 4 referencias bibliograficas y usando formato IEEE de referencias bibliograficas.*

---

### Licencia

Este proyecto usa la licencia **MIT**. Ver [LICENSE](LICENSE) para detalles.

---
