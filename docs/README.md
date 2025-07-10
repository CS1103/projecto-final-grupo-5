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

* **Tema**: Desarrollar un agente de IA para jugar Pong usando C++
* **Grupo**: `Grupo 5`
* **Integrantes**:

  * Guevara Vargas Eduardo S. – 202410096 (Responsable de investigación teórica y documentación técnica)
  * Cayllahua Hilario Joel M. - 202410731 (Desarrollo de la arquitectura y estructura base del código)
  * Tamayo Hilario Maria K.   - 202410766 (Implementación del modelo neuronal)
  * García López Bruno W.     - 202410719 (Pruebas, benchmarking y ejecución del demo)
  * Rosales Bazán Diana S.    - 202410535 (Documentación, asistencia en la demo y estructura del proyecto)

---

### Requisitos e instalación

 **1. Requisitos del sistema**

- Compilador compatible con **C++20** (ej. GCC 11+ o Clang 13+)
- **CMake 3.18+** para gestionar la compilación
- Sistema operativo: Linux, Windows o macOS

 **2. Dependencias**

Este proyecto es **auto-contenido**: no requiere librerías externas adicionales como TensorFlow.

Sin embargo, organiza su código en carpetas modulares, por lo que requiere:

- Soporte para `std::array`, `std::vector`, `std::unique_ptr`
- Acceso al compilador desde consola (para compilar con CMake)
- Una versión moderna de `g++` o `clang++` con soporte a **templates avanzados**

> **Nota**: No se utilizan bibliotecas de álgebra externas; `Tensor<T, Rank>` está implementado desde cero.

### **3. Clonar y compilar - Instalación**

```bash
# Clonar el repositorio
git clone https://github.com/CS1103/projecto-final-grupo-5.git
cd projecto-final-grupo-5

# Crea carpeta de build
mkdir build && cd build

# Genera los archivos de compilación
cmake ..

# Compila el proyecto
make
```

---
### 1. Investigación teórica

* **Objetivo**: Explorar fundamentos y arquitecturas de redes neuronales.
### **1.1 Historia y evolución de las NNs**
Las redes neuronales artificiales tienen sus raíces en los estudios del cerebro humano. El primer modelo matemático fue propuesto por McCulloch y Pitts en 1943, quienes desarrollaron el primer modelo de neurona artificial binaria.
  * 1943: Primera neurona artificial por McCulloch-Pitts.
  * 1949: Regla de aprendizaje hebbiano por Hebb.
  * 1958: Perceptrón, primer algoritmo de aprendizaje por Rosenblatt.

### **1.2 Principales arquitecturas: MLP, CNN, RNN**
#### Multi-Layer Perceptron (MLP):
El MLP es la arquitectura más básica de feedforward neuronal rojo, compuesta por múltiples capas de neuronas completamente conectadas.
* Cada neurona está conectada a todas las neuronas de la capa siguiente, genera capas densas.
* Función de activación, introducir no linealidad (ReLU, Sigmoid, Tanh).
* Propagación hacia adelante, los datos fluyen en una sola dirección.
#### Convolutional Neural Networks (CNN):
Las CNN están diseñadas específicamente para procesar datos con estructura de parrilla, como imágenes.
* Capa Convolucional: Aplicación de filtros (kernels) a la entrada, detecta características locales.
* **Operación:(f * g)(t) = ∑ f(τ)g(t-τ)**
* Capa de Pooling: Reduce la dimensionalidad y proporciona invariancia a traducciones.
* **Tipos: Agrupación máxima, Agrupación promedio.**
* Capa Completamente Conectada: Generalmente al final para la clasificación. Es similar a MLP.
#### Recurrent Neural Networks (RNN):
Las RNN pueden procesar secuencias de datos de longitud variable, manteniendo un estado interno que actúa como memoria.
* LSTM (Long Short-Term Memory):
  1. Resuelva el problema del gradiente que desaparece.
  2. Tres puertas: forget, input, output.
  3. Memoria de largo plazo más efectiva.
* GRU (Gated Recurrent Unit):
  1. Versión simplificada de LSTM.
  2. Dos puertas: reset, update.
  3. Menor complejidad computacional.

### **1.3 Algoritmos de entrenamiento: backpropagation, optimizadores**
#### Backpropagation:
El algoritmo de backpropagation es el método estándar para entrenar redes neuronales mediante el cálculo eficiente de gradientes.
* Forward Pass: Calcular la salida de la red.
* Calcular Error: Comparar con el valor objetivo.
* Backward Pass: Propagar el error hacia atrás.
* Actualizar Pesos: Usando el gradiente calculado.
#### Optimizadores:
* **Gradient Descent (GD):** Versión más básica, usa todo el dataset en cada actualización y convergencia lenta pero estable.
* **Stochastic Gradient Descent (SGD):** Usa una muestra por actualización, puede escapar de mínimos locales y es más rápido pero con mayor varianza.
* **Adam (Adaptive Moment Estimation):** Es adaptativo por parámetro, combina momentum y RMSprop (Adaptativo para learning rate).
#### Técnicas de Regularización:
* **Dropout:** Elimina aleatoriamente neuronas durante entrenamiento, reduce overfitting y fuerza a la red a no depender de neuronas específicas.
* **Batch Normalization:** Normaliza las activaciones de cada capa, acelera el entrenamiento y reduce la sensibilidad a la inicialización.

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
proyecto-final-grupo-5/
├── cmake-build-debug/          # Carpeta de compilación
├── Data/
│   ├── pong_train.csv            # Datos para entrenamiento automático
│   ├── pong_train_manual.csv     # Datos generados manualmente
│   ├── pong_model_dense1.weights # Pesos de la capa densa 1
│   └── pong_model_dense2.weights # Pesos de la capa densa 2
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
│       ├── agent/
│       │   ├── EnvGym.cpp
│       │   └── PongAgent.cpp
│       └── DataGenerator.cpp   # Generador de datos sintéticos para entrenamiento
└── tests/                      # Casos de prueba
│  └── test_agent_env.cpp
└── main.cpp                 # Menú interactivo principal 
```

---

### **2.2 Manual de uso y casos de prueba**

#### **Cómo ejecutar**

Luego de compilar el proyecto con CMake, se puede ejecutar el menú principal del simulador de Pong desde consola:

```bash
./build/main
```


Este menú permite realizar las siguientes acciones:

| Opción | Acción |
|--------|--------|
| 1 | Entrenar y cargar un modelo IA desde `Data/pong_train.csv`. |
| 2 | Ejecutar una simulación automática donde el agente juega por sí solo. |
| 3 | Jugar manualmente con teclado (`W`, `S`, `D`) y guardar los datos en `pong_train_manual.csv`. |
| 4 | Entrenar la IA usando los datos generados manualmente. |
| 5 | Guardar los pesos del modelo entrenado. |
| 6 | Cargar un modelo previamente guardado desde archivos `.weights`. |
| 7 | Salir del programa. |

> Antes de ejecutar simulaciones automáticas (opción 2), es necesario haber entrenado o cargado un modelo (opciones 1, 4 o 6).

---

### Ejecución alternativa: Generador de datos sintéticos

Para generar un conjunto base de entrenamiento desde código (sin juego real):

```bash
./build/DataGenerator
```

Esto crea un archivo `Data/pong_train.csv` con datos sintéticos para entrenamiento supervisado.

---

### Controles del juego manual

Durante la ejecución de la opción 3, el usuario puede controlar la paleta usando el teclado:

- `W` → Subir paleta
- `S` → Bajar paleta
- `D` → Mantener posición
- `Q` → Salir del modo manual

Cada acción es registrada en `Data/pong_train_manual.csv` junto con el estado y recompensa, útil para personalizar el estilo de juego del agente.

---

### Casos de prueba implementados

Los archivos dentro del directorio `tests/` contienen validaciones unitarias de los componentes fundamentales del sistema:

- **Dense Layer**: Validación de `forward()` y `backward()`, incluyendo gradientes y pesos.
- **ReLU Activation**: Comprobación del comportamiento esperado ante entradas positivas y negativas.
- **MSELoss**: Confirmación del cálculo correcto de la pérdida y sus derivadas.
- **Red XOR**: Entrenamiento de una red simple para aprender la compuerta lógica XOR. Se considera éxito si la pérdida cae por debajo de 0.1 en menos de 1000 épocas.
- **PongAgent**: Verificación de la función `act()` dada una entrada, asegurando que retorna valores dentro del rango {-1, 0, 1}.
- **EnvGym**: Comprobación de la simulación del entorno, asegurando que responde coherentemente a las acciones del agente.

---

### Test destacado: integración entre agente y entorno

Un caso clave de prueba consiste en simular una partida completa de Pong en 30 pasos, utilizando un modelo con pesos definidos manualmente.

Se monitorean:

- Las acciones ejecutadas por el agente.
- La posición de la bola y la paleta.
- Las recompensas obtenidas por cada interacción.
- Los reinicios automáticos tras fallar.

Además, se imprime un resumen final con métricas como:

- Puntos totales obtenidos.
- Frecuencia de éxito (contacto con la bola).
- Evaluación general del comportamiento de la IA.

Este test permite validar que la interacción entre `PongAgent` y `EnvGym` se mantiene coherente a lo largo de múltiples ciclos de simulación.

---

### Cómo correr los tests

Usando CMake y CTest, los casos pueden ejecutarse desde la carpeta `build`:

```bash
cd build
ctest
```

O también ejecutando directamente el binario de pruebas si existe:

```bash
./tests/test_agent_env
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

### **Métricas del agente en entorno Pong**

Durante la simulación de 30 pasos usando el modelo denso 3x3 con pesos definidos, el agente obtuvo los siguientes resultados:

- **Puntos ganados (GOLPE):** variable según simulación
- **Puntos perdidos (FALLO):** variable según simulación
- **Tasa de éxito:** alrededor de **60% - 75%** (dependiendo de posición inicial)
- **Tiempo de ejecución:** menos de 1 segundo en CPU

> *Nota: El rendimiento varía según los parámetros del entorno (velocidad de bola, duración, etc.).*

---

### **Ventajas observadas**

- **Arquitectura modular y clara**, separando entorno, agente y red neuronal.
- **Uso de templates y punteros inteligentes** (`std::unique_ptr`), que facilitan escalabilidad.
- **Código liviano**, sin dependencias externas complejas.
- **Buena precisión del agente** en simulaciones simples.
- **Sistema de pruebas robusto**, validando individualmente y en conjunto cada módulo.

---

### **Limitaciones actuales**

- No se implementa un sistema de aprendizaje real (refuerzo o backprop) en el agente durante la simulación.
- Solo se usa una red densa básica, sin capas ocultas profundas.
- No hay paralelismo ni GPU (CUDA), lo cual limitaría la escalabilidad en simulaciones masivas.
- El entrenamiento depende de datos externos generados o simulados, no del entorno en tiempo real.

---

### **Posibles mejoras futuras**

- Implementar un bucle de entrenamiento con **aprendizaje por refuerzo** (Q-learning, SARSA, DQN).
- Agregar **entrenamiento por lotes y replay buffer** para aprendizaje off-policy.
- Crear entornos más complejos y parametrizables (velocidad, gravedad...).
- Explorar redes más profundas y funciones de activación alternativas.
- Incorporar paralelismo con `std::thread` o CUDA para acelerar simulaciones.
- Diseñar una interfaz gráfica para visualización de episodios y métricas.

---


### 5. Trabajo en equipo y asignación de tareas

> Todos los miembros colaboraron activamente en el desarrollo y revisión del código.  
> Los roles mostrados reflejan las principales responsabilidades asumidas por cada integrante.

| Tarea                     | Miembro                   | Rol principal                        |
|---------------------------|---------------------------|--------------------------------------|
| Investigación teórica     | Guevara Vargas Eduardo S. | Comentarios in-code, README, documentación técnica |
| Diseño de la arquitectura | Cayllahua Hilario Joel M. | Estructura de capas y clases base    |
| Implementación del modelo | Tamayo Hilario Maria K.   | Capas, funciones de activación, forward/backward |
| Pruebas y ejecución final | García López Bruno W.     | Validación de comportamiento y demo final |
| Documentación y soporte   | Rosales Bazán Diana S.    | Bases teóricas, documentación, apoyo a presentación |

---

### 6. Conclusiones

* **Logros**: Implementar NN desde cero, validar en dataset de ejemplo.
              Implementación de interfaces ILayer, activaciones, funciones de pérdida y optimización.
* **Evaluación**: Calidad y rendimiento adecuados para propósito académico.
                  Simulación completa sin errores o comportamientos erráticos y golpe exitoso con diferencia de 0.0000 entre bola y paleta.
* **Aprendizajes**: Profundización en backpropagation y optimización.
                    Dominio de templates C++20 y POO.
* **Recomendaciones**: Escalar a datasets más grandes, optimizar memoria, añadir obstáculos, múltiples pelotas o física más realista y uso de algoritmos más avanzados.
* **Impacto**: Framework extensible para futuros proyectos de machine learning.
---

### 7. Bibliografía

W. S. McCulloch and W. Pitts, "A logical calculus of the ideas immanent in nervous activity," Bulletin of Mathematical Biophysics, vol. 5, no. 4, pp. 115-133, 1943.
F. Rosenblatt, "The perceptron: a probabilistic model for information storage and organization in the brain," Psychological Review, vol. 65, no. 6, pp. 386-408, 1958.
D. E. Rumelhart, G. E. Hinton, and R. J. Williams, "Learning representations by back-propagating errors," Nature, vol. 323, no. 6088, pp. 533-536, 1986.
Y. LeCun, Y. Bengio, and G. Hinton, "Deep learning," Nature, vol. 521, no. 7553, pp. 436-444, 2015.
I. Goodfellow, Y. Bengio, and A. Courville, Deep Learning. Cambridge, MA: MIT Press, 2016.
S. Hochreiter and J. Schmidhuber, "Long short-term memory," Neural Computation, vol. 9, no. 8, pp. 1735-1780, 1997.
A. Vaswani et al., "Attention is all you need," in Advances in Neural Information Processing Systems, 2017, pp. 5998-6008.
D. P. Kingma and J. Ba, "Adam: A method for stochastic optimization," arXiv preprint arXiv:1412.6980, 2014.




---

### Licencia

Este proyecto usa la licencia **MIT**. Ver [LICENSE](LICENSE) para detalles.

---
