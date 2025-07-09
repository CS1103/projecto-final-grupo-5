#ifndef DENSE_H
#define DENSE_H

#include "interfaces.h"
#include <functional>
#include <type_traits>
#include <fstream>
#include <string>

namespace utec::neural_network {

/// @brief Capa densa (fully connected) para una red neuronal.
///
/// Realiza una transformación lineal: y = xW + b.
/// Implementa los métodos de forward, backward, y actualización de pesos.
template <typename T>
class Dense final : public ILayer<T> {
private:
    Tensor<T, 2> W_, dW_;   ///< Pesos y gradientes de los pesos
    Tensor<T, 1> b_, db_;   ///< Bias y gradientes del bias
    Tensor<T, 2> last_x_;   ///< Entrada del forward, usada para el backward

public:
    using Initializer = std::function<void(Tensor<T, 2>&)>;
    using InitializerBias = std::function<void(Tensor<T, 1>&)>;

    /// @brief Constructor con funciones de inicialización personalizadas.
    /// @param in_f Cantidad de entradas (features)
    /// @param out_f Cantidad de salidas (neuronas)
    /// @param init_w_fun Función para inicializar W
    /// @param init_b_fun Función para inicializar b
    template <typename InitW, typename InitB>
    Dense(size_t in_f, size_t out_f, InitW&& init_w_fun, InitB&& init_b_fun) {
        W_ = Tensor<T, 2>(in_f, out_f);
        dW_ = Tensor<T, 2>(in_f, out_f);
        init_w_fun(W_);

        b_ = Tensor<T, 1>(out_f);
        db_ = Tensor<T, 1>(out_f);

        // Adaptar bias 1D a 2D para usar misma función de inicialización
        Tensor<T, 2> b_view(1, out_f);
        init_b_fun(b_view);
        for (size_t j = 0; j < out_f; ++j) {
            b_(j) = b_view(0, j);
        }
    }

    /// @brief Constructor único cuando se pasa una sola función de inicialización.
    template <typename Init,
              typename = std::enable_if_t<
                  std::is_invocable_r_v<void, Init, Tensor<T, 2>&>
              >>
    Dense(size_t in_f, size_t out_f, Init&& init_fun)
        : Dense(in_f, out_f,
                [&](auto& tensor) { init_fun(tensor); },
                [&](auto& tensor) {
                    Tensor<T, 2> temp(1, tensor.size());
                    init_fun(temp);
                    for (size_t j = 0; j < tensor.size(); ++j) {
                        tensor[j] = temp(0, j);
                    }
                }) {}

    /// @brief Propagación hacia adelante (y = xW + b)
    Tensor<T, 2> forward(const Tensor<T, 2>& x) override {
        last_x_ = x;
        Tensor<T, 2> output(x.shape()[0], W_.shape()[1]);
        for (size_t i = 0; i < x.shape()[0]; ++i) {
            for (size_t j = 0; j < W_.shape()[1]; ++j) {
                T sum = 0;
                for (size_t k = 0; k < W_.shape()[0]; ++k) {
                    sum += x(i, k) * W_(k, j);
                }
                output(i, j) = sum + b_(j);
            }
        }
        return output;
    }

    /// @brief Retropropagación de gradientes (cálculo de dW y db)
    Tensor<T, 2> backward(const Tensor<T, 2>& dZ) override {
        dW_.fill(0);
        for (size_t i = 0; i < last_x_.shape()[0]; ++i) {
            for (size_t k = 0; k < last_x_.shape()[1]; ++k) {
                for (size_t j = 0; j < dZ.shape()[1]; ++j) {
                    dW_(k, j) += last_x_(i, k) * dZ(i, j);
                }
            }
        }

        db_.fill(0);
        for (size_t i = 0; i < dZ.shape()[0]; ++i) {
            for (size_t j = 0; j < dZ.shape()[1]; ++j) {
                db_(j) += dZ(i, j);
            }
        }

        // Calcular gradiente respecto a la entrada (dX)
        algebra::Tensor<T, 2> dX(last_x_.shape()[0], last_x_.shape()[1]);
        for (size_t i = 0; i < dX.shape()[0]; ++i) {
            for (size_t k = 0; k < dX.shape()[1]; ++k) {
                T sum = 0;
                for (size_t j = 0; j < W_.shape()[1]; ++j) {
                    sum += dZ(i, j) * W_(k, j);
                }
                dX(i, k) = sum;
            }
        }
        return dX;
    }

    /// @brief Aplica el optimizador a los pesos y bias
    void update_params(IOptimizer<T>& optimizer) override {
        optimizer.update(W_, dW_);

        // Convertir bias a 2D para poder aplicar el optimizador
        algebra::Tensor<T, 2> b2d(1, b_.shape()[0]);
        algebra::Tensor<T, 2> db2d(1, db_.shape()[0]);
        for (size_t j = 0; j < b_.shape()[0]; ++j) {
            b2d(0, j) = b_(j);
            db2d(0, j) = db_(j);
        }

        optimizer.update(b2d, db2d);

        // Restaurar bias en forma original
        for (size_t j = 0; j < b_.shape()[0]; ++j) {
            b_(j) = b2d(0, j);
        }
    }

    /// @brief Guarda los pesos y bias a un archivo de texto.
    void save_weights(const std::string& filename) const {
        std::ofstream file(filename);
        if (!file.is_open()) return;

        // Guardar dimensiones
        file << W_.shape()[0] << ' ' << W_.shape()[1] << '\n';

        // Guardar pesos W_
        for (size_t i = 0; i < W_.shape()[0]; ++i)
            for (size_t j = 0; j < W_.shape()[1]; ++j)
                file << W_(i, j) << ' ';
        file << '\n';

        // Guardar bias b_
        for (size_t j = 0; j < b_.size(); ++j)
            file << b_(j) << ' ';
        file << '\n';

        file.close();
    }

    /// @brief Carga pesos y bias desde un archivo de texto.
    void load_weights(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("No se pudo abrir el archivo de pesos: " + filename);
        }

        std::string dummy_line;
        std::getline(file, dummy_line); // Ignorar dimensiones

        // Leer pesos W_
        for (size_t i = 0; i < W_.shape()[0]; ++i) {
            for (size_t j = 0; j < W_.shape()[1]; ++j) {
                if (!(file >> W_(i, j))) {
                    throw std::runtime_error("Error leyendo pesos W en " + filename);
                }
            }
        }

        // Leer bias b_
        for (size_t j = 0; j < b_.size(); ++j) {
            if (!(file >> b_(j))) {
                throw std::runtime_error("Error leyendo bias b en " + filename);
            }
        }
    }

    /// @brief Devuelve una referencia a los pesos (W)
    const Tensor<T, 2>& weights() const {
        return W_;
    }
};

} // namespace utec::neural_network

#endif // DENSE_H
