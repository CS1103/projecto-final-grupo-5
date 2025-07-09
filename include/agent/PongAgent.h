#pragma once
#ifndef PONG_AGENT_H
#define PONG_AGENT_H

#include "../nn/interfaces.h"
#include "../nn/dense.h"
#include "../nn/loss.h"
#include "../nn/optimizer.h"
#include "../nn/activation.h"
#include "EnvGym.h"
#include <algorithm>
#include <memory>
#include <fstream>
#include <sstream>
#include <vector>
#include <iostream>
#include <cstdlib>
#include <cmath>

namespace utec::nn {

constexpr std::size_t IN  = 5;
constexpr std::size_t H   = 16;
constexpr std::size_t OUT = 3;

struct PongSample {
    float ball_x, ball_y, ball_vx, ball_vy, paddle_y;
    int action;
    float reward;
};

template <typename T>
class PongAgent {
public:
    struct Sequential : utec::neural_network::ILayer<T> {
        std::unique_ptr<utec::neural_network::Dense<T>> l1;
        std::unique_ptr<utec::neural_network::ReLU<T>> act;
        std::unique_ptr<utec::neural_network::Dense<T>> l2;

        Sequential(std::unique_ptr<utec::neural_network::Dense<T>> a,
                   std::unique_ptr<utec::neural_network::ReLU<T>> b,
                   std::unique_ptr<utec::neural_network::Dense<T>> c)
            : l1(std::move(a)), act(std::move(b)), l2(std::move(c)) {}

        utec::algebra::Tensor<T, 2> forward(const utec::algebra::Tensor<T, 2>& x) override {
            return l2->forward(act->forward(l1->forward(x)));
        }

        utec::algebra::Tensor<T, 2> backward(const utec::algebra::Tensor<T, 2>& grad) override {
            return l1->backward(act->backward(l2->backward(grad)));
        }

        void update_params(utec::neural_network::IOptimizer<T>& opt) override {
            l1->update_params(opt);
            l2->update_params(opt);
        }
    };

private:
    std::unique_ptr<utec::neural_network::ILayer<T>> model_;

    static void initialize_weights(utec::algebra::Tensor<T, 2>& t) {
        for (size_t i = 0; i < t.shape()[0]; ++i)
            for (size_t j = 0; j < t.shape()[1]; ++j)
                t(i, j) = static_cast<T>((rand() / (T)RAND_MAX - 0.5) * 0.2);
    }

    static void initialize_zeros(utec::algebra::Tensor<T, 2>& t) {
        t.fill(0);
    }

    static utec::algebra::Tensor<T, 2> softmax(const utec::algebra::Tensor<T, 2>& z) {
        utec::algebra::Tensor<T, 2> s(z.shape());
        for (std::size_t i = 0; i < z.shape()[0]; ++i) {
            T maxz = z(i, 0);
            for (std::size_t j = 1; j < z.shape()[1]; ++j) maxz = std::max(maxz, z(i, j));
            T sum = 0;
            for (std::size_t j = 0; j < z.shape()[1]; ++j) {
                s(i, j) = std::exp(z(i, j) - maxz);
                sum += s(i, j);
            }
            for (std::size_t j = 0; j < z.shape()[1]; ++j) {
                s(i, j) /= sum;
            }
        }
        return s;
    }

public:
    explicit PongAgent(std::unique_ptr<utec::neural_network::ILayer<T>> m)
        : model_(std::move(m)) {}

    int act(const State& s, float epsilon = 0.1f) {
        if ((float)rand() / RAND_MAX < epsilon) {
            return (rand() % 3) - 1;
        }
        utec::algebra::Tensor<T, 2> input(1, IN);
        input(0, 0) = s.ball_x;
        input(0, 1) = s.ball_y;
        input(0, 2) = s.ball_vx;
        input(0, 3) = s.ball_vy;
        input(0, 4) = s.paddle_y;

        utec::algebra::Tensor<T, 2> output = model_->forward(input);

        T max_val = output(0, 0);
        int max_idx = 0;
        bool tie = false;
        for (size_t j = 1; j < output.shape()[1]; ++j) {
            if (output(0, j) > max_val) {
                max_val = output(0, j);
                max_idx = j;
                tie = false;
            } else if (output(0, j) == max_val) {
                tie = true;
            }
        }
        return max_idx == 0 ? -1 : (max_idx == 1 ? 0 : 1);
    }

    utec::neural_network::ILayer<T>* get_model() { return model_.get(); }
    utec::neural_network::Dense<T>* get_dense1() {
        auto* seq = dynamic_cast<Sequential*>(model_.get());
        return seq ? seq->l1.get() : nullptr;
    }
    utec::neural_network::Dense<T>* get_dense2() {
        auto* seq = dynamic_cast<Sequential*>(model_.get());
        return seq ? seq->l2.get() : nullptr;
    }

    static std::vector<PongSample> load_training_data(const std::string& filename) {
        std::vector<PongSample> data;
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "No se pudo abrir el archivo: " << filename << std::endl;
            return data;
        }
        std::string line;
        std::getline(file, line);
        while (std::getline(file, line)) {
            if (line.empty()) continue;
            std::stringstream ss(line);
            PongSample sample;
            std::string val;

            std::getline(ss, val, ','); sample.ball_x = std::stof(val);
            std::getline(ss, val, ','); sample.ball_y = std::stof(val);
            std::getline(ss, val, ','); sample.ball_vx = std::stof(val);
            std::getline(ss, val, ','); sample.ball_vy = std::stof(val);
            std::getline(ss, val, ','); sample.paddle_y = std::stof(val);
            std::getline(ss, val, ','); sample.action = std::stoi(val);
            std::getline(ss, val, ','); sample.reward = std::stof(val);

            data.push_back(sample);
        }
        return data;
    }

    static std::unique_ptr<utec::neural_network::ILayer<T>> train_from_csv(
        const std::string& csv_path, int epochs = 100, T lr = 0.001) {

        auto data = load_training_data(csv_path);

        auto capa1 = std::make_unique<utec::neural_network::Dense<T>>(IN, H, initialize_weights, initialize_zeros);
        auto relu = std::make_unique<utec::neural_network::ReLU<T>>();
        auto capa2 = std::make_unique<utec::neural_network::Dense<T>>(H, OUT, initialize_weights, initialize_zeros);

        auto model = std::make_unique<Sequential>(std::move(capa1), std::move(relu), std::move(capa2));
        utec::neural_network::SGD<T> optimizer(lr * 0.1);

        for (int epoch = 0; epoch < epochs; ++epoch) {
            T total_loss = 0;
            for (const auto& sample : data) {
                utec::algebra::Tensor<T, 2> input(1, IN);
                input(0, 0) = sample.ball_x;
                input(0, 1) = sample.ball_y;
                input(0, 2) = sample.ball_vx;
                input(0, 3) = sample.ball_vy;
                input(0, 4) = sample.paddle_y;

                utec::algebra::Tensor<T, 2> target(1, OUT);
                target(0, 0) = (sample.action == -1) ? 1 : 0;
                target(0, 1) = (sample.action == 0) ? 1 : 0;
                target(0, 2) = (sample.action == 1) ? 1 : 0;

                auto z = model->forward(input);
                auto y_pred = softmax(z);
                utec::algebra::Tensor<T, 2> grad(1, OUT);
                T loss = 0;
                for (int i = 0; i < OUT; ++i) {
                    grad(0, i) = y_pred(0, i) - target(0, i);
                    loss += -target(0, i) * std::log(std::max((T)1e-6, y_pred(0, i)));
                }
                total_loss += loss / OUT;
                model->backward(grad);
                model->update_params(optimizer);
            }
            if (epoch % 10 == 0) {
                std::cout << "Epoch " << epoch << ", Loss: " << total_loss / data.size() << "\n";
            }
        }
        return model;
    }
    //  Carga pesos externos (dense-1 y dense-2) y construye la MLP 5-16-3
    static std::unique_ptr<utec::neural_network::ILayer<T>>
    create_sequential_with_weights(const std::string& w1,
                                   const std::string& w2)
    {
        constexpr std::size_t IN  = 5;
        constexpr std::size_t H   = 16;
        constexpr std::size_t OUT = 3;

        auto l1  = std::make_unique<utec::neural_network::Dense<T>>
                                      (IN , H , initialize_zeros, initialize_zeros);
        auto act = std::make_unique<utec::neural_network::ReLU <T>>();
        auto l2  = std::make_unique<utec::neural_network::Dense<T>>
                                      (H  , OUT, initialize_zeros, initialize_zeros);

        l1->load_weights(w1);
        l2->load_weights(w2);

        return std::make_unique<Sequential>(std::move(l1),
                                            std::move(act),
                                            std::move(l2));
    }
};

} // namespace utec::nn

#endif // PONG_AGENT_H
