#ifndef PARALLEL_INFERENCE_H
#define PARALLEL_INFERENCE_H

#include "thread_pool.h"
#include "../nn/neural_network.h"
#include "../algebra/tensor.h"
#include <vector>
#include <future>

namespace utec::neural_network {

    template <typename T>
    class ParallelInference {
    public:
        ParallelInference(NeuralNetwork<T>& model, size_t num_threads = 0)
            : model(model), pool(num_threads ? num_threads : std::thread::hardware_concurrency()) {}

        Tensor<T, 2> run_batch(const Tensor<T, 2>& input) {
            const size_t batch_size = input.shape()[0];
            const size_t num_threads = pool.size();
            const size_t chunk_size = (batch_size + num_threads - 1) / num_threads;

            Tensor<T, 2> output(batch_size, model.predict(input.slice(0, 1)).shape()[1]);
            std::vector<std::future<void>> futures;

            for (size_t i = 0; i < num_threads; ++i) {
                const size_t start = i * chunk_size;
                const size_t end = std::min(start + chunk_size, batch_size);

                if (start >= end) break;

                futures.push_back(pool.enqueue([&, start, end] {
                    auto input_slice = input.slice(start, end);
                    auto result = model.predict(input_slice);

                    for (size_t j = 0; j < result.shape()[0]; ++j) {
                        for (size_t k = 0; k < result.shape()[1]; ++k) {
                            output(start + j, k) = result(j, k);
                        }
                    }
                }));
            }

            for (auto& future : futures) {
                future.get();
            }

            return output;
        }

    private:
        NeuralNetwork<T>& model;
        utec::ThreadPool pool;
    };

} // namespace utec::neural_network

#endif // PARALLEL_INFERENCE_H