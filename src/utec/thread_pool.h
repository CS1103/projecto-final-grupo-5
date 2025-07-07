#ifndef THREAD_POOL_H
#define THREAD_POOL_H

#include "concurrent_queue.h"
#include <vector>
#include <thread>
#include <functional>
#include <future>
#include <stdexcept>

namespace utec {

    class ThreadPool {
    public:
        explicit ThreadPool(size_t threads = std::thread::hardware_concurrency())
            : stop(false) {
            for(size_t i = 0; i < threads; ++i)
                workers.emplace_back([this] {
                    for(;;) {
                        std::function<void()> task;

                        if(!tasks.pop(task))
                            break;

                        task();
                    }
                });
        }

        template<class F, class... Args>
        auto enqueue(F&& f, Args&&... args)
            -> std::future<typename std::result_of<F(Args...)>::type> {
            using return_type = typename std::result_of<F(Args...)>::type;

            auto task = std::make_shared<std::packaged_task<return_type()>>(
                std::bind(std::forward<F>(f), std::forward<Args>(args)...)
            );

            std::future<return_type> res = task->get_future();
            tasks.push([task](){ (*task)(); });
            return res;
        }

        ~ThreadPool() {
            tasks.stop();
            for(std::thread &worker: workers)
                if(worker.joinable()) worker.join();
        }

        size_t size() const { return workers.size(); }

    private:
        std::vector<std::thread> workers;
        ConcurrentQueue<std::function<void()>> tasks;
        std::atomic<bool> stop;
    };

} // namespace utec

#endif // THREAD_POOL_H