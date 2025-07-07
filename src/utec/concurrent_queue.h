#ifndef CONCURRENT_QUEUE_H
#define CONCURRENT_QUEUE_H

#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>

namespace utec {

    template <typename T>
    class ConcurrentQueue {
    public:
        void push(const T& item) {
            {
                std::lock_guard<std::mutex> lock(mutex_);
                queue_.push(item);
            }
            cond_.notify_one();
        }

        void push(T&& item) {
            {
                std::lock_guard<std::mutex> lock(mutex_);
                queue_.push(std::move(item));
            }
            cond_.notify_one();
        }

        bool pop(T& item) {
            std::unique_lock<std::mutex> lock(mutex_);
            cond_.wait(lock, [this]() { return !queue_.empty() || stop_; });

            if (stop_) return false;
            if (queue_.empty()) return false;

            item = std::move(queue_.front());
            queue_.pop();
            return true;
        }

        size_t size() const {
            std::lock_guard<std::mutex> lock(mutex_);
            return queue_.size();
        }

        bool empty() const {
            std::lock_guard<std::mutex> lock(mutex_);
            return queue_.empty();
        }

        void stop() {
            {
                std::lock_guard<std::mutex> lock(mutex_);
                stop_ = true;
            }
            cond_.notify_all();
        }

        void restart() {
            stop_ = false;
        }

    private:
        std::queue<T> queue_;
        mutable std::mutex mutex_;
        std::condition_variable cond_;
        std::atomic<bool> stop_{false};
    };

} // namespace utec

#endif // CONCURRENT_QUEUE_H