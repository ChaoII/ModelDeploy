#include "frame_pool.hpp"
#include <algorithm>
#include <cstring>
#include <iostream>

FramePool::FramePool(size_t pool_size)
    : pool_size_(pool_size) {
}

FramePool::~FramePool() {
    clear();
}

uint8_t* FramePool::acquire(size_t size) {
    std::lock_guard<std::mutex> lock(mtx_);

    // 查找足够大的空闲块
    for (auto it = pool_.begin(); it != pool_.end(); ++it) {
        if (it->size >= size) {
            uint8_t* ptr = it->ptr;
            in_use_[ptr] = *it;
            pool_.erase(it);
            return ptr;
        }
    }

    // 没有合适块 → 新分配
    uint8_t* ptr = nullptr;
    if (cudaMalloc(&ptr, size) != cudaSuccess) {
        return nullptr;
    }
    total_allocated_ += size;
    in_use_[ptr] = {ptr, size};
    return ptr;
}

void FramePool::release(uint8_t* ptr) {
    if (!ptr) return;
    std::lock_guard<std::mutex> lock(mtx_);

    auto it = in_use_.find(ptr);
    if (it == in_use_.end()) return; // 不属于此池

    if (pool_.size() < pool_size_) {
        pool_.push_back(it->second);
    } else {
        total_allocated_ -= it->second.size;
        cudaFree(ptr);
    }
    in_use_.erase(it);
}

void FramePool::clear() {
    std::lock_guard<std::mutex> lock(mtx_);

    // 释放所有在用的
    for (auto& [ptr, blk] : in_use_) {
        cudaFree(ptr);
    }
    in_use_.clear();

    // 释放所有缓存的
    for (auto& blk : pool_) {
        cudaFree(blk.ptr);
    }
    pool_.clear();
    total_allocated_ = 0;
}

size_t FramePool::cached_count() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return pool_.size();
}
