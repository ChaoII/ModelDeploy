#pragma once
#include <cuda_runtime.h>
#include <vector>
#include <map>
#include <mutex>
#include <memory>

/// CUDA 显存池：避免每帧反复 cudaMalloc / cudaFree
class FramePool {
public:
    explicit FramePool(size_t pool_size = 16);
    ~FramePool();

    /// 分配或复用一块至少 size 字节的显存
    uint8_t* acquire(size_t size);

    /// 归还显存到池中
    void release(uint8_t* ptr);

    /// 清空池中所有缓存
    void clear();

    /// 当前池中缓存块数
    size_t cached_count() const;
    size_t total_allocated() const { return total_allocated_; }

private:
    struct Block {
        uint8_t* ptr = nullptr;
        size_t size = 0;
    };

    std::vector<Block> pool_;        // 空闲块
    std::map<uint8_t*, Block> in_use_; // 已分配块
    size_t pool_size_;
    size_t total_allocated_ = 0;
    mutable std::mutex mtx_;
};
