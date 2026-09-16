#pragma once
#include <string>

class ModelFetcher {
public:
    ModelFetcher(std::string cache_dir, std::string s3_endpoint = "")
        : cache_dir_(std::move(cache_dir)), s3_endpoint_(std::move(s3_endpoint)) {}
    bool fetch(const std::string& url, std::string* local_path, std::string* err) const;

    /// 本地模型加载根目录白名单（可选）。
    /// 设置后，`file://` 与裸本地路径必须落在该根目录内（或模型缓存目录内），否则拒绝：
    /// 防止云侧/攻击者通过 model_path 让 Agent 加载任意本地文件（安全审计 HIGH #6）。
    /// 未设置时保持既有行为（允许任意本地路径），但 AgentRuntime 会在启动时告警。
    void set_local_root(std::string root) { local_root_ = std::move(root); }

private:
    std::string cache_dir_, s3_endpoint_;
    std::string local_root_;
    bool download(const std::string& base, const std::string& path,
                  const std::string& dest, std::string* err) const;
    /// 本地路径是否允许加载（缓存目录始终允许；否则须在 local_root_ 内）
    bool local_path_allowed(const std::string& p, std::string* err) const;
};
