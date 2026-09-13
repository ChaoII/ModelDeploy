#pragma once
#include <string>

class ModelFetcher {
public:
    ModelFetcher(std::string cache_dir, std::string s3_endpoint = "")
        : cache_dir_(std::move(cache_dir)), s3_endpoint_(std::move(s3_endpoint)) {}
    bool fetch(const std::string& url, std::string* local_path, std::string* err) const;

private:
    std::string cache_dir_, s3_endpoint_;
    bool download(const std::string& base, const std::string& path,
                  const std::string& dest, std::string* err) const;
};
