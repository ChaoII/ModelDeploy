#include "model_fetcher.hpp"
#include "httplib.h"
#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
#include <thread>

namespace fs = std::filesystem;

namespace {
/// 规范化 + 小写（Windows 路径大小写不敏感），并统一为 '/' 分隔符
std::string norm_lower(const fs::path& p) {
    std::string s = p.lexically_normal().generic_string();
    for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    while (s.size() > 1 && s.back() == '/') s.pop_back();
    return s;
}

/// target 是否位于 root 之内（含等于 root）；root 为空 → false
bool under_root(const std::string& target, const std::string& root) {
    if (root.empty()) return false;
    if (target == root) return true;
    return target.size() > root.size() && target.compare(0, root.size(), root) == 0 &&
           target[root.size()] == '/';
}
}  // namespace

static bool basename_of(const std::string& url, std::string* out) {
    const auto q = url.find('?');
    const std::string clean = q == std::string::npos ? url : url.substr(0, q);
    const auto slash = clean.find_last_of('/');
    const std::string name = slash == std::string::npos ? clean : clean.substr(slash + 1);
    if (name.empty() || name == "." || name == "..") return false;
    if (name.find('/') != std::string::npos || name.find('\\') != std::string::npos) return false;
    for (unsigned char c : name) {
        if (c < 0x20) return false;
    }
    if (name.find_first_of("<>:\"|?*") != std::string::npos) return false;
    out->assign(name);
    return true;
}

static std::string temp_path_for(const std::string& dest) {
    // 每线程独立 RNG（thread_local 避免并发调用的数据竞争），叠加进程级原子
    // 计数器，保证并发下载即使种子相同也各自得到唯一临时文件名。
    static std::atomic<uint64_t> seq{0};
    thread_local std::mt19937_64 rng(std::random_device{}());
    const uint64_t n = seq.fetch_add(1, std::memory_order_relaxed);
    return dest + ".tmp-" + std::to_string(rng()) + "-" + std::to_string(n);
}

bool ModelFetcher::download(const std::string& base, const std::string& path,
                            const std::string& dest, std::string* err) const {
    httplib::Client cli(base);
    cli.set_connection_timeout(5, 0);
    cli.set_read_timeout(120, 0);
    auto res = cli.Get(path.c_str());
    if (!res || res->status != 200) {
        if (err) *err = "download failed (" + std::to_string(res ? res->status : -1) + "): " + base + path;
        return false;
    }
    if (res->body.empty()) { if (err) *err = "downloaded file empty"; return false; }

    const std::string tmp = temp_path_for(dest);
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        if (!f.is_open()) { if (err) *err = "cannot write " + dest; return false; }
        f.write(res->body.data(), static_cast<std::streamsize>(res->body.size()));
        f.flush();
        if (!f.good()) {
            f.close();
            std::error_code rmec; fs::remove(tmp, rmec);
            if (err) *err = "failed to write " + dest;
            return false;
        }
        f.close();
        if (!f.good()) {
            std::error_code rmec; fs::remove(tmp, rmec);
            if (err) *err = "failed to write " + dest;
            return false;
        }
    }

    std::error_code ec;
    fs::rename(tmp, dest, ec);
    // 并发发布同一 dest（同名模型被多任务同时拉取）时，rename 可能因短暂的
    // 共享冲突而失败。这里必须**退避重试且不先删除 dest**：删除会与其它线程的
    // rename 交错，造成丢更新（调用方收到 false，且窗口内容缺失）。
    for (int attempt = 0; ec && attempt < 100; ++attempt) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        ec.clear();
        fs::rename(tmp, dest, ec);
    }
    if (ec) {
        // 兼容不支持覆盖式 rename 的文件系统：删除过期 dest 后最后再试一次。
        std::error_code rmec; fs::remove(dest, rmec);
        ec.clear();
        fs::rename(tmp, dest, ec);
    }
    if (ec) {
        std::error_code rmec; fs::remove(tmp, rmec);
        if (err) *err = "cannot write " + dest + ": " + ec.message();
        return false;
    }
    return true;
}

bool ModelFetcher::local_path_allowed(const std::string& p, std::string* err) const {
    std::error_code ec;
    // weakly_canonical：解析 '..' 与符号链接（不存在时按词法归一）后再做前缀比较，
    // 避免 "root/../../etc/passwd" 之类的绕过。
    const std::string target = norm_lower(fs::weakly_canonical(fs::absolute(p, ec), ec));
    if (ec) return true;   // 归一失败交由后续 fs::exists 报错，不在这里误拒
    // 模型缓存目录始终允许（下载落地点）
    std::error_code ec2;
    if (under_root(target, norm_lower(fs::weakly_canonical(fs::absolute(cache_dir_, ec2), ec2))))
        return true;
    if (local_root_.empty()) return true;   // 未配置根：保持既有行为（启动时告警）
    if (under_root(target, norm_lower(fs::weakly_canonical(fs::absolute(local_root_, ec2), ec2))))
        return true;
    if (err)
        *err = "local model path outside allowed root (--model-root): " + p;
    return false;
}

bool ModelFetcher::fetch(const std::string& url, std::string* local_path, std::string* err) const {
    if (url.empty()) { if (err) *err = "empty model url"; return false; }
    auto fail = [&](const std::string& m) { if (err) *err = m; return false; };

    // 本地路径 / file://（受 --model-root 白名单约束）
    if (url.rfind("file://", 0) == 0) {
        const std::string p = url.substr(7);
        if (!local_path_allowed(p, err)) return false;
        if (!fs::exists(p)) return fail("local model not found: " + p);
        if (local_path) *local_path = p;
        return true;
    }
    if (url.find("://") == std::string::npos) {
        if (!local_path_allowed(url, err)) return false;
        if (!fs::exists(url)) return fail("local model not found: " + url);
        if (local_path) *local_path = url;
        return true;
    }

    std::error_code ec;
    fs::create_directories(cache_dir_, ec);
    std::string name;
    if (!basename_of(url, &name)) return fail("invalid model url (no filename): " + url);
    const std::string dest = (fs::path(cache_dir_) / name).string();
    // 用 error_code 重载：并发发布时 dest 可能在 exists() 与 file_size() 之间被
    // 替换/短暂移除，抛出 filesystem_error（工作线程将 terminate）。命中失败时
    // 直接走下载路径即可。
    std::error_code cache_ec;
    if (fs::exists(dest, cache_ec)) {
        const auto sz = fs::file_size(dest, cache_ec);
        if (!cache_ec && sz > 0) { if (local_path) *local_path = dest; return true; }
    }

    if (url.rfind("http://", 0) == 0 || url.rfind("https://", 0) == 0) {
#ifndef CPPHTTPLIB_OPENSSL_SUPPORT
        if (url.rfind("https://", 0) == 0) return fail("https not supported in this build");
#endif
        const auto scheme = url.find("://");
        const std::string rest = url.substr(scheme + 3);
        const auto slash = rest.find('/');
        const std::string base = url.substr(0, scheme + 3 + (slash == std::string::npos ? rest.size() : slash));
        const std::string path = slash == std::string::npos ? "/" : rest.substr(slash);
        if (!download(base, path, dest, err)) return false;
        if (local_path) *local_path = dest;
        return true;
    }

    if (url.rfind("s3://", 0) == 0) {
        if (s3_endpoint_.empty()) return fail("s3 url requires --s3-endpoint: " + url);
        // path-style: <endpoint>/<bucket>/<key>
        std::string rest = url.substr(5);
        const auto slash = rest.find('/');
        if (slash == std::string::npos) return fail("invalid s3 url: " + url);
        const std::string bucket = rest.substr(0, slash);
        const std::string key = rest.substr(slash + 1);
        const std::string base = s3_endpoint_;
        const std::string path = "/" + bucket + "/" + key;
        if (!download(base, path, dest, err)) return false;
        if (local_path) *local_path = dest;
        return true;
    }

    return fail("unsupported model url scheme: " + url);
}
