#include <catch2/catch_test_macros.hpp>
#include <filesystem>
#include <fstream>
#include <functional>
#include <random>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>
#include "serving/config.h"
#include "serving/model_repo.h"

namespace fs = std::filesystem;
using namespace modeldeploy::serving;

namespace {

// 返回唯一临时仓库根（不放在仓库/源目录）；调用方可 remove_all 清理。
std::string make_temp_repo() {
    static std::mt19937 rng{std::random_device{}()};
    auto base = fs::temp_directory_path() /
                ("md_serving_" + std::to_string(static_cast<unsigned>(rng())));
    fs::create_directories(base);
    return base.string();
}

// 建 repo/{name}/{ver}/model.onnx（写一个占位字节，让合法目录被登记）。
void write_model(const std::string& repo, const std::string& name, const std::string& ver) {
    auto dir = fs::path(repo) / name / ver;
    fs::create_directories(dir);
    std::ofstream ofs(dir / "model.onnx", std::ios::binary);
    ofs.put(static_cast<char>(0));
}

// 注入假 InferFn：调用时把 name/version 写进 out["handle"]，用于核对句柄身份。
HandleBuilder echo_builder() {
    return [](const std::string& name, const std::string& ver, const std::string&) {
        std::string key = name + "/" + ver;
        return InferFn([key](const nlohmann::json& in, nlohmann::json* out, std::string* err) {
            (void)in;
            (void)err;
            if (out) (*out)["handle"] = key;
            return true;
        });
    };
}

std::string handle_of(const ModelHandle& h) {
    nlohmann::json out;
    std::string err;
    REQUIRE(h.infer(nlohmann::json::object(), &out, &err));
    REQUIRE(out.contains("handle"));
    return out["handle"].get<std::string>();
}

}  // namespace

TEST_CASE("ModelRepo scan get list", "[serving]") {
    auto repo = make_temp_repo();
    ServingConfig cfg;
    cfg.model_repo = repo;
    write_model(repo, "det", "1");
    write_model(repo, "det", "2");
    write_model(repo, "cls", "latest");

    ModelRepo m(cfg, echo_builder());
    auto changed = m.scan();
    REQUIRE(std::find(changed.begin(), changed.end(), "det") != changed.end());
    REQUIRE(std::find(changed.begin(), changed.end(), "cls") != changed.end());

    ModelHandle h;
    REQUIRE(m.get("det", "latest", &h));
    REQUIRE(h.name == "det");
    REQUIRE(h.version == "2");
    REQUIRE(h.ready);
    REQUIRE(handle_of(h) == "det/2");

    REQUIRE(m.get("det", "2", &h));
    REQUIRE(h.version == "2");
    REQUIRE(handle_of(h) == "det/2");

    REQUIRE(m.get("det", "1", &h));
    REQUIRE(h.version == "1");
    REQUIRE(handle_of(h) == "det/1");

    REQUIRE(m.get("cls", "latest", &h));
    REQUIRE(h.version == "latest");

    REQUIRE(m.get("cls", "", &h));  // 空串也解析 latest
    REQUIRE(h.version == "latest");

    auto all = m.list();
    REQUIRE(all.size() == 2);
    std::vector<std::string> names;
    for (auto& e : all) names.push_back(e.name);
    REQUIRE(std::find(names.begin(), names.end(), "det") != names.end());
    REQUIRE(std::find(names.begin(), names.end(), "cls") != names.end());

    fs::remove_all(repo);
}

TEST_CASE("ModelRepo latest numeric order", "[serving]") {
    auto repo = make_temp_repo();
    ServingConfig cfg;
    cfg.model_repo = repo;
    write_model(repo, "m", "1");
    write_model(repo, "m", "2");
    write_model(repo, "m", "10");

    ModelRepo m(cfg);
    m.scan();

    ModelHandle h;
    REQUIRE(m.get("m", "latest", &h));
    REQUIRE(h.version == "10");  // 数字比较，否则 10<2

    fs::remove_all(repo);
}

TEST_CASE("ModelRepo hot update", "[serving]") {
    auto repo = make_temp_repo();
    ServingConfig cfg;
    cfg.model_repo = repo;
    write_model(repo, "det", "1");

    ModelRepo m(cfg, echo_builder());
    m.scan();

    ModelHandle oldh;
    REQUIRE(m.get("det", "latest", &oldh));
    REQUIRE(oldh.version == "1");

    // 旧持有者：值拷贝的 InferFn，可在热换后继续完成。
    auto old_infer = oldh.infer;

    write_model(repo, "det", "2");
    auto changed = m.scan();
    REQUIRE(std::find(changed.begin(), changed.end(), "det") != changed.end());

    ModelHandle newh;
    REQUIRE(m.get("det", "latest", &newh));
    REQUIRE(newh.version == "2");

    // 旧版本句柄仍可读取。
    REQUIRE(m.get("det", "1", &newh));
    REQUIRE(newh.version == "1");

    // 旧持有的 InferFn 不受热换影响。
    nlohmann::json out;
    std::string err;
    REQUIRE(old_infer(nlohmann::json::object(), &out, &err));
    REQUIRE(out["handle"] == "det/1");

    fs::remove_all(repo);
}

TEST_CASE("ModelRepo unknown names/versions", "[serving]") {
    auto repo = make_temp_repo();
    ServingConfig cfg;
    cfg.model_repo = repo;
    write_model(repo, "det", "1");

    ModelRepo m(cfg);
    m.scan();

    ModelHandle h;
    REQUIRE_FALSE(m.get("nope", "latest", &h));
    REQUIRE_FALSE(m.get("det", "v999", &h));

    fs::remove_all(repo);
}
