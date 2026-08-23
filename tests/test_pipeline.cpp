#include <catch2/catch_test_macros.hpp>
#include "csrc/pipeline/node.h"
#include "csrc/pipeline/edge.h"
#include <any>

using namespace modeldeploy::pipeline;

// 桩：int 输入 -> int 输出（乘 2）。可派生子类。
struct DoubleNode : Node {
    DoubleNode(std::string n) : Node(std::move(n), {{"in", "int"}}, {{"out", "int"}}) {}
    bool run() override {
        int v;
        if (!get_in<int>("in", &v)) return false;
        set_out<int>("out", v * 2);
        return true;
    }
};

// 桩：Image 类型输入，无输出（用于类型不匹配连接的校验）。
struct ImageSinkNode : Node {
    ImageSinkNode(std::string n) : Node(std::move(n), {{"in", "Image"}}, {}) {}
    bool run() override { return true; }
};

// 桩：int 输入 -> int 输出（原样透传）。
struct IdentityNode : Node {
    IdentityNode(std::string n) : Node(std::move(n), {{"in", "int"}}, {{"out", "int"}}) {}
    bool run() override {
        int v;
        if (!get_in<int>("in", &v)) return false;
        set_out<int>("out", v);
        return true;
    }
};

TEST_CASE("Node port schema", "[pipeline]") {
    DoubleNode n("d");
    REQUIRE(n.name() == "d");
    REQUIRE(n.inputs().size() == 1);
    REQUIRE(n.inputs()[0].name == "in");
    REQUIRE(n.inputs()[0].type == "int");
    REQUIRE(n.outputs().size() == 1);
    REQUIRE(n.outputs()[0].type == "int");
}

TEST_CASE("Node set/get data via any", "[pipeline]") {
    DoubleNode n("d");
    REQUIRE_FALSE(n.has_input("in"));
    n.set_input("in", std::any(3));
    REQUIRE(n.has_input("in"));
    REQUIRE(n.run());
    REQUIRE(std::any_cast<int>(n.get_output("out")) == 6);
}

TEST_CASE("Edge type compatibility", "[pipeline]") {
    Edge same{"a", "out", "int", "b", "in", "int"};
    REQUIRE(edge_type_compatible(same));
    Edge diff{"a", "out", "int", "b", "in", "Image"};
    REQUIRE_FALSE(edge_type_compatible(diff));
}
