#include <catch2/catch_test_macros.hpp>
#include "csrc/pipeline/node.h"
#include "csrc/pipeline/edge.h"
#include "csrc/pipeline/dag.h"
#include <algorithm>
#include <any>
#include <deque>
#include <memory>

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

TEST_CASE("Dag execute dataflow sequential", "[pipeline]") {
    Dag dag;
    dag.add_node(std::make_unique<DoubleNode>("d1"));
    dag.add_node(std::make_unique<DoubleNode>("d2"));
    REQUIRE(dag.connect("d1", "out", "d2", "in"));
    dag.get_node("d1")->set_input("in", std::any(3));
    REQUIRE(dag.build());
    REQUIRE(dag.execute());
    REQUIRE(dag.execution_order().size() == 2);
    REQUIRE(std::any_cast<int>(dag.get_node("d2")->get_output("out")) == 12);
}

TEST_CASE("Dag connect port/type validation", "[pipeline]") {
    Dag dag;
    // 不存在节点 / 不存在端口 → connect false
    dag.add_node(std::make_unique<DoubleNode>("a"));
    dag.add_node(std::make_unique<DoubleNode>("b"));
    REQUIRE_FALSE(dag.connect("nope", "out", "b", "in"));
    REQUIRE_FALSE(dag.connect("a", "nope", "b", "in"));
    // 类型不匹配：a.out(int) -> img_in.in(Image) → connect false
    dag.add_node(std::make_unique<ImageSinkNode>("sink"));
    REQUIRE_FALSE(dag.connect("a", "out", "sink", "in"));
}

TEST_CASE("Dag cycle detected by build", "[pipeline]") {
    Dag dag;
    dag.add_node(std::make_unique<DoubleNode>("a"));
    dag.add_node(std::make_unique<DoubleNode>("b"));
    REQUIRE(dag.connect("a", "out", "b", "in"));
    REQUIRE(dag.connect("b", "out", "a", "in"));
    REQUIRE_FALSE(dag.build());
}

TEST_CASE("Dag build fails on unconnected required input", "[pipeline]") {
    Dag dag;
    dag.add_node(std::make_unique<DoubleNode>("a"));
    dag.add_node(std::make_unique<DoubleNode>("b"));
    // b 未连边也未种子 → build false
    REQUIRE_FALSE(dag.build());
}

TEST_CASE("Dag fan-out executes once each", "[pipeline]") {
    Dag dag;
    dag.add_node(std::make_unique<DoubleNode>("a"));
    dag.add_node(std::make_unique<IdentityNode>("c"));
    dag.add_node(std::make_unique<IdentityNode>("d"));
    REQUIRE(dag.connect("a", "out", "c", "in"));
    REQUIRE(dag.connect("a", "out", "d", "in"));
    dag.get_node("a")->set_input("in", std::any(5));
    REQUIRE(dag.build());
    REQUIRE(dag.execute());
    REQUIRE(std::any_cast<int>(dag.get_node("c")->get_output("out")) == 10);
    REQUIRE(std::any_cast<int>(dag.get_node("d")->get_output("out")) == 10);
}

TEST_CASE("Dag exec order given by topo sort", "[pipeline]") {
    Dag dag;
    dag.add_node(std::make_unique<DoubleNode>("a"));
    dag.add_node(std::make_unique<DoubleNode>("b"));
    dag.add_node(std::make_unique<DoubleNode>("c"));
    REQUIRE(dag.connect("a", "out", "b", "in"));
    REQUIRE(dag.connect("a", "out", "c", "in"));
    dag.get_node("a")->set_input("in", std::any(2));
    REQUIRE(dag.build());
    auto order = dag.execution_order();
    REQUIRE(order.size() == 3);
    // a 必须在 b、c 之前
    auto ia = std::find(order.begin(), order.end(), "a");
    auto ib = std::find(order.begin(), order.end(), "b");
    auto ic = std::find(order.begin(), order.end(), "c");
    REQUIRE(ia < ib);
    REQUIRE(ia < ic);
    REQUIRE(dag.execute());
}
