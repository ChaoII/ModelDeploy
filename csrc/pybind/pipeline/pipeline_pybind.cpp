#include <pybind11/pybind11.h>
#include <functional>
#include <any>
#include "csrc/pipeline/node.h"
#include "csrc/pipeline/dag.h"
#include "csrc/pipeline/planner.h"

namespace py = pybind11;
using namespace modeldeploy::pipeline;

namespace modeldeploy::pipeline {

namespace {
    // Python 友好的 TransformNode：包装一个 py 可调用；in/out 端口 type 为注册时传入的 type 字符串
    class PyTransformNode : public Node {
    public:
        PyTransformNode(std::string name, std::string type, py::object fn)
            : Node(std::move(name), {{"in", type}}, {{"out", type}}),
              fn_(std::move(fn)) {}
        bool run() override {
            py::object v;
            if (!get_in<py::object>("in", &v)) return false;
            py::object r = fn_(v);
            set_out<py::object>("out", r);
            return true;
        }
    private:
        py::object fn_;
    };
} // namespace

void bind_pipeline(pybind11::module& m) {
    // Node：供 Dag.get_node 结果读写数据；以 py::object 承载
    py::class_<Node>(m, "Node")
        .def("name", &Node::name)
        .def("set_input", [](Node& n, const std::string& p, py::object v) {
            n.set_input(p, std::any(py::object(v)));
        })
        .def("get_output", [](Node& n, const std::string& p) -> py::object {
            std::any a = n.get_output(p);
            if (!a.has_value()) return py::none();
            return std::any_cast<py::object>(a);
        });

    // Dag：节点由 Planner 建立；Python 侧主要 get_node + build + execute
    py::class_<Dag>(m, "Dag")
        .def(py::init<>())
        .def("get_node", &Dag::get_node, py::return_value_policy::reference_internal)
        .def("connect", &Dag::connect)
        .def("build", &Dag::build)
        .def("execute", &Dag::execute)
        .def("execution_order", &Dag::execution_order);

    // Planner：注册 Python 变换函数，据 DSL 布图返回 Dag
    py::class_<Planner>(m, "Planner")
        .def(py::init<>())
        .def("register_transform", [](Planner& p, const std::string& name,
                                      const std::string& type, py::object fn) {
            p.register_model(name, [fn = py::object(fn), type](const std::string& inst) {
                return std::unique_ptr<Node>(new PyTransformNode(inst, type, fn));
            });
        }, py::arg("name"), py::arg("type"), py::arg("fn"))
        .def("build", [](Planner& p, const std::string& spec) { return p.build(spec); });

    m.attr("__doc__") = "Pipeline DAG orchestration module.";
}

} // namespace modeldeploy::pipeline
