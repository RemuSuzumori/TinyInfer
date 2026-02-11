#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // 关键：自动转换 std::vector <-> Python list
#include "tiny_infer/tensor.hpp"
#include "tiny_infer/node.hpp"

namespace py = pybind11;
using namespace tiny_infer;

PYBIND11_MODULE(tiny_infer_py, m) {
    m.doc() = "TinyInfer: High-performance C++20 Inference Engine";

    // 映射 Tensor 类
    py::class_<Tensor>(m, "Tensor")
        .def(py::init<std::vector<size_t>>())
        .def("fill", &Tensor::fill)
        .def("shape", &Tensor::shape)
        // 假设你正在绑定 at 方法
        .def("at", [](tiny_infer::Tensor &t, size_t row, size_t col) {
            return t.at(row, col);
        })
        .def("__repr__", [](const Tensor &t) {
            return "<TinyInfer.Tensor shape=" + std::to_string(t.shape()[0]) + "x...>";
        });

    // 映射 Linear 层
    py::class_<Linear>(m, "Linear")
        .def(py::init<size_t, size_t>())
        .def("forward", &Linear::forward)
        .def("load_params", &Linear::load_params);

    // 映射 ReLU 层
    py::class_<ReLU>(m, "ReLU")
        .def(py::init<>())
        .def("forward", &ReLU::forward);
}