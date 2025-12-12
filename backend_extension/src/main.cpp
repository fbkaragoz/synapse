#include <pybind11/pybind11.h>
#include "neural_probe.h"

namespace py = pybind11;

PYBIND11_MODULE(neural_probe, m) {
    m.doc() = "Neural-Flow High-Performance Probe Extensions";

    m.def("start_server", &start_server, "Start the uWebSockets activation server",
          py::arg("host") = "localhost",
          py::arg("port") = 9000);

    m.def("stop_server", &stop_server, "Stop the server");

    m.def("set_threshold", &set_threshold, "Set activation threshold for sparse capture",
          py::arg("threshold"));

    m.def("log_activation", &log_activation, "Log a tensor activation for a specific layer",
          py::arg("layer_id"),
          py::arg("tensor"));
}
