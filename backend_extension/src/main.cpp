#include <pybind11/pybind11.h>
#include "neural_probe.h"

namespace py = pybind11;

PYBIND11_MODULE(neural_probe, m) {
    m.doc() = "Neural-Flow High-Performance Probe Extensions";

    py::enum_<NF_Result>(m, "Result")
        .value("OK", NF_OK)
        .value("ERR_NOT_STARTED", NF_ERR_NOT_STARTED)
        .value("ERR_INVALID_DIMS", NF_ERR_INVALID_DIMS)
        .value("ERR_BUFFER_FULL", NF_ERR_BUFFER_FULL)
        .value("ERR_NULL_POINTER", NF_ERR_NULL_POINTER)
        .value("ERR_LAYER_NOT_FOUND", NF_ERR_LAYER_NOT_FOUND)
        .value("ERR_INVALID_THRESHOLD", NF_ERR_INVALID_THRESHOLD)
        .export_values();

    m.def("start_server", &start_server, "Start the uWebSockets activation server",
          py::arg("host") = "localhost",
          py::arg("port") = 9000);

    m.def("stop_server", &stop_server, "Stop the server");

    m.def("set_threshold", &set_threshold, "Set activation threshold for sparse capture",
          py::arg("threshold"));

    m.def("set_accumulation_steps", &set_accumulation_steps, 
          "Set number of steps to accumulate before broadcasting",
          py::arg("steps"));

    m.def("set_broadcast_interval", &set_broadcast_interval,
          "Set minimum interval between broadcasts in milliseconds",
          py::arg("interval_ms"));

    m.def("set_max_sparse_points", &set_max_sparse_points,
          "Set maximum number of sparse activation points per packet",
          py::arg("max_points"));

    m.def("log_activation", &log_activation, "Log a tensor activation for a specific layer",
          py::arg("layer_id"),
          py::arg("tensor"));
}
