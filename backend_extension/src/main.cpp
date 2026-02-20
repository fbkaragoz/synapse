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

    m.def("log_gradient", &log_gradient, "Log gradient statistics for a layer",
          py::arg("layer_id"),
          py::arg("grad"),
          py::arg("weight"));

    m.def("flush_gradient_batch", &flush_gradient_batch, 
          "Flush pending gradient statistics as a batch packet");

    m.def("set_training_step", &set_training_step,
          "Set the current training step counter",
          py::arg("step"));

    m.def("get_training_step", &get_training_step,
          "Get the current training step counter");

    m.def("set_use_v2", &set_use_v2,
          "Enable or disable V2 extended statistics (13 metrics)",
          py::arg("use_v2"));

    m.def("get_use_v2", &get_use_v2,
          "Check if V2 extended statistics mode is enabled");

    m.def("set_sample_rate", &set_sample_rate,
          "Set sample rate - capture every Nth forward pass (default 1)",
          py::arg("rate"));

    m.def("get_sample_rate", &get_sample_rate,
          "Get current sample rate");

    m.def("set_layer_selection_mode", &set_layer_selection_mode,
          "Set layer selection mode: 0=all, 1=whitelist, 2=blacklist",
          py::arg("mode"));

    m.def("add_layer_to_whitelist", &add_layer_to_whitelist,
          "Add a layer ID to the whitelist",
          py::arg("layer_id"));

    m.def("add_layer_to_blacklist", &add_layer_to_blacklist,
          "Add a layer ID to the blacklist",
          py::arg("layer_id"));

    m.def("clear_layer_selection", &clear_layer_selection,
          "Clear layer selection (capture all layers)");

    py::enum_<NF_AttentionMode>(m, "AttentionMode")
        .value("FULL", NF_ATTENTION_FULL)
        .value("TOP_K", NF_ATTENTION_TOP_K)
        .value("THRESHOLD", NF_ATTENTION_THRESHOLD)
        .value("BAND", NF_ATTENTION_BAND)
        .export_values();

    m.def("log_attention", &log_attention,
          "Log attention weights for a transformer layer/head",
          py::arg("layer_id"),
          py::arg("head_id"),
          py::arg("weights"),
          py::arg("mode") = NF_ATTENTION_TOP_K,
          py::arg("max_entries") = 1000,
          py::arg("threshold") = 0.0f);
}
