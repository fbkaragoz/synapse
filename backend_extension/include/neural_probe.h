#ifndef NEURAL_PROBE_H
#define NEURAL_PROBE_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "protocol.h"

namespace py = pybind11;

NF_Result start_server(const std::string& host, int port);
void stop_server();
void set_threshold(float threshold);

NF_Result set_accumulation_steps(uint32_t steps);
NF_Result set_broadcast_interval(uint32_t interval_ms);
NF_Result set_max_sparse_points(uint32_t max_points);

void handle_control_message(const uint8_t* data, size_t len);
std::vector<uint8_t> get_state_snapshot();
std::vector<std::vector<uint8_t>> get_state_snapshot_packets();
std::vector<uint8_t> get_model_meta_packet();

NF_Result log_activation(int layer_id, py::array_t<float> tensor);

NF_Result log_gradient(int layer_id, py::array_t<float> grad, py::array_t<float> weight);

void flush_gradient_batch();

void set_training_step(uint64_t step);
uint64_t get_training_step();

#endif
