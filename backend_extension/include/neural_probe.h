#ifndef NEURAL_PROBE_H
#define NEURAL_PROBE_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

// Global initialization of the background server
void start_server(const std::string& host, int port);
void stop_server();
void set_threshold(float threshold);

// Control/Networking
void handle_control_message(const uint8_t* data, size_t len);
std::vector<uint8_t> get_state_snapshot();
std::vector<uint8_t> get_model_meta_packet();

// The main hook function called from Python
// layer_id: Unique ID for the layer
// tensor: The activation tensor (numpy array)
void log_activation(int layer_id, py::array_t<float> tensor);

#endif // NEURAL_PROBE_H
