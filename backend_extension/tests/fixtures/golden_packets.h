#pragma once

#include <cstdint>
#include <vector>
#include <fstream>
#include "protocol_parser.h"

namespace nf {
namespace fixtures {

inline std::vector<uint8_t> generate_layer_summary_fixture() {
    std::vector<ParsedLayerSummary> summaries = {
        {0, 512, 0.1f, 1.0f},
        {1, 1024, 0.2f, 2.0f},
        {2, 2048, 0.3f, 3.0f}
    };
    return build_layer_summary_packet(summaries, 1, 1000000000);
}

inline std::vector<uint8_t> generate_control_fixture() {
    return build_control_packet(NF_OP_SET_THRESHOLD, 0, 0.5f, 2, 2000000000);
}

inline std::vector<uint8_t> generate_sparse_fixture() {
    std::vector<uint8_t> payload(16 + 3 * 8);
    
    *reinterpret_cast<uint32_t*>(payload.data()) = 5;
    *reinterpret_cast<uint32_t*>(payload.data() + 4) = 3;
    
    struct Entry { uint32_t idx; float val; };
    Entry entries[3] = {{0, 0.5f}, {10, 0.8f}, {100, 1.2f}};
    std::memcpy(payload.data() + 16, entries, sizeof(entries));
    
    return build_packet(NF_MSG_SPARSE_ACTIVATIONS, NF_FLAG_FP32, 3, 3000000000,
                        payload.data(), static_cast<uint32_t>(payload.size()));
}

inline bool write_fixture_to_file(const std::string& path, const std::vector<uint8_t>& data) {
    std::ofstream file(path, std::ios::binary);
    if (!file) return false;
    file.write(reinterpret_cast<const char*>(data.data()), data.size());
    return true;
}

inline bool read_fixture_from_file(const std::string& path, std::vector<uint8_t>& data) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) return false;
    
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    data.resize(size);
    file.read(reinterpret_cast<char*>(data.data()), size);
    return true;
}

}
}
