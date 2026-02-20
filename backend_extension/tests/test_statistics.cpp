#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <cmath>
#include <vector>
#include "protocol_parser.h"

using namespace nf;
using Catch::Approx;

TEST_CASE("Statistics computation - mean", "[statistics]") {
    
    SECTION("Simple mean calculation") {
        std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        Statistics stats;
        compute_statistics(data, stats);
        
        REQUIRE(stats.mean == Approx(3.0f));
    }
    
    SECTION("Mean of uniform values") {
        std::vector<float> data(100, 5.0f);
        Statistics stats;
        compute_statistics(data, stats);
        
        REQUIRE(stats.mean == Approx(5.0f));
    }
    
    SECTION("Mean of negative values") {
        std::vector<float> data = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
        Statistics stats;
        compute_statistics(data, stats);
        
        REQUIRE(stats.mean == Approx(0.0f));
    }
    
    SECTION("Empty data returns zeros") {
        Statistics stats;
        compute_statistics(static_cast<float*>(nullptr), 0, stats);
        
        REQUIRE(stats.mean == Approx(0.0f));
        REQUIRE(stats.std == Approx(0.0f));
    }
}

TEST_CASE("Statistics computation - std deviation", "[statistics]") {
    
    SECTION("Standard deviation of uniform values") {
        std::vector<float> data(10, 5.0f);
        Statistics stats;
        compute_statistics(data, stats);
        
        REQUIRE(stats.std == Approx(0.0f).margin(0.0001f));
    }
    
    SECTION("Standard deviation calculation") {
        std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        Statistics stats;
        compute_statistics(data, stats);
        
        double expected_std = std::sqrt(2.0);
        REQUIRE(stats.std == Approx(expected_std).margin(0.01f));
    }
    
    SECTION("Standard deviation with negative values") {
        std::vector<float> data = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
        Statistics stats;
        compute_statistics(data, stats);
        
        double expected_std = std::sqrt(2.0);
        REQUIRE(stats.std == Approx(expected_std).margin(0.01f));
    }
}

TEST_CASE("Statistics computation - min and max", "[statistics]") {
    
    SECTION("Min and max extraction") {
        std::vector<float> data = {-5.0f, 0.0f, 3.0f, 10.0f, -2.0f};
        Statistics stats;
        compute_statistics(data, stats);
        
        REQUIRE(stats.min == Approx(-5.0f));
        REQUIRE(stats.max == Approx(10.0f));
    }
    
    SECTION("Single value") {
        std::vector<float> data = {42.0f};
        Statistics stats;
        compute_statistics(data, stats);
        
        REQUIRE(stats.min == Approx(42.0f));
        REQUIRE(stats.max == Approx(42.0f));
    }
}

TEST_CASE("Statistics computation - L2 norm", "[statistics]") {
    
    SECTION("L2 norm calculation") {
        std::vector<float> data = {3.0f, 4.0f};
        Statistics stats;
        compute_statistics(data, stats);
        
        REQUIRE(stats.l2_norm == Approx(5.0f));
    }
    
    SECTION("L2 norm of zeros") {
        std::vector<float> data(5, 0.0f);
        Statistics stats;
        compute_statistics(data, stats);
        
        REQUIRE(stats.l2_norm == Approx(0.0f));
    }
    
    SECTION("L2 norm with negative values") {
        std::vector<float> data = {-3.0f, 4.0f};
        Statistics stats;
        compute_statistics(data, stats);
        
        REQUIRE(stats.l2_norm == Approx(5.0f));
    }
}

TEST_CASE("Statistics computation - zero ratio", "[statistics]") {
    
    SECTION("No zeros") {
        std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        Statistics stats;
        compute_statistics(data, stats);
        
        REQUIRE(stats.zero_ratio == Approx(0.0f));
    }
    
    SECTION("All zeros") {
        std::vector<float> data(10, 0.0f);
        Statistics stats;
        compute_statistics(data, stats);
        
        REQUIRE(stats.zero_ratio == Approx(1.0f));
    }
    
    SECTION("Half zeros") {
        std::vector<float> data = {0.0f, 1.0f, 0.0f, 2.0f, 0.0f, 3.0f, 0.0f, 4.0f};
        Statistics stats;
        compute_statistics(data, stats);
        
        REQUIRE(stats.zero_ratio == Approx(0.5f));
    }
    
    SECTION("Dead neuron detection scenario") {
        std::vector<float> data(100, 0.0f);
        for (int i = 0; i < 30; ++i) {
            data[i] = static_cast<float>(i) * 0.1f;
        }
        
        Statistics stats;
        compute_statistics(data, stats);
        
        REQUIRE(stats.zero_ratio == Approx(0.7f).margin(0.05f));
    }
}

TEST_CASE("Statistics computation - percentiles", "[statistics]") {
    
    SECTION("Percentiles of uniform distribution") {
        std::vector<float> data;
        for (int i = 1; i <= 100; ++i) {
            data.push_back(static_cast<float>(i));
        }
        
        Statistics stats;
        compute_statistics(data, stats);
        
        REQUIRE(stats.p5 == Approx(5.5f).margin(0.5f));
        REQUIRE(stats.p25 == Approx(25.5f).margin(0.5f));
        REQUIRE(stats.p75 == Approx(75.5f).margin(0.5f));
        REQUIRE(stats.p95 == Approx(95.5f).margin(0.5f));
    }
    
    SECTION("Percentiles of single value") {
        std::vector<float> data = {42.0f};
        Statistics stats;
        compute_statistics(data, stats);
        
        REQUIRE(stats.p5 == Approx(42.0f));
        REQUIRE(stats.p25 == Approx(42.0f));
        REQUIRE(stats.p75 == Approx(42.0f));
        REQUIRE(stats.p95 == Approx(42.0f));
    }
    
    SECTION("Percentiles of two values") {
        std::vector<float> data = {0.0f, 100.0f};
        Statistics stats;
        compute_statistics(data, stats);
        
        REQUIRE(stats.p5 == Approx(5.0f).margin(1.0f));
        REQUIRE(stats.p95 == Approx(95.0f).margin(1.0f));
    }
}

TEST_CASE("Statistics computation - kurtosis and skewness", "[statistics]") {
    
    SECTION("Skewness of symmetric distribution") {
        std::vector<float> data = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
        Statistics stats;
        compute_statistics(data, stats);
        
        REQUIRE(std::abs(stats.skewness) < 0.01f);
    }
    
    SECTION("Skewness of right-skewed distribution") {
        std::vector<float> data;
        for (int i = 0; i < 50; ++i) data.push_back(0.0f);
        for (int i = 0; i < 10; ++i) data.push_back(10.0f);
        
        Statistics stats;
        compute_statistics(data, stats);
        
        REQUIRE(stats.skewness > 0.5f);
    }
    
    SECTION("Kurtosis of uniform distribution") {
        std::vector<float> data;
        for (int i = 0; i < 100; ++i) {
            data.push_back(static_cast<float>(i));
        }
        
        Statistics stats;
        compute_statistics(data, stats);
        
        REQUIRE(std::abs(stats.kurtosis) < 1.5f);
    }
    
    SECTION("Zero std results in zero skewness/kurtosis") {
        std::vector<float> data(10, 5.0f);
        Statistics stats;
        compute_statistics(data, stats);
        
        REQUIRE(stats.skewness == Approx(0.0f));
        REQUIRE(stats.kurtosis == Approx(0.0f));
    }
}

TEST_CASE("Statistics computation - pathological patterns", "[statistics]") {
    
    SECTION("Exploding activation detection") {
        std::vector<float> data(1000, 0.001f);
        data[500] = 1000.0f;
        
        Statistics stats;
        compute_statistics(data, stats);
        
        REQUIRE(stats.max > 100.0f * stats.mean);
    }
    
    SECTION("ReLU saturation pattern") {
        std::vector<float> data;
        for (int i = 0; i < 1000; ++i) {
            data.push_back(static_cast<float>(i % 10) * 0.1f);
        }
        
        Statistics stats;
        compute_statistics(data, stats);
        
        REQUIRE(stats.zero_ratio == Approx(0.1f));
        REQUIRE(stats.mean > 0.0f);
    }
    
    SECTION("Dead ReLU pattern") {
        std::vector<float> data(1000, 0.0f);
        
        Statistics stats;
        compute_statistics(data, stats);
        
        REQUIRE(stats.zero_ratio == Approx(1.0f));
        REQUIRE(stats.mean == Approx(0.0f));
        REQUIRE(stats.std == Approx(0.0f));
    }
}

TEST_CASE("Statistics computation - numerical stability", "[statistics]") {
    
    SECTION("Very large values") {
        std::vector<float> data = {1e10f, 2e10f, 3e10f};
        Statistics stats;
        compute_statistics(data, stats);
        
        REQUIRE(std::isfinite(stats.mean));
        REQUIRE(std::isfinite(stats.std));
        REQUIRE(std::isfinite(stats.l2_norm));
    }
    
    SECTION("Very small values") {
        std::vector<float> data = {1e-10f, 2e-10f, 3e-10f};
        Statistics stats;
        compute_statistics(data, stats);
        
        REQUIRE(std::isfinite(stats.mean));
        REQUIRE(std::isfinite(stats.std));
        REQUIRE(std::isfinite(stats.l2_norm));
    }
    
    SECTION("Mixed scale values") {
        std::vector<float> data = {1e-5f, 1.0f, 1e5f};
        Statistics stats;
        compute_statistics(data, stats);
        
        REQUIRE(std::isfinite(stats.mean));
        REQUIRE(std::isfinite(stats.std));
        REQUIRE(std::isfinite(stats.l2_norm));
    }
}
