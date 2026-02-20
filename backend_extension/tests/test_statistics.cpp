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

TEST_CASE("Welford accumulator - basic operations", "[statistics]") {
    
    SECTION("Empty accumulator") {
        WelfordAccumulator acc;
        REQUIRE(acc.count() == 0);
        REQUIRE(acc.mean() == Approx(0.0));
    }
    
    SECTION("Single value") {
        WelfordAccumulator acc;
        acc.add(5.0f);
        
        REQUIRE(acc.count() == 1);
        REQUIRE(acc.mean() == Approx(5.0));
        
        Statistics stats = acc.get_statistics();
        REQUIRE(stats.mean == Approx(5.0f));
        REQUIRE(stats.min == Approx(5.0f));
        REQUIRE(stats.max == Approx(5.0f));
    }
    
    SECTION("Multiple values") {
        WelfordAccumulator acc;
        for (int i = 1; i <= 5; ++i) {
            acc.add(static_cast<float>(i));
        }
        
        REQUIRE(acc.count() == 5);
        REQUIRE(acc.mean() == Approx(3.0));
        
        Statistics stats = acc.get_statistics();
        REQUIRE(stats.mean == Approx(3.0f));
        REQUIRE(stats.min == Approx(1.0f));
        REQUIRE(stats.max == Approx(5.0f));
        REQUIRE(stats.std > 0);
    }
    
    SECTION("Add array of values") {
        std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        
        WelfordAccumulator acc;
        acc.add(data.data(), data.size());
        
        REQUIRE(acc.count() == 5);
        REQUIRE(acc.mean() == Approx(3.0));
    }
}

TEST_CASE("Welford accumulator - matches compute_statistics", "[statistics]") {
    
    SECTION("Uniform distribution") {
        std::vector<float> data;
        for (int i = 1; i <= 100; ++i) {
            data.push_back(static_cast<float>(i));
        }
        
        Statistics direct;
        compute_statistics(data, direct);
        
        WelfordAccumulator acc;
        acc.add(data.data(), data.size());
        Statistics accumulated = acc.get_statistics();
        
        REQUIRE(accumulated.mean == Approx(direct.mean));
        REQUIRE(accumulated.std == Approx(direct.std).margin(0.001f));
        REQUIRE(accumulated.min == Approx(direct.min));
        REQUIRE(accumulated.max == Approx(direct.max));
        REQUIRE(accumulated.l2_norm == Approx(direct.l2_norm));
    }
    
    SECTION("Mixed values") {
        std::vector<float> data = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        
        Statistics direct;
        compute_statistics(data, direct);
        
        WelfordAccumulator acc;
        for (float v : data) {
            acc.add(v);
        }
        Statistics accumulated = acc.get_statistics();
        
        REQUIRE(accumulated.mean == Approx(direct.mean));
        REQUIRE(accumulated.std == Approx(direct.std).margin(0.001f));
        REQUIRE(accumulated.zero_ratio == Approx(direct.zero_ratio));
    }
}

TEST_CASE("Welford accumulator - reset", "[statistics]") {
    
    SECTION("Reset clears all state") {
        WelfordAccumulator acc;
        acc.add(1.0f);
        acc.add(2.0f);
        acc.add(3.0f);
        
        REQUIRE(acc.count() == 3);
        
        acc.reset();
        
        REQUIRE(acc.count() == 0);
        REQUIRE(acc.mean() == Approx(0.0));
        
        Statistics stats = acc.get_statistics();
        REQUIRE(stats.mean == Approx(0.0f));
    }
    
    SECTION("Reuse after reset") {
        WelfordAccumulator acc;
        
        acc.add(100.0f);
        acc.add(200.0f);
        acc.reset();
        
        acc.add(1.0f);
        acc.add(2.0f);
        acc.add(3.0f);
        
        REQUIRE(acc.count() == 3);
        REQUIRE(acc.mean() == Approx(2.0));
    }
}

TEST_CASE("Welford accumulator - incremental updates", "[statistics]") {
    
    SECTION("Incremental mean accuracy") {
        WelfordAccumulator acc;
        
        acc.add(10.0f);
        REQUIRE(acc.mean() == Approx(10.0));
        
        acc.add(20.0f);
        REQUIRE(acc.mean() == Approx(15.0));
        
        acc.add(30.0f);
        REQUIRE(acc.mean() == Approx(20.0));
        
        acc.add(40.0f);
        REQUIRE(acc.mean() == Approx(25.0));
    }
    
    SECTION("Numerical stability with many small updates") {
        WelfordAccumulator acc;
        
        for (int i = 0; i < 10000; ++i) {
            acc.add(0.001f);
        }
        
        Statistics stats = acc.get_statistics();
        REQUIRE(std::isfinite(stats.mean));
        REQUIRE(std::isfinite(stats.std));
        REQUIRE(stats.mean == Approx(0.001f));
    }
}

TEST_CASE("Welford accumulator - zero ratio tracking", "[statistics]") {
    
    SECTION("Zero ratio computed correctly") {
        WelfordAccumulator acc;
        
        for (int i = 0; i < 10; ++i) acc.add(0.0f);
        for (int i = 0; i < 10; ++i) acc.add(1.0f);
        
        Statistics stats = acc.get_statistics();
        REQUIRE(stats.zero_ratio == Approx(0.5f));
    }
    
    SECTION("All zeros") {
        WelfordAccumulator acc;
        for (int i = 0; i < 100; ++i) acc.add(0.0f);
        
        Statistics stats = acc.get_statistics();
        REQUIRE(stats.zero_ratio == Approx(1.0f));
    }
    
    SECTION("No zeros") {
        WelfordAccumulator acc;
        for (int i = 1; i <= 100; ++i) acc.add(static_cast<float>(i));
        
        Statistics stats = acc.get_statistics();
        REQUIRE(stats.zero_ratio == Approx(0.0f));
    }
}
