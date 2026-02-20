cmake_minimum_required(VERSION 3.14)
project(neural_probe LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Build options
option(NF_BUILD_TESTS "Build unit tests" ON)
option(NF_USE_FETCHCONTENT "Allow downloading dependencies (git clone)" ON)
option(NF_USE_SYSTEM_PYBIND11 "Use system-installed pybind11 (find_package)" OFF)

set(NF_THIRD_PARTY_DIR "${CMAKE_CURRENT_LIST_DIR}/third_party")


# Fetch Dependencies
include(FetchContent)

#
# pybind11
#
if(NF_USE_SYSTEM_PYBIND11)
  find_package(pybind11 CONFIG REQUIRED)
elseif(EXISTS "${NF_THIRD_PARTY_DIR}/pybind11/CMakeLists.txt")
  add_subdirectory("${NF_THIRD_PARTY_DIR}/pybind11" "${CMAKE_BINARY_DIR}/pybind11")
elseif(NF_USE_FETCHCONTENT)
  FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11
    GIT_TAG        v2.10.0
  )
  FetchContent_MakeAvailable(pybind11)
else()
  message(FATAL_ERROR "pybind11 not found. Vendor it under backend_extension/third_party/pybind11 or set NF_USE_SYSTEM_PYBIND11=ON.")
endif()

# uSockets used by uWebSockets
set(NF_USOCKETS_DIR "" CACHE PATH "Path to vendored uSockets checkout (optional)")
if(NF_USOCKETS_DIR STREQUAL "" AND EXISTS "${NF_THIRD_PARTY_DIR}/uSockets/src")
  set(NF_USOCKETS_DIR "${NF_THIRD_PARTY_DIR}/uSockets")
endif()

# uSockets CMake is sometimes tricky. We might need to manually add source if it doesn't support CMake nicely as a subdir.
# Actually uWebSockets has a CMakeLists that usually handles uSockets? 
# No, uWebSockets is header only but depends on uSockets which is C lib.
# Let's try fetching uSockets and building it. 
# NOTE: uSockets repo often doesn't have a CMakeLists.txt. 
# If it fails, we will use a known working fork or manually define the library.
# For this environment, we assume standard uSockets repo structure. 
# Update: uSockets v0.8.5 has a Makefile mostly.
# We will define a target for uSockets manually to be safe.

if(NF_USOCKETS_DIR STREQUAL "")
  if(NF_USE_FETCHCONTENT)
    FetchContent_Declare(
      usockets
      GIT_REPOSITORY https://github.com/uNetworking/uSockets
      GIT_TAG        master
    )
    FetchContent_GetProperties(usockets)
    if(NOT usockets_POPULATED)
      FetchContent_Populate(usockets)
    endif()
    set(NF_USOCKETS_DIR "${usockets_SOURCE_DIR}")
  else()
    message(FATAL_ERROR "uSockets not found. Vendor it under backend_extension/third_party/uSockets or set NF_USOCKETS_DIR.")
  endif()
endif()

# Manual target for uSockets (repo doesn't reliably ship CMakeLists)
file(GLOB USOCKETS_SOURCES
  "${NF_USOCKETS_DIR}/src/*.c"
  "${NF_USOCKETS_DIR}/src/eventing/*.c"
  "${NF_USOCKETS_DIR}/src/crypto/*.c"
)
add_library(uSockets STATIC ${USOCKETS_SOURCES})
target_include_directories(uSockets PUBLIC "${NF_USOCKETS_DIR}/src")
target_compile_definitions(uSockets PRIVATE LIBUS_NO_SSL) # Disable SSL for simplicity


# uWebSockets (Header Only)
set(NF_UWEBSOCKETS_DIR "" CACHE PATH "Path to vendored uWebSockets checkout (optional)")
if(NF_UWEBSOCKETS_DIR STREQUAL "" AND EXISTS "${NF_THIRD_PARTY_DIR}/uWebSockets/src/App.h")
  set(NF_UWEBSOCKETS_DIR "${NF_THIRD_PARTY_DIR}/uWebSockets")
endif()

if(NF_UWEBSOCKETS_DIR STREQUAL "")
  if(NF_USE_FETCHCONTENT)
    FetchContent_Declare(
      uwebsockets
      GIT_REPOSITORY https://github.com/uNetworking/uWebSockets
      GIT_TAG        v20.48.0
    )
    FetchContent_MakeAvailable(uwebsockets)
    set(NF_UWEBSOCKETS_DIR "${uwebsockets_SOURCE_DIR}")
  else()
    message(FATAL_ERROR "uWebSockets not found. Vendor it under backend_extension/third_party/uWebSockets or set NF_UWEBSOCKETS_DIR.")
  endif()
endif()

# The Extension
pybind11_add_module(neural_probe 
    src/main.cpp
    src/neural_probe.cpp
    src/server_uwebsockets.cpp
)

target_include_directories(neural_probe PRIVATE 
    include
    ${NF_UWEBSOCKETS_DIR}/src
    ${NF_USOCKETS_DIR}/src
)

target_link_libraries(neural_probe PRIVATE uSockets)

# Fix for uSockets C++ linking if needed
if(UNIX)
    target_link_libraries(neural_probe PRIVATE pthread)
endif()

# =============================================================================
# Testing with Catch2
# =============================================================================
if(NF_BUILD_TESTS)
    # Catch2
    set(NF_CATCH2_DIR "" CACHE PATH "Path to vendored Catch2 checkout (optional)")
    if(NF_CATCH2_DIR STREQUAL "" AND EXISTS "${NF_THIRD_PARTY_DIR}/Catch2/CMakeLists.txt")
        set(NF_CATCH2_DIR "${NF_THIRD_PARTY_DIR}/Catch2")
    endif()
    
    if(NF_CATCH2_DIR STREQUAL "")
        if(NF_USE_FETCHCONTENT)
            FetchContent_Declare(
                Catch2
                GIT_REPOSITORY https://github.com/catchorg/Catch2
                GIT_TAG        v3.5.0
            )
            FetchContent_MakeAvailable(Catch2)
        else()
            message(FATAL_ERROR "Catch2 not found. Vendor it under backend_extension/third_party/Catch2 or set NF_CATCH2_DIR.")
        endif()
    else()
        add_subdirectory("${NF_CATCH2_DIR}" "${CMAKE_BINARY_DIR}/Catch2")
    endif()
    
    # Test executable
    add_executable(neural_probe_tests
        tests/test_main.cpp
        tests/test_protocol.cpp
        tests/test_ring_buffer.cpp
        tests/test_statistics.cpp
    )
    
    target_include_directories(neural_probe_tests PRIVATE
        include
    )
    
    target_link_libraries(neural_probe_tests PRIVATE
        Catch2::Catch2WithMain
        pthread
    )
    
    # Discover tests for CTest
    include(CTest)
    include(Catch)
    catch_discover_tests(neural_probe_tests)
endif()
