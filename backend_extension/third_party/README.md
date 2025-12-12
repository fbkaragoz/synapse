# Vendored Dependencies (optional)

The backend extension uses CMake. By default it uses `FetchContent` (git clone) to download dependencies:

- `pybind11`
- `uWebSockets` (header-only)
- `uSockets` (C library)

If you are offline (or behind restricted DNS), builds will fail when CMake tries to clone from GitHub.

## Option A: Vendor deps in this folder (recommended for reproducible builds)

Place checkouts here:

- `backend_extension/third_party/pybind11/`
- `backend_extension/third_party/uWebSockets/`
- `backend_extension/third_party/uSockets/`

Then configure with:

`cmake -S backend_extension -B backend_extension/build -DNF_USE_FETCHCONTENT=OFF`

## Option B: Use system pybind11

Install pybind11 via your OS package manager and configure with:

`cmake -S backend_extension -B backend_extension/build -DNF_USE_SYSTEM_PYBIND11=ON`

`uWebSockets` and `uSockets` are still required (vendor them or set paths).

## Option C: Explicit paths

You can point CMake at local checkouts:

- `-DNF_USOCKETS_DIR=/path/to/uSockets`
- `-DNF_UWEBSOCKETS_DIR=/path/to/uWebSockets`

