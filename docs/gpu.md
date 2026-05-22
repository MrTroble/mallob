# GPU support

To use GPU accelerated sat solving, the mallob gpu api (MGI) needs to be enabled.

## Prerequisits

> [!WARNING]  
> Currently MGI only supports **OpenCL 3.0** as backend

### NVIDIA

You need the newest [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) from NVIDIA

> [!INFO]
> This is currently tested on *Fedora 43* and *Debian 13*

### AMD

TODO

> [!WARNING]  
> AMD Support is currently untested

### Intel

The Intel [Graphics Compute Runtime](https://github.com/intel/compute-runtime) is needed for OICDs.
Hardware support and installation instructions can be found on their Github page.

> [!WARNING]  
> Intel Support is currently untested

### WSL

TODO -> Try

## Usage

In order to compile mallob with GPU features the support must be enabled through CMake.

### Using presets

We provide a `CMakePresets.json` with GPU [configure presets](https://cmake.org/cmake/help/latest/manual/cmake-presets.7.html) for ease of use.
The `linux-debug-gpu` (Nativ Linux GPU Debug) preset is currently the main gpu preset.
You can either enable this your IDE/CMake Extension or manually use it in command line.
> cmake --preset linux-debug-gpu

### Manually

Otherwise the `MALLOB_USE_GPU` cmake flag can be set.

> cmake ./ -DMALLOB_USE_GPU=1

Additionally if you want more output consider using the `MALLOB_ASSERT` flag which enables debug output for the API layer.
Provided the `MALLOB_LOG_VERBOSITY` is set correctly (>=debug).

## Flags

|Flag|Usage|Default|
|-|-|-|
|`MALLOB_MGI_OCL_LAYERS`| Automatically download and add a compile target for the OpenCL validation layers. Needs environment configuration. This is automatically configured if the preset is used | On
|`MALLOB_MGI_NO_TEST_COMPILE`| Disable clang OpenCL-C++ to SPIR-V compiler that automatically run to check your kernels for errors. | Off

## Development

TODO -> From Notes