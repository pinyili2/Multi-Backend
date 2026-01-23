cmake_minimum_required(VERSION 3.25.5 FATAL_ERROR)
set(PROJECT_NAME_BASE "MultiBack")
if(DEFINED PROJECT_NAME_SUFFIX)
    set(PROJECT_NAME "${PROJECT_NAME_BASE}${PROJECT_NAME_SUFFIX}")
else()
    set(PROJECT_NAME "${PROJECT_NAME_BASE}")
endif()

# Single project() call
if(USE_CUDA)
    project("${PROJECT_NAME}" VERSION 2.0 LANGUAGES CXX CUDA)
else()
    project("${PROJECT_NAME}" VERSION 2.0 LANGUAGES CXX)
endif()


# --- Project Options ---
option(USE_CUDA "Use CUDA" OFF)
option(USE_SYCL_ACPP "Use SYCL with AdaptiveCpp (acpp)" OFF)
option(USE_SYCL_ICPX "Use SYCL with Intel DPC++ (icpx)" OFF)
option(DEBUG_BUILD "Build with debug flags (overrides CMAKE_BUILD_TYPE)" OFF)
option(USE_NCCL "Use NCCL for single node GPU peer communication" OFF)
option(USE_NVTX "Build with NVTX profiling ranges" OFF)
option(USE_PYBIND "Build Python bindings" OFF)
option(USE_METAL "Use Apple Silicon Metal" OFF)
option(USE_MPI "Use MPI for distributed computing" OFF)

# --- Backend Selection Validation ---
# Ensure only one backend is selected
set(BACKEND_COUNT 0)
if(USE_CUDA)
    math(EXPR BACKEND_COUNT "${BACKEND_COUNT} + 1")
endif()
if(USE_METAL)
    math(EXPR BACKEND_COUNT "${BACKEND_COUNT} + 1")
endif()
if(USE_SYCL_ACPP OR USE_SYCL_ICPX)
    math(EXPR BACKEND_COUNT "${BACKEND_COUNT} + 1")
endif()

if(BACKEND_COUNT EQUAL 0)
    message(FATAL_ERROR "No backend selected. Please enable exactly one of: USE_CUDA, USE_METAL, USE_SYCL_ACPP, or USE_SYCL_ICPX")
elseif(BACKEND_COUNT GREATER 1)
    message(FATAL_ERROR "Multiple backends selected. Please enable exactly one of: USE_CUDA, USE_METAL, USE_SYCL_ACPP, or USE_SYCL_ICPX")
endif()

# Validate SYCL options (only one SYCL variant should be selected)
if(USE_SYCL_ACPP AND USE_SYCL_ICPX)
    message(FATAL_ERROR "Both USE_SYCL_ACPP and USE_SYCL_ICPX are ON. Please choose only one SYCL implementation.")
endif()

# --- Basic CMake Setup ---
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Define C++23 macros as empty to avoid compatibility issues
add_compile_definitions(_GLIBCXX23_CONSTEXPR=)

# Generate compile_commands.json for clangd and other tools
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find threading library for use across the project
find_package(Threads REQUIRED)
include_directories(${CMAKE_SOURCE_DIR}/extern)


if(USE_CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_EXTENSIONS OFF)
    set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89;90") #100, 120 are not supported by all GPUs and compilers
    # Add CUDA compiler flags for compatibility and force older GCC
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --allow-unsupported-compiler")
    message(STATUS "CUDA architectures set to: ${CMAKE_CUDA_ARCHITECTURES}")
    message(STATUS "CUDA flags: ${CMAKE_CUDA_FLAGS}")
endif()

# Set a default build type if none was specified
set(default_build_type "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
    STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "RelWithDebInfo" "MinSizeRel")
endif()

if(NOT DEFINED OPT_LEVEL)
    set(OPT_LEVEL 3)
endif()

message(STATUS "Setting optimization level to -O${OPT_LEVEL}")

# Configure optimization flags for different compilers
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    # GCC
    set(CMAKE_CXX_FLAGS_RELEASE "-O${OPT_LEVEL} -DNDEBUG")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O${OPT_LEVEL} -g -DNDEBUG")

    # Add additional GCC-specific optimizations
    if(OPT_LEVEL GREATER_EQUAL 3)
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native -mtune=native")
        set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -march=native -mtune=native")
    endif()

    if(USE_CUDA AND CMAKE_CUDA_FLAGS_RELEASE)
        set(CMAKE_CUDA_FLAGS_RELEASE "-O${OPT_LEVEL} -DNDEBUG")
        set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "-O${OPT_LEVEL} -g -DNDEBUG")
    endif()

elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    # Clang/AppleClang
    set(CMAKE_CXX_FLAGS_RELEASE "-O${OPT_LEVEL} -DNDEBUG")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O${OPT_LEVEL} -g -DNDEBUG")

    # Add additional Clang-specific optimizations
    if(OPT_LEVEL GREATER_EQUAL 3)
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native -mtune=native")
        set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -march=native -mtune=native")
    endif()

    if(USE_CUDA AND CMAKE_CUDA_FLAGS_RELEASE)
        set(CMAKE_CUDA_FLAGS_RELEASE "-O${OPT_LEVEL} -DNDEBUG")
        set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "-O${OPT_LEVEL} -g -DNDEBUG")
    endif()

elseif(CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM" OR CMAKE_CXX_COMPILER_ID MATCHES "Intel")
    # Intel compilers (icpx, icc)
    set(CMAKE_CXX_FLAGS_RELEASE "-O${OPT_LEVEL} -DNDEBUG")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O${OPT_LEVEL} -g -DNDEBUG")

    # Add Intel-specific optimizations
    if(OPT_LEVEL GREATER_EQUAL 3)
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -xHost")
        set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -xHost")
    endif()

elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # Microsoft Visual C++
    set(CMAKE_CXX_FLAGS_RELEASE "/O${OPT_LEVEL} /DNDEBUG")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "/O${OPT_LEVEL} /Zi /DNDEBUG")

else()
    # Generic fallback
    set(CMAKE_CXX_FLAGS_RELEASE "-O${OPT_LEVEL} -DNDEBUG")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O${OPT_LEVEL} -g -DNDEBUG")
endif()

# Special handling for SYCL compilers (acpp, icpx)
if(USE_SYCL_ACPP OR USE_SYCL_ICPX)
    # SYCL compilers need optimization flags to be explicitly set
    message(STATUS "SYCL detected: Ensuring optimization flags are properly set")
    set(CMAKE_CXX_FLAGS_RELEASE "-O${OPT_LEVEL} -DNDEBUG")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O${OPT_LEVEL} -g -DNDEBUG")

    # For ACPP specifically, ensure optimization flags are added to general flags too
    if(USE_SYCL_ACPP)
        # Add optimization flags to general CXX flags to ensure they're always present
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O${OPT_LEVEL}")
        message(STATUS "ACPP: Added -O${OPT_LEVEL} to CMAKE_CXX_FLAGS")
    endif()
endif()


# Ensure optimization flags are applied globally (including to external dependencies)
if(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    # Add optimization flags to the general CMAKE_CXX_FLAGS to ensure they're always present
    string(FIND "${CMAKE_CXX_FLAGS}" "-O" OPT_FLAG_FOUND)
    if(OPT_FLAG_FOUND EQUAL -1)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O${OPT_LEVEL}")
        message(STATUS "Added -O${OPT_LEVEL} to CMAKE_CXX_FLAGS for global optimization")
    endif()
endif()

message(STATUS "C++ Release flags: ${CMAKE_CXX_FLAGS_RELEASE}")
message(STATUS "C++ General flags: ${CMAKE_CXX_FLAGS}")
if(USE_CUDA)
    message(STATUS "CUDA Release flags: ${CMAKE_CUDA_FLAGS_RELEASE}")
endif()

# Metal specific setup
if(USE_METAL AND APPLE)
    # Enable Objective-C++ for .mm files
    set(CMAKE_OBJCXX_STANDARD 20)
    set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)
    set(CMAKE_OBJCXX_EXTENSIONS OFF)

    # Enable ARC (Automatic Reference Counting) for Objective-C++
    set(CMAKE_OBJCXX_FLAGS "${CMAKE_OBJCXX_FLAGS} -fobjc-arc")
endif()

# --- SYCL Configuration (Explicit ACPP or ICPX) ---

if(USE_SYCL_ACPP OR USE_SYCL_ICPX)
    message(STATUS "SYCL support is requested.")
    add_definitions(-DUSE_SYCL) # Generic definition for SYCL API usage in C++

    if(NOT CMAKE_CXX_COMPILER)
        message(FATAL_ERROR "A SYCL option (USE_SYCL_ACPP or USE_SYCL_ICPX) is ON, but CMAKE_CXX_COMPILER is not set.
Please specify your SYCL compiler via -D CMAKE_CXX_COMPILER=<path_to_sycl_compiler>
Example for USE_SYCL_ICPX on Linux:      cmake -D CMAKE_CXX_COMPILER=icpx -D USE_SYCL_ICPX=ON ..
Example for USE_SYCL_ACPP on macOS:    cmake -D CMAKE_CXX_COMPILER=/opt/homebrew/bin/acpp -D USE_SYCL_ACPP=ON ..")
    endif()

    set(SYCL_COMPILER_CONFIGURED_CORRECTLY FALSE) # Flag to check if config matches expectation

    if(USE_SYCL_ACPP)
        message(STATUS "Configuring for SYCL with AdaptiveCpp (acpp).")
        add_definitions(-DPROJECT_USES_SYCL_ACPP) # Specific definition for acpp

        if(CMAKE_CXX_COMPILER MATCHES "acpp")
            message(STATUS "Verified AdaptiveCpp (acpp) compiler: ${CMAKE_CXX_COMPILER}")

            # Add rpath to acpp libraries to solve linking issues
            get_filename_component(ACPP_COMPILER_PATH ${CMAKE_CXX_COMPILER} DIRECTORY)
            get_filename_component(ACPP_INSTALL_PREFIX ${ACPP_COMPILER_PATH}/.. REALPATH)
            set(ACPP_LIB_DIR ${ACPP_INSTALL_PREFIX}/lib)
            if(EXISTS ${ACPP_LIB_DIR})
                message(STATUS "Found AdaptiveCpp library path: ${ACPP_LIB_DIR}. Adding to RPATH.")
                set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath,${ACPP_LIB_DIR}")
            else()
                message(WARNING "Could not find AdaptiveCpp library path at ${ACPP_LIB_DIR}. Linking may fail.")
            endif()

            set(ACPP_TARGET_FLAGS "-O2")
            if(CMAKE_SYSTEM_NAME STREQUAL "Darwin") # macOS
                message(STATUS "Setting AdaptiveCpp default target flags for macOS (CPU via OpenMP).")
                set(ACPP_TARGET_FLAGS "--acpp-targets=omp")
            elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
                message(STATUS "Setting AdaptiveCpp default target flags for Linux (CUDA and OpenMP ).")
                set(ACPP_TARGET_FLAGS "--acpp-targets=omp") #--acpp-targets=cuda:sm_80

            else() # Other OS
                message(WARNING "AdaptiveCpp on unknown OS ${CMAKE_SYSTEM_NAME}. Defaulting to OpenMP target.")
                set(ACPP_TARGET_FLAGS "--acpp-targets=omp")
            endif()

            if(NOT ACPP_TARGET_FLAGS STREQUAL "")
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ACPP_TARGET_FLAGS}")
                message(STATUS "Applied AdaptiveCpp flags: ${ACPP_TARGET_FLAGS}")
            endif()
            set(SYCL_COMPILER_CONFIGURED_CORRECTLY TRUE)
        else()
            message(WARNING "USE_SYCL_ACPP is ON, but CMAKE_CXX_COMPILER (${CMAKE_CXX_COMPILER}) does not appear to be 'acpp'.
Ensure CMAKE_CXX_COMPILER is set to your AdaptiveCpp executable path (e.g., /opt/homebrew/bin/acpp for macOS).")
        endif()
    endif()

    if(USE_SYCL_ICPX)
        message(STATUS "Configuring for SYCL with Intel DPC++ (icpx).")
        add_definitions(-DPROJECT_USES_SYCL_ICPX)

        if(CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM" OR CMAKE_CXX_COMPILER MATCHES "icpx")
            message(STATUS "Verified Intel DPC++ (icpx) compiler: ${CMAKE_CXX_COMPILER}")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-c99-designator")

            if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
                # ----------------------------------------------------------------
                # 1. Attempt to detect AMD GPUs (Frontier/Delta)
                # ----------------------------------------------------------------
                find_program(ROCM_SMI rocm-smi)
                find_program(NVIDIA_SMI nvidia-smi)

                if(ROCM_SMI)
                    message(STATUS "Found rocm-smi: Configuring SYCL for AMD backend")

                    # Target AMD GCN architecture
                    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -fsycl-targets=amdgcn-amd-amdhsa -O3")

                    # CRITICAL: Set Sub-Group Size to 64 for AMD
                    add_compile_definitions(SG_SIZE=64)

                    # Attempt to detect architecture (gfx90a = Frontier, gfx908 = Delta)
                    # You can override this by passing -DAMD_ARCH=gfx90a to cmake
                    if(NOT DEFINED AMD_ARCH)
                        execute_process(COMMAND ${ROCM_SMI} --showproductname --json
                                      OUTPUT_VARIABLE AMD_GPU_INFO
                                      ERROR_QUIET)
                        if(AMD_GPU_INFO MATCHES "MI200" OR AMD_GPU_INFO MATCHES "MI250")
                            set(AMD_ARCH "gfx90a") # Frontier
                        elseif(AMD_GPU_INFO MATCHES "MI100")
                            set(AMD_ARCH "gfx908") # Delta
                        else()
                            set(AMD_ARCH "gfx90a") # Default to newer (Frontier) if unknown
                            message(WARNING "Could not auto-detect AMD Arch. Defaulting to ${AMD_ARCH}. Use -DAMD_ARCH=gfx... to override.")
                        endif()
                    endif()

                    message(STATUS "Targeting AMD Architecture: ${AMD_ARCH}")
                    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xsycl-target-backend --offload-arch=${AMD_ARCH}")

                # ----------------------------------------------------------------
                # 2. Attempt to detect NVIDIA GPUs
                # ----------------------------------------------------------------
                elseif(NVIDIA_SMI)
                    execute_process(COMMAND ${NVIDIA_SMI} --query-gpu=count --format=csv,noheader,nounits
                                  OUTPUT_VARIABLE GPU_COUNT
                                  OUTPUT_STRIP_TRAILING_WHITESPACE
                                  ERROR_QUIET)

                    if(GPU_COUNT AND GPU_COUNT GREATER 0)
                        message(STATUS "Found ${GPU_COUNT} NVIDIA GPU(s), configuring SYCL for CUDA backend")

                        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -fsycl-targets=nvptx64-nvidia-cuda -O3")
                        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_80")

                        # CRITICAL: Set Sub-Group Size to 32 for NVIDIA
                        add_compile_definitions(SG_SIZE=32)
                    else()
                        # Fallback for empty NVIDIA node
                        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -O3")
                        add_compile_definitions(SG_SIZE=32)
                    endif()

                # ----------------------------------------------------------------
                # 3. Fallback (CPU / Intel GPU)
                # ----------------------------------------------------------------
                else()
                    message(STATUS "No dedicated GPU tools (nvidia-smi/rocm-smi) found. Using default SYCL targets.")
                    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -O3")
                    # Default to 32 (Safe for Intel GPUs/CPU)
                    add_compile_definitions(SG_SIZE=32)
                endif()
            else()
                # Non-Linux
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -O3")
                add_compile_definitions(SG_SIZE=32)
            endif()

            set(SYCL_COMPILER_CONFIGURED_CORRECTLY TRUE)
        else()
            message(WARNING "USE_SYCL_ICPX is ON, but compiler is not icpx.")
        endif()
    endif()

    if(NOT SYCL_COMPILER_CONFIGURED_CORRECTLY) # If neither acpp nor icpx matched the selected option
         message(WARNING "A SYCL option was enabled, but the specified CMAKE_CXX_COMPILER does not match the expected SYCL compiler for that option.
Please review your -D CMAKE_CXX_COMPILER=... setting and the USE_SYCL_ACPP / USE_SYCL_ICPX CMake options.
Compilation will proceed, but may not use SYCL correctly.")
    endif()

endif()
# --- End SYCL Configuration ---

# --- Metal Configuration ---
if(USE_METAL)
    if(NOT APPLE)
        message(FATAL_ERROR "Metal is only available on Apple platforms")
    endif()

    message(STATUS "Metal support is enabled for Apple platforms.")
    add_definitions(-DUSE_METAL)

    # Add metal-cmake subdirectory
    add_subdirectory(${CMAKE_SOURCE_DIR}/extern/metal-cpp-cmake/metal-cmake EXCLUDE_FROM_ALL)

    # Set up include paths for metal-cpp headers
    include_directories(
        ${CMAKE_SOURCE_DIR}/extern/metal-cpp-cmake/metal-cmake/metal-cpp
        ${CMAKE_SOURCE_DIR}/extern/metal-cpp-cmake/metal-cmake/metal-cpp-extensions
    )

    # Find Metal framework
    find_library(METAL_FRAMEWORK Metal REQUIRED)
    find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
    find_library(QUARTZCORE_FRAMEWORK QuartzCore REQUIRED)

    if(METAL_FRAMEWORK AND FOUNDATION_FRAMEWORK AND QUARTZCORE_FRAMEWORK)
        message(STATUS "Found Metal framework: ${METAL_FRAMEWORK}")
        message(STATUS "Found Foundation framework: ${FOUNDATION_FRAMEWORK}")
        message(STATUS "Found QuartzCore framework: ${QUARTZCORE_FRAMEWORK}")
    else()
        message(FATAL_ERROR "Required frameworks not found")
    endif()
    set(ARBD_METAL_FRAMEWORK_PATH "" CACHE PATH "Path to the macOS SDK Frameworks directory (e.g., /Applications/Xcode.app/...)")

    # If the user hasn't provided a path, set a default to the standard location.
    if(NOT ARBD_METAL_FRAMEWORK_PATH)
        set(ARBD_METAL_FRAMEWORK_PATH "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/System/Library/Frameworks")
    endif()

    # Verify that the specified path is valid. If not, stop the build with a helpful error.
    if(NOT EXISTS "${ARBD_METAL_FRAMEWORK_PATH}")
        message(FATAL_ERROR "========================================================================\n"
                            "macOS SDK Frameworks not found at the specified path:\n"
                            "  ${ARBD_METAL_FRAMEWORK_PATH}\n\n"
                            "If you are using a non-standard Xcode installation (e.g., Xcode-beta.app), "
                            "please specify the correct path by setting the ARBD_METAL_FRAMEWORK_PATH CMake variable.\n\n"
                            "Example:\n"
                            "  cmake -DARBD_METAL_FRAMEWORK_PATH=/Applications/Xcode-beta.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/System/Library/Frameworks ...\n"
                            "========================================================================")
    endif()

    # Add the framework path to the compiler flags for all targets.
    add_compile_options("-F${ARBD_METAL_FRAMEWORK_PATH}")

    # Add framework path for Metal.framework
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -F${ARBD_METAL_FRAMEWORK_PATH}")
    #set(CMAKE_OBJCXX_FLAGS "${CMAKE_OBJCXX_FLAGS} -F${ARBD_METAL_FRAMEWORK_PATH}")
endif()
# --- End Metal Configuration ---

# --- Metal Shader Compilation ---
if(USE_METAL)
    # Set up Metal shader compilation at project level
    set(METAL_OUTPUT_DIR ${CMAKE_BINARY_DIR})

    # Find all Metal source files in the src directory
    file(GLOB_RECURSE METAL_SOURCES
        ${CMAKE_SOURCE_DIR}/src/*.metal
        ${CMAKE_SOURCE_DIR}/src/Random/*.metal
    )

    # Also include test Metal files to ensure all kernels are available
    file(GLOB_RECURSE TEST_METAL_SOURCES
        ${CMAKE_SOURCE_DIR}/Tests/*.metal
        ${CMAKE_SOURCE_DIR}/Tests/Kernels/*.metal
        ${CMAKE_SOURCE_DIR}/Tests/Math/*.metal
    )

    # Combine all Metal sources
    list(APPEND METAL_SOURCES ${TEST_METAL_SOURCES})

    if(METAL_SOURCES)
        message(STATUS "Found Metal shader sources: ${METAL_SOURCES}")

        # Create a list to hold the compiled .air files
        set(METAL_AIR_FILES)

        # Compile each .metal file to .air
        foreach(METAL_SOURCE ${METAL_SOURCES})
            get_filename_component(METAL_NAME ${METAL_SOURCE} NAME_WE)
            set(AIR_FILE ${METAL_OUTPUT_DIR}/${METAL_NAME}.air)

            add_custom_command(
                OUTPUT ${AIR_FILE}
                COMMAND xcrun -sdk macosx metal -c ${METAL_SOURCE} -o ${AIR_FILE} -I${CMAKE_SOURCE_DIR}/src -gline-tables-only -frecord-sources
                DEPENDS ${METAL_SOURCE}
                COMMENT "Compiling Metal shader: ${METAL_NAME}.metal"
                VERBATIM
            )

            list(APPEND METAL_AIR_FILES ${AIR_FILE})
        endforeach()

        # Create the Metal library for the main project
        set(METAL_LIBRARY ${METAL_OUTPUT_DIR}/default.metallib)
        add_custom_command(
            OUTPUT ${METAL_LIBRARY}
            COMMAND xcrun -sdk macosx metallib ${METAL_AIR_FILES} -o ${METAL_LIBRARY}
            DEPENDS ${METAL_SOURCES} ${METAL_AIR_FILES}
            COMMENT "Creating Metal library: default.metallib"
            VERBATIM
        )

        # Create a custom target for Metal shaders
        add_custom_target(metal_shaders_main ALL DEPENDS ${METAL_LIBRARY})

        message(STATUS "Metal shader compilation configured for main project")
    else()
        message(STATUS "No Metal shader sources found in src directory")
    endif()
endif()
# --- End Metal Shader Compilation ---

# --- MPI Configuration ---
if(USE_MPI)
    message(STATUS "MPI support is enabled.")


    # Set MPI hints from cache variables if provided (from CMakePresets.json)
    if(DEFINED MPI_CXX_COMPILER)
        set(MPI_CXX_COMPILER "${MPI_CXX_COMPILER}" CACHE FILEPATH "MPI CXX compiler")
    endif()
    if(DEFINED MPI_C_COMPILER)
        set(MPI_C_COMPILER "${MPI_C_COMPILER}" CACHE FILEPATH "MPI C compiler")
    endif()
    if(DEFINED MPI_CXX_INCLUDE_PATH)
        set(MPI_CXX_INCLUDE_PATH "${MPI_CXX_INCLUDE_PATH}" CACHE PATH "MPI CXX include path")
    endif()
    if(DEFINED MPI_CXX_LIBRARIES)
        set(MPI_CXX_LIBRARIES "${MPI_CXX_LIBRARIES}" CACHE FILEPATH "MPI CXX libraries")
    endif()

    # Find MPI package
    find_package(MPI REQUIRED COMPONENTS CXX)

    if(MPI_CXX_FOUND)
        message(STATUS "Found MPI:")
        message(STATUS "  MPI_CXX_COMPILER: ${MPI_CXX_COMPILER}")
        message(STATUS "  MPI_CXX_INCLUDE_DIRS: ${MPI_CXX_INCLUDE_DIRS}")
        message(STATUS "  MPI_CXX_LIBRARIES: ${MPI_CXX_LIBRARIES}")
        message(STATUS "  MPI version: ${MPI_CXX_VERSION}")
    else()
        message(FATAL_ERROR "MPI C++ support not found")
    endif()

    # Add MPI definitions
    add_definitions(-DUSE_MPI)

    # For OpenMPI RHEL 5.0.7 with GCC 14 - specific optimizations
    if(MPI_CXX_COMPILER MATCHES "mpi")
        # Detect OpenMPI version and configure accordingly
        execute_process(
            COMMAND ${MPI_CXX_COMPILER} --showme:version
            OUTPUT_VARIABLE MPI_VERSION_OUTPUT
            ERROR_QUIET
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )

        if(MPI_VERSION_OUTPUT MATCHES "Open MPI")
            message(STATUS "Detected OpenMPI: ${MPI_VERSION_OUTPUT}")

            # Enable OpenMPI specific optimizations
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DOMPI_SKIP_MPICXX")

            # For RHEL systems with OpenMPI 5.0.7
            if(MPI_VERSION_OUTPUT MATCHES "5\\.0\\.")
                message(STATUS "Configuring for OpenMPI 5.0.x")
                # Add any specific flags for OpenMPI 5.0.x
            endif()
        endif()
    endif()

    # Configure MPI with SYCL if both are enabled
    if(USE_SYCL_ACPP OR USE_SYCL_ICPX)
        message(STATUS "Configuring MPI with SYCL backend")

        # For Intel DPC++ with MPI
        if(USE_SYCL_ICPX)
            # Ensure MPI works with Intel DPC++ compiler
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DMPI_NO_CPPBIND")
            message(STATUS "Added MPI_NO_CPPBIND for Intel DPC++ compatibility")
        endif()

        # For AdaptiveCpp with MPI
        if(USE_SYCL_ACPP)
            # Ensure MPI works with AdaptiveCpp compiler
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DMPI_NO_CPPBIND")
            message(STATUS "Added MPI_NO_CPPBIND for AdaptiveCpp compatibility")
        endif()
    endif()

    # Set MPI compiler wrapper detection
    if(CMAKE_CXX_COMPILER MATCHES "mpi")
        message(STATUS "Using MPI compiler wrapper: ${CMAKE_CXX_COMPILER}")
        # MPI flags are automatically handled by compiler wrapper
    else()
        message(STATUS "Using standard compiler with MPI libraries")
        # Manual MPI linking will be handled in target_link_libraries
    endif()
else()
    message(STATUS "MPI support is disabled.")
endif()
# --- End MPI Configuration ---

## Set flags and definitions after project()
if(USE_CUDA)
  add_definitions(-DUSE_CUDA) # Kept for CUDA-specific sections in C++
endif()

if(DEBUG_BUILD)
  set(CMAKE_BUILD_TYPE Debug)
  message(STATUS "DEBUG_BUILD is ON: Build type set to Debug.")
endif()

if(USE_NVTX)
  add_definitions(-DUSE_NVTX)
endif()

if(USE_NCCL AND USE_CUDA)
  add_definitions(-DUSE_NCCL)
endif()


if(USE_CUDA)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
endif()
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(APPLE)
    set(CMAKE_MACOSX_RPATH ON)
endif()
if(USE_CUDA)
    set(CMAKE_BUILD_RPATH "${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}")
endif()

add_subdirectory(Backend)
add_subdirectory(Tests)

# Main application executable
add_executable(backends_main simple_main.cpp)
target_include_directories(backends_main PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(backends_main PRIVATE "lib${PROJECT_NAME}")

# Link backend-specific libraries to main executable
if(USE_CUDA)
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(backends_main PRIVATE
        CUDA::cuda_driver
        CUDA::cudart
        CUDA::curand
        CUDA::cublas)
    if(USE_NCCL)
        target_link_libraries(backends_main PRIVATE nccl)
    endif()
    if(USE_NVTX AND TARGET CUDA::nvToolsExt)
        target_link_libraries(backends_main PRIVATE CUDA::nvToolsExt)
    endif()
endif()

if(USE_METAL)
    target_link_libraries(backends_main PRIVATE
        METAL_CPP
        ${METAL_FRAMEWORK}
        ${FOUNDATION_FRAMEWORK}
        ${QUARTZCORE_FRAMEWORK}
    )
    target_include_directories(backends_main PRIVATE
        ${CMAKE_SOURCE_DIR}/extern/metal-cpp-cmake/metal-cmake/metal-cpp
        ${CMAKE_SOURCE_DIR}/extern/metal-cpp-cmake/metal-cmake/metal-cpp-extensions
    )
    if(TARGET metal_shaders_main)
        add_dependencies(backends_main metal_shaders_main)
    endif()
endif()

if(USE_MPI)
    if(CMAKE_CXX_COMPILER MATCHES "mpi")
        message(STATUS "MPI linking for backends_main handled automatically by compiler wrapper")
    else()
        target_link_libraries(backends_main PRIVATE MPI::MPI_CXX)
        target_include_directories(backends_main PRIVATE ${MPI_CXX_INCLUDE_DIRS})
        message(STATUS "Manually linked MPI libraries to backends_main")
    endif()
endif()

message(STATUS "Added main application executable: backends_main")

if(USE_PYBIND)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/extern/pybind11/CMakeLists.txt")
    add_subdirectory(extern/pybind11)
    add_subdirectory(src/pybindings)
  else()
    message(WARNING "pybind11 directory or CMakeLists.txt not found at extern/pybind11.")
  endif()
endif()
