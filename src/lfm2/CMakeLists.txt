# Local, opt-in: LFM2-MoE (Liquid LFM2 / LFM2.5 mixture-of-experts) engine for one AMD wave32 GPU through HIP.
# Built only with -DSTRATA_LFM2=ON from the root, or on its own: cmake -S src/lfm2 -B build-lfm2.
# Nothing here touches the Qwen targets.
cmake_minimum_required(VERSION 3.21)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(strata_lfm2 LANGUAGES CXX)
  include(CTest)
  set(_lfm2_build_tests ${BUILD_TESTING})
else()
  # Follow the parent's test switch without introducing global CTest settings.
  set(_lfm2_build_tests ${STRATA_BUILD_TESTS})
endif()

if(_lfm2_build_tests)
  add_executable(lfm2_unit_test tests/unit.cpp)
  set_target_properties(lfm2_unit_test PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
  target_include_directories(lfm2_unit_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  add_test(NAME lfm2_unit COMMAND lfm2_unit_test)
endif()

if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
  set(CMAKE_HIP_ARCHITECTURES gfx1100)
endif()
enable_language(HIP)

add_executable(strata-lfm2 lfm2_engine.hip)
set_target_properties(strata-lfm2 PROPERTIES HIP_STANDARD 17 HIP_STANDARD_REQUIRED ON)
target_include_directories(strata-lfm2 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(strata-lfm2 PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-O3>)

# CPU reference (oracle) - slow, for checking the GPU engine.
option(STRATA_LFM2_REF "Also build the LFM2 CPU reference (lfm2_ref)" ON)
if(STRATA_LFM2_REF)
  find_package(OpenMP)
  add_executable(lfm2_ref lfm2_ref.cpp)
  set_target_properties(lfm2_ref PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
  target_include_directories(lfm2_ref PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_compile_options(lfm2_ref PRIVATE -O3 -march=native)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(lfm2_ref PRIVATE OpenMP::OpenMP_CXX)
  endif()
endif()
