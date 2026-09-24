cmake_minimum_required(VERSION 3.16)
project(abcurves_renderer C)

set(ABC_MODEL_BINDING_DIR "" CACHE PATH "Optional directory from training.renderer.bind_native")
set(ABC_RENDERER_BLOB "${CMAKE_CURRENT_SOURCE_DIR}/../../models/renderer_global_h80.bin" CACHE FILEPATH "Artifact to test")
if(ABC_MODEL_BINDING_DIR AND NOT EXISTS "${ABC_MODEL_BINDING_DIR}/abc_model_binding.h")
  message(FATAL_ERROR "Binding directory has no generated abc_model_binding.h")
endif()

set(ABC_RENDERER_SOURCES
    src/abc_online.c
    src/abc_fixed.c
    src/abc_w5_stream.c
)

add_library(abcurves_renderer_core STATIC ${ABC_RENDERER_SOURCES})
add_library(abcurves_renderer SHARED ${ABC_RENDERER_SOURCES})

foreach(target abcurves_renderer_core abcurves_renderer)
  target_include_directories(${target} PUBLIC include PRIVATE src)
  if(ABC_MODEL_BINDING_DIR)
    target_include_directories(${target} PRIVATE "${ABC_MODEL_BINDING_DIR}")
    target_compile_definitions(${target} PRIVATE ABC_CUSTOM_MODEL_BINDING=1)
  endif()
  set_target_properties(${target} PROPERTIES
      C_STANDARD 99
      C_STANDARD_REQUIRED YES
      C_EXTENSIONS NO
  )
  if(MSVC)
    target_compile_options(${target} PRIVATE /W4 /fp:strict)
  else()
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
    target_link_libraries(${target} PUBLIC m)
  endif()
endforeach()

if(MSVC)
  target_link_options(abcurves_renderer PRIVATE /Brepro /INCREMENTAL:NO)
endif()

target_compile_definitions(abcurves_renderer PRIVATE ABC_RENDERER_EXPORTS)
set_target_properties(abcurves_renderer PROPERTIES
    C_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES
)

enable_testing()
add_executable(abc_renderer_tests tests/test_loader.c)
target_link_libraries(abc_renderer_tests PRIVATE abcurves_renderer_core)
if(ABC_MODEL_BINDING_DIR)
  target_compile_definitions(abc_renderer_tests PRIVATE ABC_CUSTOM_MODEL_BINDING=1)
endif()
set_target_properties(abc_renderer_tests PROPERTIES
    C_STANDARD 99
    C_STANDARD_REQUIRED YES
    C_EXTENSIONS NO
)
add_test(NAME abc_renderer_loader COMMAND abc_renderer_tests "${ABC_RENDERER_BLOB}")

add_executable(abc_fixed_math_tests tests/test_fixed_math.c)
target_include_directories(abc_fixed_math_tests PRIVATE include)
set_target_properties(abc_fixed_math_tests PROPERTIES
    C_STANDARD 99
    C_STANDARD_REQUIRED YES
    C_EXTENSIONS NO
)
if(NOT MSVC)
  target_link_libraries(abc_fixed_math_tests PRIVATE m)
endif()
add_test(NAME abc_fixed_math COMMAND abc_fixed_math_tests)

add_executable(abc_renderer_benchmark tests/benchmark_runtime.c)
target_link_libraries(abc_renderer_benchmark PRIVATE abcurves_renderer_core)
set_target_properties(abc_renderer_benchmark PROPERTIES
    C_STANDARD 99
    C_STANDARD_REQUIRED YES
    C_EXTENSIONS NO
)
