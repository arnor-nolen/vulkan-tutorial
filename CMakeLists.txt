cmake_minimum_required (VERSION 3.16)
project (vulkan-tutorial)

# Setting up Conan dependencies
include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup(TARGETS)

# Set compiler-specific stuff
if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")

  # Suppress compiler warnings:
  # 1. Boost.Spirit.X3 global variable shadowing.
  add_compile_options(/W4 /wd4459)

  # Suppress linker warnings:
  # 1. external libraries are compiled without debug info.
  # Prevent command line window from spawning
  # add_link_options(/IGNORE:4099 /SUBSYSTEM:WINDOWS /ENTRY:mainCRTStartup)

elseif (CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")

  # -fvisibility=hidden for external libraries
  add_compile_options(-Wall -Wextra -Wpedantic -Wshadow -fvisibility=hidden)

elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU")

  # -fvisibility=hidden for external libraries
  add_compile_options(-Wall -Wextra -Wpedantic -Wshadow -fvisibility=hidden)

else()

  message(WARNING "Unknown compiler, proceeding without additional compiler options")

endif()

# Set targets
file(GLOB_RECURSE SOURCE_FILES "src/*.cpp")
add_executable (${PROJECT_NAME} ${SOURCE_FILES})

# Use c++17 standard
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)

# Add src to the include path
target_include_directories(${PROJECT_NAME} PUBLIC src)

# -------------------------------------
# Compiling shaders for Vulkan 
# -------------------------------------
find_package(Vulkan REQUIRED)

find_program(GLSL_VALIDATOR glslangValidator HINTS /usr/bin /usr/local/bin $ENV{VULKAN_SDK}/Bin/ $ENV{VULKAN_SDK}/Bin32/)

## find all the shader files under the shaders folder
file(GLOB_RECURSE GLSL_SOURCE_FILES
    "${PROJECT_SOURCE_DIR}/shaders/*.frag"
    "${PROJECT_SOURCE_DIR}/shaders/*.vert"
    "${PROJECT_SOURCE_DIR}/shaders/*.comp"
    )

## iterate each shader
foreach(GLSL ${GLSL_SOURCE_FILES})
  message(STATUS "BUILDING SHADER")
  get_filename_component(FILE_NAME ${GLSL} NAME)
  set(SPIRV "${PROJECT_SOURCE_DIR}/shaders/${FILE_NAME}.spv")
  message(STATUS ${GLSL})
  ##execute glslang command to compile that specific shader
  add_custom_command(
    OUTPUT ${SPIRV}
    COMMAND ${GLSL_VALIDATOR} -V ${GLSL} -o ${SPIRV}
    DEPENDS ${GLSL})
  list(APPEND SPIRV_BINARY_FILES ${SPIRV})
endforeach(GLSL)

add_custom_target(
    Shaders 
    DEPENDS ${SPIRV_BINARY_FILES}
    )

# -------------------------------------
# End of Compiling shaders for Vulkan 
# -------------------------------------

# Link libraries
target_link_libraries(${PROJECT_NAME} ${CONAN_TARGETS} Vulkan::Vulkan)

# Add shaders check
add_dependencies(${PROJECT_NAME} Shaders)
