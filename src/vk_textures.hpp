#pragma once

#include "vk_engine.hpp"
#include "vk_types.hpp"
#include <filesystem>

namespace vkutil {
auto load_image_from_file(VulkanEngine &engine,
                          const std::filesystem::path &file,
                          AllocatedImage &outImage) -> bool;
} // namespace vkutil
