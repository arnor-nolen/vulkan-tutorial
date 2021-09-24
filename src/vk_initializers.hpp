#pragma once

#include <cstdint>
#include <vk_types.hpp>


namespace vkinit {
auto command_pool_create_info(std::uint32_t queueFamilyIndex,
                              VkCommandPoolCreateFlags flags = 0)
    -> VkCommandPoolCreateInfo;

auto command_buffer_allocate_info(
    VkCommandPool pool, std::uint32_t count = 1,
    VkCommandBufferLevel level = VK_COMMAND_BUFFER_LEVEL_PRIMARY)
    -> VkCommandBufferAllocateInfo;
} // namespace vkinit