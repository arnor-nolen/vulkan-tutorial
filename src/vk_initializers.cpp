#include <cstdint>
#include <vk_initializers.hpp>

// Create a command pool for commands submitted to the graphics queue
auto vkinit::command_pool_create_info(uint32_t queueFamilyIndex,
                                      VkCommandPoolCreateFlags flags)
    -> VkCommandPoolCreateInfo {
  VkCommandPoolCreateInfo info = {};
  info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  info.pNext = nullptr;

  info.queueFamilyIndex = queueFamilyIndex;
  info.flags = flags;

  return info;
}

// Allocate the command buffer that will be used for rendering
auto vkinit::command_buffer_allocate_info(VkCommandPool pool,
                                          std::uint32_t count,
                                          VkCommandBufferLevel level)
    -> VkCommandBufferAllocateInfo {
  VkCommandBufferAllocateInfo info = {};
  info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  info.pNext = nullptr;

  info.commandPool = pool;
  info.commandBufferCount = count;
  info.level = level;

  return info;
}
