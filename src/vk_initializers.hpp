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

auto pipeline_shader_stage_create_info(VkShaderStageFlagBits stage,
                                       VkShaderModule shaderModule)
    -> VkPipelineShaderStageCreateInfo;

auto vertex_input_state_create_info() -> VkPipelineVertexInputStateCreateInfo;

auto input_assembly_creat_info(VkPrimitiveTopology topology)
    -> VkPipelineInputAssemblyStateCreateInfo;

auto rasterization_state_create_info(VkPolygonMode polygonMode)
    -> VkPipelineRasterizationStateCreateInfo;

auto multisampling_state_create_info() -> VkPipelineMultisampleStateCreateInfo;

auto color_blend_attachment_state() -> VkPipelineColorBlendAttachmentState;

auto pipeline_layout_create_info() -> VkPipelineLayoutCreateInfo;
} // namespace vkinit