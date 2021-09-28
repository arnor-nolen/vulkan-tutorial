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

auto vkinit::pipeline_shader_stage_create_info(VkShaderStageFlagBits stage,
                                               VkShaderModule shaderModule)
    -> VkPipelineShaderStageCreateInfo {
  VkPipelineShaderStageCreateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  info.pNext = nullptr;

  // Shader stage
  info.stage = stage;
  // Module containing the code for this shdaer stage
  info.module = shaderModule;
  // The entry point of the shader
  info.pName = "main";
  return info;
}

auto vkinit::vertex_input_state_create_info()
    -> VkPipelineVertexInputStateCreateInfo {
  VkPipelineVertexInputStateCreateInfo info = {};
  info.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
  info.pNext = nullptr;

  // No vertex bindings or attributes
  info.vertexBindingDescriptionCount = 0;
  info.vertexAttributeDescriptionCount = 0;
  return info;
}

auto vkinit::input_assembly_creat_info(VkPrimitiveTopology topology)
    -> VkPipelineInputAssemblyStateCreateInfo {
  VkPipelineInputAssemblyStateCreateInfo info = {};
  info.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
  info.pNext = nullptr;

  info.topology = topology;
  // We are not going to use primitive restart on the entire tutorial so leave
  // it on false
  info.primitiveRestartEnable = VK_FALSE;
  return info;
}

auto vkinit::rasterization_state_create_info(VkPolygonMode polygonMode)
    -> VkPipelineRasterizationStateCreateInfo {
  VkPipelineRasterizationStateCreateInfo info = {};
  info.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
  info.pNext = nullptr;

  info.depthClampEnable = VK_FALSE;
  // Discards all primitives before the rasterization stage if enabled which we
  // don't want
  info.rasterizerDiscardEnable = VK_FALSE;

  info.polygonMode = polygonMode;
  info.lineWidth = 1.0F;
  // No backface cull
  info.cullMode = VK_CULL_MODE_NONE;
  info.frontFace = VK_FRONT_FACE_CLOCKWISE;
  // No depth bias
  info.depthBiasEnable = VK_FALSE;
  info.depthBiasConstantFactor = 0.0F;
  info.depthBiasClamp = 0.0F;
  info.depthBiasSlopeFactor = 0.0F;

  return info;
}

auto vkinit::multisampling_state_create_info()
    -> VkPipelineMultisampleStateCreateInfo {
  VkPipelineMultisampleStateCreateInfo info = {};
  info.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
  info.pNext = nullptr;

  info.sampleShadingEnable = VK_FALSE;
  // Multisampling defaulted to no multisampling (1 sample per pixel)
  info.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
  info.minSampleShading = 1.0F;
  info.pSampleMask = nullptr;
  info.alphaToCoverageEnable = VK_FALSE;
  info.alphaToOneEnable = VK_FALSE;

  return info;
}

auto vkinit::color_blend_attachment_state()
    -> VkPipelineColorBlendAttachmentState {
  VkPipelineColorBlendAttachmentState colorBlendAttachment = {};
  colorBlendAttachment.colorWriteMask =
      // NOLINTNEXTLINE(hicpp-signed-bitwise)
      VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
      VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

  colorBlendAttachment.blendEnable = VK_FALSE;
  return colorBlendAttachment;
}

auto vkinit::pipeline_layout_create_info() -> VkPipelineLayoutCreateInfo {
  VkPipelineLayoutCreateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  info.pNext = nullptr;

  // Empty defaults
  info.flags = 0;
  info.setLayoutCount = 0;
  info.pSetLayouts = nullptr;
  info.pushConstantRangeCount = 0;
  info.pPushConstantRanges = nullptr;

  return info;
}
