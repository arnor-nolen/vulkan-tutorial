#include "vk_engine.hpp"

#include <SDL.h>
#include <SDL_vulkan.h>

#include <vk_initializers.hpp>
#include <vk_types.hpp>

#include <VkBootstrap.h>

#include <cstdint>
#include <fstream>
#include <iostream>

// we want to immediately abort when there is an error. In normal engines this
// would give an error message to the user, or perform a dump of state.
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define VK_CHECK(x)                                                            \
  do {                                                                         \
    VkResult err = x;                                                          \
    if (err) {                                                                 \
      std::cout << "Detected Vulkan error: " << err << std::endl;              \
      abort();                                                                 \
    }                                                                          \
  } while (0)

void VulkanEngine::init() {
  // We initialize SDL and create a window with it.
  SDL_Init(SDL_INIT_VIDEO);

  auto window_flags = static_cast<SDL_WindowFlags>(SDL_WINDOW_VULKAN);

  // NOLINTNEXTLINE(hicpp-signed-bitwise)
  _window = SDL_CreateWindow("Vulkan Engine", SDL_WINDOWPOS_UNDEFINED,
                             // NOLINTNEXTLINE(hicpp-signed-bitwise)
                             SDL_WINDOWPOS_UNDEFINED, _windowExtent.width,
                             _windowExtent.height, window_flags);

  // Initialization
  init_vulkan();
  init_swapchain();
  init_commands();
  init_default_renderpass();
  init_framebuffers();
  init_sync_structures();
  init_pipelines();

  // everything went fine
  _isInitialized = true;
}

void VulkanEngine::init_vulkan() {
  vkb::InstanceBuilder builder;

  // make the Vulkan instance, with basic debug features
  auto inst_ret = builder.set_app_name("Example Vulkan Application")
                      .request_validation_layers(true)
                      .require_api_version(1, 2, 0)
                      .use_default_debug_messenger()
                      .build();
  vkb::Instance vkb_inst = inst_ret.value();

  _instance = vkb_inst.instance;
  _debug_messenger = vkb_inst.debug_messenger;

  // get the surface of the window we opened with SDL
  SDL_Vulkan_CreateSurface(_window, _instance, &_surface);

  // use vkbootstrap to select a GPU.
  // we want a GPU that can write to the SDL surface and supports Vulkan 1.1
  vkb::PhysicalDeviceSelector selector{vkb_inst};
  vkb::PhysicalDevice physicalDevice =
      selector.set_minimum_version(1, 2).set_surface(_surface).select().value();

  // create the final Vulkan device
  vkb::DeviceBuilder deviceBuilder{physicalDevice};
  vkb::Device vkbDevice = deviceBuilder.build().value();

  // get the VkDevice handle used in the rest of a Vulkan application
  _device = vkbDevice.device;
  _chosenGPU = physicalDevice.physical_device;

  // Use vkbootstrap to get a Graphics queue
  _graphicsQueue = vkbDevice.get_queue(vkb::QueueType::graphics).value();
  _graphicsQueueFamily =
      vkbDevice.get_queue_index(vkb::QueueType::graphics).value();
}

void VulkanEngine::init_swapchain() {
  vkb::SwapchainBuilder swapchainBuilder{_chosenGPU, _device, _surface};
  vkb::Swapchain vkbSwapchain =
      swapchainBuilder
          .use_default_format_selection()
          // Use vsync preset mode
          .set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR)
          .set_desired_extent(_windowExtent.width, _windowExtent.height)
          .build()
          .value();

  // Store swapchain and its related images
  _swapchain = vkbSwapchain.swapchain;
  _swapchainImages = vkbSwapchain.get_images().value();
  _swapchainImageViews = vkbSwapchain.get_image_views().value();
  _swapchainImageFormat = vkbSwapchain.image_format;
}

void VulkanEngine::init_commands() {
  // Create a command pool for commands submitted to the graphics queue
  VkCommandPoolCreateInfo commandPoolInfo = vkinit::command_pool_create_info(
      _graphicsQueueFamily, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);
  VK_CHECK(
      vkCreateCommandPool(_device, &commandPoolInfo, nullptr, &_commandPool));

  // Allocate the default command buffer that we will use for rendering
  VkCommandBufferAllocateInfo cmdAllocInfo =
      vkinit::command_buffer_allocate_info(_commandPool, 1);
  VK_CHECK(
      vkAllocateCommandBuffers(_device, &cmdAllocInfo, &_mainCommandBuffer));
}

void VulkanEngine::init_default_renderpass() {
  // The renderpass will use this color attachment
  VkAttachmentDescription color_attachment = {};
  // The attachment will have the format needed by the swapchain
  color_attachment.format = _swapchainImageFormat;
  // 1 sample, we won't be doing MSAA
  color_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
  // We clear when this attachment is loaded
  color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  // We keep the attachment stored when the renderpass ends
  color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  // We don't care about stencil
  color_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  color_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;

  // We don't know or care about the starting layout of the attachment
  color_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

  // After the renderpass ends, the image has to be on a layout ready for
  // display
  color_attachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

  VkAttachmentReference color_attachment_ref = {};
  // Attachment number will index into the pAttachments array in the parent
  // rederpass itself
  color_attachment_ref.attachment = 0;
  color_attachment_ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

  // We are going to create 1 subpass, which is the minimum you can do
  VkSubpassDescription subpass = {};
  subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
  subpass.colorAttachmentCount = 1;
  subpass.pColorAttachments = &color_attachment_ref;

  VkRenderPassCreateInfo render_pass_info = {};
  render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;

  // Connect the color attachment to the info
  render_pass_info.attachmentCount = 1;
  render_pass_info.pAttachments = &color_attachment;
  // Connect the subpass to the info
  render_pass_info.subpassCount = 1;
  render_pass_info.pSubpasses = &subpass;

  VK_CHECK(
      vkCreateRenderPass(_device, &render_pass_info, nullptr, &_renderPass));
}

void VulkanEngine::init_framebuffers() {
  // Create the framebuffers for the swapchain images. This will connect the
  // render-pass to the images for rendering
  VkFramebufferCreateInfo fb_info = {};
  fb_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
  fb_info.pNext = nullptr;

  fb_info.renderPass = _renderPass;
  fb_info.attachmentCount = 1;
  fb_info.width = _windowExtent.width;
  fb_info.height = _windowExtent.height;
  fb_info.layers = 1;

  // Grab how many images we have in the swapchain
  auto swapchain_imagecount =
      static_cast<std::uint32_t>(_swapchainImages.size());
  _framebuffers = std::vector<VkFramebuffer>(swapchain_imagecount);

  // Create framebuffers for each of the swapchain image views
  for (std::uint32_t i = 0; i != swapchain_imagecount; ++i) {
    fb_info.pAttachments = &_swapchainImageViews[i];
    VK_CHECK(
        vkCreateFramebuffer(_device, &fb_info, nullptr, &_framebuffers[i]));
  }
}

void VulkanEngine::init_sync_structures() {
  // Create synchronization structures
  VkFenceCreateInfo fenceCreateInfo = {};
  fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  fenceCreateInfo.pNext = nullptr;

  // We want ot create the fence with the Create Signaled flag, so we can wait
  // on it before using it on a GPU command (fore the first time)
  fenceCreateInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

  VK_CHECK(vkCreateFence(_device, &fenceCreateInfo, nullptr, &_renderFence));

  // For the semaphores we don't need any flags
  VkSemaphoreCreateInfo semaphoreCreateInfo = {};
  semaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
  semaphoreCreateInfo.pNext = nullptr;
  semaphoreCreateInfo.flags = 0;

  VK_CHECK(vkCreateSemaphore(_device, &semaphoreCreateInfo, nullptr,
                             &_presentSemaphore));
  VK_CHECK(vkCreateSemaphore(_device, &semaphoreCreateInfo, nullptr,
                             &_renderSemaphore));
}

void VulkanEngine::init_pipelines() {
  VkShaderModule triangleFragShader;
  if (!load_shader_module("./shaders/triangle.frag.spv", &triangleFragShader)) {
    std::cout << "Error when building the triangle fragment shader module"
              << '\n';
  } else {
    std::cout << "Triangle fragment shader successfully loaded" << '\n';
  }

  VkShaderModule triangleVertexShader;
  if (!load_shader_module("./shaders/triangle.vert.spv",
                          &triangleVertexShader)) {
    std::cout << "Error when building the triangle vertex shader module"
              << '\n';
  } else {
    std::cout << "Triangle vertex shader successfully loaded" << '\n';
  }
}

auto VulkanEngine::load_shader_module(const char *filePath,
                                      VkShaderModule *outShaderModule) -> bool {
  // Open the file with cursor at the end
  std::ifstream file(filePath, std::ios::ate | std::ios::binary);
  if (!file.is_open()) {
    return false;
  }

  // Find what the size of the file is by looking up hte location of the cursor
  // Because the cursor is at the end, it gives the size directly in bytes
  size_t fileSize = static_cast<size_t>(file.tellg());

  // SPIR-V expects the buffer to be on uint32, so make sure to reserve an int
  // vector big enough for the entire file
  std::vector<std::uint32_t> buffer(fileSize / sizeof(std::uint32_t));

  // Put file cursor at the beginning
  file.seekg(0);

  // Load the entire file into the buffer

  // I have no choice becase of SPIR-V only accepting std::uint32_t
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
  file.read(reinterpret_cast<char *>(buffer.data()), fileSize);

  // Now that the file is loaded into the buffer, we can close it
  file.close();

  // Create a new shader module, using the buffer we loaded
  VkShaderModuleCreateInfo createInfo = {};
  createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  createInfo.pNext = nullptr;

  // codeSize has to be in bytes
  createInfo.codeSize = buffer.size() * sizeof(std::uint32_t);
  createInfo.pCode = buffer.data();

  // Check that the creation goes well
  VkShaderModule shaderModule;
  if (vkCreateShaderModule(_device, &createInfo, nullptr, &shaderModule) !=
      VK_SUCCESS) {
    return false;
  }
  *outShaderModule = shaderModule;
  return true;
}

void VulkanEngine::cleanup() {
  if (_isInitialized) {
    vkDestroyCommandPool(_device, _commandPool, nullptr);
    vkDestroySwapchainKHR(_device, _swapchain, nullptr);
    // Destroy the main renderpass
    vkDestroyRenderPass(_device, _renderPass, nullptr);
    // Destroy swapchain resources
    for (auto i : _swapchainImageViews) {
      vkDestroyImageView(_device, i, nullptr);
    }

    vkDestroyDevice(_device, nullptr);
    vkDestroySurfaceKHR(_instance, _surface, nullptr);
    vkb::destroy_debug_utils_messenger(_instance, _debug_messenger);
    vkDestroyInstance(_instance, nullptr);

    SDL_DestroyWindow(_window);
  }
}

void VulkanEngine::draw() {
  // Wait until the GPU has finished rendering the last frame. Timeout of 1
  // second.
  VK_CHECK(vkWaitForFences(_device, 1, &_renderFence, true, 1000000000));
  VK_CHECK(vkResetFences(_device, 1, &_renderFence));

  // Request image from the swapchain, one second timeout
  std::uint32_t swapchainImageIndex;
  VK_CHECK(vkAcquireNextImageKHR(_device, _swapchain, 1000000000,
                                 _presentSemaphore, 0, &swapchainImageIndex));

  // Now that we are sure that the commands finished executing, we can safely
  // reset the command buffer to begin recording again
  VK_CHECK(vkResetCommandBuffer(_mainCommandBuffer, 0));

  VkCommandBuffer cmd = _mainCommandBuffer;

  // Begin teh command buffer recording. We will use this command buffer exactly
  // once, so wa want to let Vulkan know that
  VkCommandBufferBeginInfo cmdBeginInfo = {};
  cmdBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  cmdBeginInfo.pNext = nullptr;

  cmdBeginInfo.pInheritanceInfo = nullptr;
  cmdBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

  VK_CHECK(vkBeginCommandBuffer(cmd, &cmdBeginInfo));

  // Make a clear-color from frame number. This will falsh with a 120*pi frame
  // period.
  VkClearValue clearValue;
  const float colorMagic = 120.0F;
  float flash = abs(sin(static_cast<float>(_frameNumber) / colorMagic));
  clearValue.color = {{0.0F, 0.0F, flash, 1.0F}};

  // Start the main renderpass.
  // We will use the clear color from above, and the framebuffer of the index
  // the swapchain gave us
  VkRenderPassBeginInfo rpInfo = {};
  rpInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
  rpInfo.pNext = nullptr;

  rpInfo.renderPass = _renderPass;
  rpInfo.renderArea.offset.x = 0;
  rpInfo.renderArea.offset.y = 0;
  rpInfo.renderArea.extent = _windowExtent;
  rpInfo.framebuffer = _framebuffers[swapchainImageIndex];

  // Connect clear values
  rpInfo.clearValueCount = 1;
  rpInfo.pClearValues = &clearValue;

  vkCmdBeginRenderPass(cmd, &rpInfo, VK_SUBPASS_CONTENTS_INLINE);

  // Finalize the render pass
  vkCmdEndRenderPass(cmd);
  // Finalize the comamnd buffer(we can no longer add commands, but it can now
  // be executed)
  VK_CHECK(vkEndCommandBuffer(cmd));

  // Prepare the submission to the queue
  // We want to wait on the _presentSemaphore, as that semaphore is siganled
  // when the swapchain is ready We will signal the _renderSemaphore, to signal
  // that rendering has finished
  VkSubmitInfo submit = {};
  submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit.pNext = nullptr;

  VkPipelineStageFlags waitStage =
      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  submit.pWaitDstStageMask = &waitStage;

  submit.waitSemaphoreCount = 1;
  submit.pWaitSemaphores = &_presentSemaphore;

  submit.waitSemaphoreCount = 1;
  submit.pSignalSemaphores = &_renderSemaphore;

  submit.commandBufferCount = 1;
  submit.pCommandBuffers = &cmd;

  // Submit command buffer to the queue and execute it.
  // _renderFence will now block until the graphic commands finish execution
  VK_CHECK(vkQueueSubmit(_graphicsQueue, 1, &submit, _renderFence));

  // This will put hte  image we just rendered into the visible window
  // We want to wait on the _renderSempahore for that,
  // as it's necessary that drawing commands have finished before the image is
  // displayed to the user
  VkPresentInfoKHR presentInfo = {};
  presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
  presentInfo.pNext = nullptr;

  presentInfo.pSwapchains = &_swapchain;
  presentInfo.swapchainCount = 1;

  presentInfo.pWaitSemaphores = &_renderSemaphore;
  presentInfo.waitSemaphoreCount = 1;

  presentInfo.pImageIndices = &swapchainImageIndex;

  VK_CHECK(vkQueuePresentKHR(_graphicsQueue, &presentInfo));

  // Increase the number of frames drawn
  ++_frameNumber;
}

void VulkanEngine::run() {
  SDL_Event e;
  bool bQuit = false;

  // main loop
  while (!bQuit) {
    // Handle events on queue
    while (SDL_PollEvent(&e) != 0) {
      // close the window when user alt-f4s or clicks the X button
      if (e.type == SDL_QUIT) {
        bQuit = true;
      }
    }

    draw();
  }
}
