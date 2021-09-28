﻿#pragma once

#include <cstdint>
#include <vector>
#include <vk_types.hpp>

constexpr int window_w = 1700;
constexpr int window_h = 900;

class VulkanEngine {
public:
  // initializes everything in the engine
  void init();

  // shuts down the engine
  void cleanup();

  // draw loop
  void draw();

  // run main loop
  void run();

private:
  // Members, all are public in Tutorial
  bool _isInitialized{false};
  int _frameNumber{0};

  VkExtent2D _windowExtent{window_w, window_h};

  struct SDL_Window *_window{nullptr};

  VkInstance _instance;                      // Vulkan library header
  VkDebugUtilsMessengerEXT _debug_messenger; // Vulkan debug output handle
  VkPhysicalDevice _chosenGPU;               // GPU chosen as the default device
  VkDevice _device;                          // Vulkan device for commands
  VkSurfaceKHR _surface;                     // Vulkan window surface

  VkSwapchainKHR _swapchain;
  // image format expected by the windowing system
  VkFormat _swapchainImageFormat;
  // array of images from the swapchain
  std::vector<VkImage> _swapchainImages;
  // array of image-views from the swapchain
  std::vector<VkImageView> _swapchainImageViews;

  VkQueue _graphicsQueue;             // Queue we will submit to
  std::uint32_t _graphicsQueueFamily; // Family of the queue
  VkCommandPool _commandPool;         // Command pool for our commands
  VkCommandBuffer _mainCommandBuffer; // Buffer we are recording to

  VkRenderPass _renderPass;
  std::vector<VkFramebuffer> _framebuffers;

  VkSemaphore _presentSemaphore, _renderSemaphore;
  VkFence _renderFence;

  VkPipelineLayout _trianglePipelineLayout;
  VkPipeline _trianglePipeline;

  // Functions
  void init_vulkan();
  void init_swapchain();
  void init_commands();
  void init_default_renderpass();
  void init_framebuffers();
  void init_sync_structures();
  void init_pipelines();
  // Loads a shader module from a SPIR-V file. Returns false if it errors.
  auto load_shader_module(const char *filePath, VkShaderModule *outShaderModule)
      -> bool;
};

class PipelineBuilder {
public:
  std::vector<VkPipelineShaderStageCreateInfo> _shaderStages;
  VkPipelineVertexInputStateCreateInfo _vertexInputInfo;
  VkPipelineInputAssemblyStateCreateInfo _inputAssembly;
  VkViewport _viewport;
  VkRect2D _scissor;
  VkPipelineRasterizationStateCreateInfo _rasterizer;
  VkPipelineColorBlendAttachmentState _colorBlendAttachment;
  VkPipelineMultisampleStateCreateInfo _multisampling;
  VkPipelineLayout _pipelineLayout;

  auto build_pipeline(VkDevice device, VkRenderPass pass) -> VkPipeline;
};