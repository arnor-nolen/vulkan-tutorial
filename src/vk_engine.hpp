// vulkan_guide.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#include <vk_types.hpp>

constexpr int window_x = 1700;
constexpr int window_y = 900;

class VulkanEngine {
  bool _isInitialized{false};
  int _frameNumber{0};

  VkExtent2D _windowExtent{window_x, window_y};

  struct SDL_Window *_window{nullptr};

public:
  // initializes everything in the engine
  void init();

  // shuts down the engine
  void cleanup();

  // draw loop
  void draw();

  // run main loop
  void run();
};
