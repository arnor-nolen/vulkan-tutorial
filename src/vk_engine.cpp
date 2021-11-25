﻿#include "vk_engine.hpp"

#include <SDL.h>
#include <SDL_vulkan.h>
#include <glm/gtx/transform.hpp>

#include "vk_initializers.hpp"
#include "vk_types.hpp"

#include <VkBootstrap.h>

#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>

#include "vma_implementation.hpp"

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
  _window = SDL_CreateWindow(
      // NOLINTNEXTLINE(hicpp-signed-bitwise)
      "Vulkan Engine", SDL_WINDOWPOS_UNDEFINED,
      // NOLINTNEXTLINE(hicpp-signed-bitwise)
      SDL_WINDOWPOS_UNDEFINED, static_cast<int>(_windowExtent.width),
      static_cast<int>(_windowExtent.height), window_flags);

  // Initialization
  init_vulkan();
  init_swapchain();
  init_commands();
  init_default_renderpass();
  init_framebuffers();
  init_sync_structures();
  init_descriptors();
  init_pipelines();
  load_meshes();
  init_scene();

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

  // Initialize the memory allocator
  VmaAllocatorCreateInfo allocatorInfo = {};
  allocatorInfo.physicalDevice = _chosenGPU;
  allocatorInfo.device = _device;
  allocatorInfo.instance = _instance;
  vmaCreateAllocator(&allocatorInfo, &_allocator);

  vkGetPhysicalDeviceProperties(_chosenGPU, &_gpuProperties);
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

  // Depth image size will match the window
  VkExtent3D depthImageExtent = {_windowExtent.width, _windowExtent.height, 1};

  // Hardcoding the depth format to 32 bit float
  _depthFormat = VK_FORMAT_D32_SFLOAT;

  // The depth image will be an image with the format we selected and Depth
  // Attachment usage flag
  auto dimg_info = vkinit::image_create_info(
      _depthFormat, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
      depthImageExtent);

  // For the depth image we watn to allocate it from GPU local memory
  VmaAllocationCreateInfo dimg_allocinfo = {};
  dimg_allocinfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
  dimg_allocinfo.requiredFlags =
      VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

  // Allocate and create the image
  vmaCreateImage(_allocator, &dimg_info, &dimg_allocinfo, &_depthImage._image,
                 &_depthImage._allocation, nullptr);

  // Build and image-view for the depth image to use for rendering
  auto dview_info = vkinit::imageview_create_info(
      _depthFormat, _depthImage._image, VK_IMAGE_ASPECT_DEPTH_BIT);

  VK_CHECK(vkCreateImageView(_device, &dview_info, nullptr, &_depthImageView));

  _mainDeletionQueue.push_function([=]() {
    vkDestroyImageView(_device, _depthImageView, nullptr);
    vmaDestroyImage(_allocator, _depthImage._image, _depthImage._allocation);
    vkDestroySwapchainKHR(_device, _swapchain, nullptr);
  });
}

void VulkanEngine::init_commands() {
  // Create a command pool for commands submitted to the graphics queue
  auto commandPoolInfo = vkinit::command_pool_create_info(
      _graphicsQueueFamily, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);

  for (auto &&frame : _frames) {
    VK_CHECK(vkCreateCommandPool(_device, &commandPoolInfo, nullptr,
                                 &frame._commandPool));
    // Allocate the default command buffer that we will use for rendering
    auto cmdAllocInfo =
        vkinit::command_buffer_allocate_info(frame._commandPool, 1);
    VK_CHECK(vkAllocateCommandBuffers(_device, &cmdAllocInfo,
                                      &frame._mainCommandBuffer));

    _mainDeletionQueue.push_function(
        [=]() { vkDestroyCommandPool(_device, frame._commandPool, nullptr); });
  }
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

  // Depth attachment
  VkAttachmentDescription depth_attachment = {};
  depth_attachment.flags = 0;
  depth_attachment.format = _depthFormat;
  depth_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
  depth_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  depth_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  depth_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  depth_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  depth_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  depth_attachment.finalLayout =
      VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

  VkAttachmentReference depth_attachment_ref = {};
  depth_attachment_ref.attachment = 1;
  depth_attachment_ref.layout =
      VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

  // We are going to create 1 subpass, which is the minimum you can do
  VkSubpassDescription subpass = {};
  subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
  subpass.colorAttachmentCount = 1;
  subpass.pColorAttachments = &color_attachment_ref;
  subpass.pDepthStencilAttachment = &depth_attachment_ref;

  // Array of 2 attachments, 1 for color, and other for depth
  auto attachments = std::array<VkAttachmentDescription, 2>(
      {color_attachment, depth_attachment});

  VkRenderPassCreateInfo render_pass_info = {};
  render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;

  // Connect the color attachment to the info
  render_pass_info.attachmentCount = 2;
  render_pass_info.pAttachments = attachments.data();
  // Connect the subpass to the info
  render_pass_info.subpassCount = 1;
  render_pass_info.pSubpasses = &subpass;

  VK_CHECK(
      vkCreateRenderPass(_device, &render_pass_info, nullptr, &_renderPass));

  _mainDeletionQueue.push_function(
      [=]() { vkDestroyRenderPass(_device, _renderPass, nullptr); });
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
    auto attachments =
        std::array<VkImageView, 2>{_swapchainImageViews[i], _depthImageView};

    fb_info.pAttachments = attachments.data();
    fb_info.attachmentCount = 2;
    VK_CHECK(
        vkCreateFramebuffer(_device, &fb_info, nullptr, &_framebuffers[i]));

    _mainDeletionQueue.push_function([=]() {
      vkDestroyFramebuffer(_device, _framebuffers[i], nullptr);
      vkDestroyImageView(_device, _swapchainImageViews[i], nullptr);
    });
  }
}

void VulkanEngine::init_sync_structures() {
  // Create synchronization structures
  // We want ot create the fence with the Create Signaled flag, so we can wait
  // on it before using it on a GPU command (for the first time)
  auto fenceCreateInfo =
      vkinit::fence_create_info(VK_FENCE_CREATE_SIGNALED_BIT);

  // For the semaphores we don't need any flags
  auto semaphoreCreateInfo = vkinit::semaphore_create_info();

  for (auto &&frame : _frames) {
    VK_CHECK(
        vkCreateFence(_device, &fenceCreateInfo, nullptr, &frame._renderFence));
    _mainDeletionQueue.push_function(
        [=]() { vkDestroyFence(_device, frame._renderFence, nullptr); });

    VK_CHECK(vkCreateSemaphore(_device, &semaphoreCreateInfo, nullptr,
                               &frame._presentSemaphore));
    VK_CHECK(vkCreateSemaphore(_device, &semaphoreCreateInfo, nullptr,
                               &frame._renderSemaphore));

    _mainDeletionQueue.push_function([=]() {
      vkDestroySemaphore(_device, frame._presentSemaphore, nullptr);
      vkDestroySemaphore(_device, frame._renderSemaphore, nullptr);
    });
  }
}

void VulkanEngine::init_pipelines() {
  // Compile mesh vertex shader
  VkShaderModule meshVertexShader;
  if (!load_shader_module("./shaders/tri_mesh.vert.spv", &meshVertexShader)) {
    std::cout << "Error when building the triangle vertex shader module"
              << '\n';
  } else {
    std::cout << "Triangle vertex shader successfully loaded" << '\n';
  }

  // Compile colored triangle modules
  VkShaderModule triangleFragShader;
  if (!load_shader_module("./shaders/default_lit.frag.spv",
                          &triangleFragShader)) {
    std::cout << "Error when building the triangle fragment shader module"
              << '\n';
  } else {
    std::cout << "Triangle fragment shader successfully loaded" << '\n';
  }

  // Build the pipeline layout that controls the inputs/outputs of the
  // shader We are not using descriptor sets or other systems yet, so no
  // need to use anything other than empty default
  auto mesh_pipeline_layout_info = vkinit::pipeline_layout_create_info();

  // Setup push constants
  VkPushConstantRange push_constant;
  // This push constant range starts at the beginning
  push_constant.offset = 0;
  // This push constant range takes up the size of a MeshPushConstants struct
  push_constant.size = sizeof(MeshPushConstants);
  // This push constant range is accessible only in the vertex shader
  push_constant.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

  // Push constant setup
  mesh_pipeline_layout_info.pPushConstantRanges = &push_constant;
  mesh_pipeline_layout_info.pushConstantRangeCount = 1;

  // Hook the global set layout
  mesh_pipeline_layout_info.setLayoutCount = 1;
  mesh_pipeline_layout_info.pSetLayouts = &_globalSetLayout;

  VK_CHECK(vkCreatePipelineLayout(_device, &mesh_pipeline_layout_info, nullptr,
                                  &_meshPipelineLayout));

  // Build the stage-create-info for both vertex and fragment stages. This
  // lets the pieline knwo the shader modules per stage
  PipelineBuilder pipelineBuilder;

  // Vertex input controls how to read vertices from vertex buffers. We
  // aren't using it yet.
  pipelineBuilder._vertexInputInfo = vkinit::vertex_input_state_create_info();

  // Input assembly is the configuration for drawing triangle lists, strips,
  // or individual points. We are just going to draw triangle list
  pipelineBuilder._inputAssembly =
      vkinit::input_assembly_create_info(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);

  // Build viewport and scissor from the swapchain extents
  pipelineBuilder._viewport.x = 0.0F;
  pipelineBuilder._viewport.y = 0.0F;
  pipelineBuilder._viewport.width = static_cast<float>(_windowExtent.width);
  pipelineBuilder._viewport.height = static_cast<float>(_windowExtent.height);
  pipelineBuilder._viewport.minDepth = 0.0F;
  pipelineBuilder._viewport.maxDepth = 1.0F;

  pipelineBuilder._scissor.offset = {0, 0};
  pipelineBuilder._scissor.extent = _windowExtent;

  // Configure the rasterizer to draw filled triangles
  pipelineBuilder._rasterizer =
      vkinit::rasterization_state_create_info(VK_POLYGON_MODE_FILL);

  // We don't use multisampling, so just run the default one
  pipelineBuilder._multisampling = vkinit::multisampling_state_create_info();

  // A single blend attachemnt with now blending and writing to RGBA
  pipelineBuilder._colorBlendAttachment =
      vkinit::color_blend_attachment_state();

  // Add depth testing
  pipelineBuilder._depthStencil = vkinit::depth_stencil_create_info(
      true, true, VK_COMPARE_OP_LESS_OR_EQUAL);

  // Build the mesh pipeline
  VertexInputDescription vertexDescription = Vertex::get_vertex_description();

  // Connect the pipeline builder vertex input info to the one we get from
  // Vertex
  pipelineBuilder._vertexInputInfo.pVertexAttributeDescriptions =
      vertexDescription.attributes.data();
  pipelineBuilder._vertexInputInfo.vertexAttributeDescriptionCount =
      static_cast<std::uint32_t>(vertexDescription.attributes.size());

  pipelineBuilder._vertexInputInfo.pVertexBindingDescriptions =
      vertexDescription.bindings.data();
  pipelineBuilder._vertexInputInfo.vertexBindingDescriptionCount =
      static_cast<std::uint32_t>(vertexDescription.bindings.size());

  // Add the other shaders
  pipelineBuilder._shaderStages.push_back(
      // NOLINTNEXTLINE(clang-analyzer-core.CallAndMessage)
      vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_VERTEX_BIT,
                                                meshVertexShader));
  // Make sure that triangleFragShader is holding the compiled
  // colored_triangle.frag
  pipelineBuilder._shaderStages.push_back(
      vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_FRAGMENT_BIT,
                                                triangleFragShader));

  pipelineBuilder._pipelineLayout = _meshPipelineLayout;

  // Build the mesh triangle pipleine
  _meshPipeline = pipelineBuilder.build_pipeline(_device, _renderPass);

  create_material(_meshPipeline, _meshPipelineLayout, "defaultMesh");

  // Destroy all shader modules, outside of the queue
  vkDestroyShaderModule(_device, meshVertexShader, nullptr);
  vkDestroyShaderModule(_device, triangleFragShader, nullptr);

  _mainDeletionQueue.push_function([=]() {
    vkDestroyPipeline(_device, _meshPipeline, nullptr);

    vkDestroyPipelineLayout(_device, _meshPipelineLayout, nullptr);
  });
}

void VulkanEngine::init_scene() {
  RenderObject monkey = {.mesh = get_mesh("monkey"),
                         .material = get_material("defaultMesh"),
                         .transformMatrix = glm::mat4{1.F}};

  _renderables.push_back(monkey);

  for (int x = -20; x <= 20; x++) {
    for (int y = -20; y <= 20; y++) {
      glm::mat4 translation =
          glm::translate(glm::mat4{1.F}, glm::vec3(x, 0.F, y));
      glm::mat4 scale = glm::scale(glm::mat4{1.F}, glm::vec3(.2F, .2F, .2F));
      RenderObject tri = {.mesh = get_mesh("triangle"),
                          .material = get_material("defaultMesh"),
                          .transformMatrix = translation * scale};
      _renderables.push_back(tri);
    }
  }
}

void VulkanEngine::init_descriptors() {
  const size_t sceneParamBufferSize =
      _frames.size() * pad_uniform_buffer_size(sizeof(GPUSceneData));
  _sceneParameterBuffer =
      create_buffer(sceneParamBufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                    VMA_MEMORY_USAGE_CPU_TO_GPU);

  // Create a descriptor pool that will hold 10 uniform buffers
  std::vector<VkDescriptorPoolSize> sizes = {
      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 10},
      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 10}};

  VkDescriptorPoolCreateInfo pool_info = {};
  pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  pool_info.flags = 0;
  pool_info.maxSets = 10;
  pool_info.poolSizeCount = static_cast<std::uint32_t>(sizes.size());
  pool_info.pPoolSizes = sizes.data();

  vkCreateDescriptorPool(_device, &pool_info, nullptr, &_descriptorPool);

  // Information about the binding
  auto cameraBind = vkinit::descriptorset_layout_binding(
      VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT, 0);

  auto sceneBind = vkinit::descriptorset_layout_binding(
      VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
      static_cast<unsigned int>(VK_SHADER_STAGE_VERTEX_BIT) |
          static_cast<unsigned int>(VK_SHADER_STAGE_FRAGMENT_BIT),
      1);

  auto bindings =
      std::array<VkDescriptorSetLayoutBinding, 2>{cameraBind, sceneBind};

  VkDescriptorSetLayoutCreateInfo setInfo = {};
  setInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  setInfo.pNext = nullptr;

  setInfo.bindingCount = 2;
  setInfo.flags = 0;
  setInfo.pBindings = bindings.data();

  vkCreateDescriptorSetLayout(_device, &setInfo, nullptr, &_globalSetLayout);

  _mainDeletionQueue.push_function([=]() {
    vkDestroyDescriptorSetLayout(_device, _globalSetLayout, nullptr);
    vkDestroyDescriptorPool(_device, _descriptorPool, nullptr);
    vmaDestroyBuffer(_allocator, _sceneParameterBuffer._buffer,
                     _sceneParameterBuffer._allocation);
  });

  for (auto &&frame : _frames) {
    frame.cameraBuffer =
        create_buffer(sizeof(GPUCameraData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                      VMA_MEMORY_USAGE_CPU_TO_GPU);

    // Allocate one descriptor set for each frame
    VkDescriptorSetAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.pNext = nullptr;

    // Using the pool we just set
    allocInfo.descriptorPool = _descriptorPool;
    // Only 1 descriptor
    allocInfo.descriptorSetCount = 1;
    // Using the global data layout
    allocInfo.pSetLayouts = &_globalSetLayout;

    vkAllocateDescriptorSets(_device, &allocInfo, &frame.globalDescriptor);

    // Information about the buffer we want to point at in the descriptor
    VkDescriptorBufferInfo cameraInfo;
    // It will be the camera buffer
    cameraInfo.buffer = frame.cameraBuffer._buffer;
    // At 0 offset
    cameraInfo.offset = 0;
    // Of the size of a camera data struct
    cameraInfo.range = sizeof(GPUCameraData);

    VkDescriptorBufferInfo sceneInfo;
    sceneInfo.buffer = _sceneParameterBuffer._buffer;
    sceneInfo.offset = 0;
    sceneInfo.range = sizeof(GPUSceneData);

    auto cameraWrite =
        vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                        frame.globalDescriptor, &cameraInfo, 0);

    auto sceneWrite = vkinit::write_descriptor_buffer(
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, frame.globalDescriptor,
        &sceneInfo, 1);

    auto setWrites =
        std::array<VkWriteDescriptorSet, 2>{cameraWrite, sceneWrite};

    vkUpdateDescriptorSets(_device, 2, setWrites.data(), 0, nullptr);

    _mainDeletionQueue.push_function([=]() {
      vmaDestroyBuffer(_allocator, frame.cameraBuffer._buffer,
                       frame.cameraBuffer._allocation);
    });
  }
}

void VulkanEngine::load_meshes() {
  // Make the array 3 vertices long
  _triangleMesh._vertices.resize(3);

  // Vertex positions
  _triangleMesh._vertices[0].position = {1.F, 1.F, 0.F};
  _triangleMesh._vertices[1].position = {-1.F, 1.F, 0.F};
  _triangleMesh._vertices[2].position = {0.F, -1.F, 0.F};

  // Vertex colors, all green
  _triangleMesh._vertices[0].color = {0.F, 1.F, 0.F};
  _triangleMesh._vertices[1].color = {0.F, 1.F, 0.F};
  _triangleMesh._vertices[2].color = {0.F, 1.F, 0.F};

  // Load the monkey
  _monkeyMesh.load_from_obj("./assets/monkey_smooth.obj");

  // We don't care about vertex normals
  upload_mesh(_triangleMesh);
  upload_mesh(_monkeyMesh);

  // Note that we are copying them. Eventually we will delete teh hardcoded
  // _monkey and _triangle meshes, so it's no problem now
  _meshes["monkey"] = _monkeyMesh;
  _meshes["triangle"] = _triangleMesh;
}

void VulkanEngine::upload_mesh(Mesh &mesh) {
  // Allocate vertex buffer
  VkBufferCreateInfo bufferInfo = {};
  bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  // This is the total size, in bytes, of the buffer we are allocating
  bufferInfo.size = mesh._vertices.size() * sizeof(Vertex);
  // This buffer is going to be used as a Vertex Buffer
  bufferInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;

  // Let the VMA library know that this data should be writeable by CPU, but
  // also readable by GPU
  VmaAllocationCreateInfo vmaallocInfo = {};
  vmaallocInfo.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;

  // Allocate the buffer
  VK_CHECK(vmaCreateBuffer(_allocator, &bufferInfo, &vmaallocInfo,
                           &mesh._vertexBuffer._buffer,
                           &mesh._vertexBuffer._allocation, nullptr));

  // Add the destruction of triangle mesh buffer to the deletion queue
  _mainDeletionQueue.push_function([=]() {
    vmaDestroyBuffer(_allocator, mesh._vertexBuffer._buffer,
                     mesh._vertexBuffer._allocation);
  });

  // Copy vertex data
  void *data;
  vmaMapMemory(_allocator, mesh._vertexBuffer._allocation, &data);
  // NOLINTNEXTLINE(clang-analyzer-core.CallAndMessage)
  memcpy(data, mesh._vertices.data(), mesh._vertices.size() * sizeof(Vertex));
  vmaUnmapMemory(_allocator, mesh._vertexBuffer._allocation);
}

auto VulkanEngine::load_shader_module(const char *filePath,
                                      VkShaderModule *outShaderModule) -> bool {
  // Open the file with cursor at the end
  std::ifstream file(filePath, std::ios::ate | std::ios::binary);
  if (!file.is_open()) {
    return false;
  }

  // Find what the size of the file is by looking up hte location of the
  // cursor Because the cursor is at the end, it gives the size directly in
  // bytes
  size_t fileSize = static_cast<size_t>(file.tellg());

  // SPIR-V expects the buffer to be on uint32, so make sure to reserve an
  // int vector big enough for the entire file
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

auto VulkanEngine::create_material(VkPipeline pipeline, VkPipelineLayout layout,
                                   const std::string &name) -> Material * {
  Material mat = {.pipeline = pipeline, .pipelineLayout = layout};
  _materials[name] = mat;
  return &_materials[name];
}

auto VulkanEngine::get_material(const std::string &name) -> Material * {
  // Search for the object and return nullptr if not found
  auto it = _materials.find(name);
  if (it == _materials.end()) {
    return nullptr;
  }
  return &(*it).second;
}

auto VulkanEngine::get_mesh(const std::string &name) -> Mesh * {
  auto it = _meshes.find(name);
  if (it == _meshes.end()) {
    return nullptr;
  }
  return &(*it).second;
}

void VulkanEngine::draw_objects(VkCommandBuffer cmd, RenderObject *first,
                                size_t count) {
  // Make a model view matrix for rendering the object
  // Camera view
  glm::vec3 camPos = {0.F, -6.F, -10.F};
  glm::mat4 view = glm::translate(glm::mat4(1.F), camPos);

  glm::mat4 projection =
      glm::perspective(glm::radians(70.F), 1700.F / 900.F, 0.1F, 200.F);
  projection[1][1] *= -1;

  // Fill a GPU camera data struct
  GPUCameraData camData = {
      .view = view, .projection = projection, .viewproj = projection * view};

  // And copy it to the buffer
  void *data;
  vmaMapMemory(_allocator, get_current_frame().cameraBuffer._allocation, &data);
  memcpy(data, &camData, sizeof(GPUCameraData));
  vmaUnmapMemory(_allocator, get_current_frame().cameraBuffer._allocation);

  Mesh *lastMesh = nullptr;
  Material *lastMaterial = nullptr;

  for (size_t i = 0; i != count; ++i) {
    RenderObject &object = first[i];

    // Only bind the pipeline if it doesn't match with the already bound one
    if (object.material != lastMaterial) {
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                        object.material->pipeline);
      lastMaterial = object.material;

      // Offset for our scene buffer
      unsigned int frameIndex = _frameNumber % _frames.size();
      auto uniform_offset = static_cast<std::uint32_t>(
          pad_uniform_buffer_size(sizeof(GPUSceneData)) * frameIndex);
      // Bind the descriptor set when changing the pipeline
      vkCmdBindDescriptorSets(
          cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, object.material->pipelineLayout,
          0, 1, &get_current_frame().globalDescriptor, 1, &uniform_offset);
    }

    MeshPushConstants constants = {.render_matrix = object.transformMatrix};

    // Upload the mesh to the GPU via push constants
    vkCmdPushConstants(cmd, object.material->pipelineLayout,
                       VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(MeshPushConstants),
                       &constants);

    // Only bind the mesh f it's a different one from last bind
    if (object.mesh != lastMesh) {
      // Bind the mesh vertex buffer with offset 0
      VkDeviceSize offset = 0;
      vkCmdBindVertexBuffers(cmd, 0, 1, &object.mesh->_vertexBuffer._buffer,
                             &offset);
      lastMesh = object.mesh;
    }
    // We can draw now
    vkCmdDraw(cmd, static_cast<uint32_t>(object.mesh->_vertices.size()), 1, 0,
              0);
  }
}

auto VulkanEngine::get_current_frame() -> FrameData & {
  return _frames.at(_frameNumber % _frames.size());
}

auto VulkanEngine::create_buffer(size_t allocSize, VkBufferUsageFlags usage,
                                 VmaMemoryUsage memoryUsage)
    -> AllocatedBuffer {
  // Allocate vertex buffer
  VkBufferCreateInfo bufferInfo = {};
  bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bufferInfo.pNext = nullptr;

  bufferInfo.size = allocSize;
  bufferInfo.usage = usage;

  VmaAllocationCreateInfo vmaallocInfo = {};
  vmaallocInfo.usage = memoryUsage;

  AllocatedBuffer newBuffer;

  VK_CHECK(vmaCreateBuffer(_allocator, &bufferInfo, &vmaallocInfo,
                           &newBuffer._buffer, &newBuffer._allocation,
                           nullptr));

  return newBuffer;
}

auto VulkanEngine::pad_uniform_buffer_size(size_t originalSize) const
    -> size_t {
  // Calculate required alignment based on minimum device offset alignment
  size_t minUboAlignment =
      _gpuProperties.limits.minUniformBufferOffsetAlignment;
  size_t alignedSize = originalSize;
  if (minUboAlignment > 0) {
    alignedSize = (alignedSize + minUboAlignment - 1) & ~(minUboAlignment - 1);
  }
  return alignedSize;
}

void VulkanEngine::cleanup() {
  if (_isInitialized) {
    // Make sure the GPU has stopped doing its things
    constexpr int timeout = 1000000000;
    --_frameNumber;
    vkWaitForFences(_device, 1, &get_current_frame()._renderFence,
                    static_cast<VkBool32>(true), timeout);
    ++_frameNumber;

    _mainDeletionQueue.flush();

    vmaDestroyAllocator(_allocator);

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
  VK_CHECK(vkWaitForFences(_device, 1, &get_current_frame()._renderFence, true,
                           1000000000));
  VK_CHECK(vkResetFences(_device, 1, &get_current_frame()._renderFence));

  // Now that we are sure that the commands finished executing, we can
  // safely reset the command buffer to begin recording again
  VK_CHECK(vkResetCommandBuffer(get_current_frame()._mainCommandBuffer, 0));

  // Request image from the swapchain, one second timeout
  std::uint32_t swapchainImageIndex;
  VK_CHECK(vkAcquireNextImageKHR(_device, _swapchain, 1000000000,
                                 get_current_frame()._presentSemaphore, 0,
                                 &swapchainImageIndex));

  VkCommandBuffer cmd = get_current_frame()._mainCommandBuffer;

  // Begin the command buffer recording. We will use this command buffer
  // exactly once, so wa want to let Vulkan know that
  auto cmdBeginInfo = vkinit::command_buffer_begin_info(
      VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

  VK_CHECK(vkBeginCommandBuffer(cmd, &cmdBeginInfo));

  // Make a clear-color from frame number. This will falsh with a 120*pi
  // frame period.
  VkClearValue clearValue;
  const float colorMagic = 120.0F;
  float framed = static_cast<float>(_frameNumber) / colorMagic;
  float flash = abs(sin(framed));
  clearValue.color = {{0.0F, 0.0F, flash, 1.0F}};

  _sceneParameters.ambientColor = {sin(framed), 0, cos(framed), 1};

  char *sceneData;
  vmaMapMemory(_allocator, _sceneParameterBuffer._allocation,
               (void **)&sceneData);

  unsigned int frameIndex = _frameNumber % _frames.size();
  sceneData += pad_uniform_buffer_size(sizeof(GPUSceneData)) * frameIndex;
  memcpy(sceneData, &_sceneParameters, sizeof(GPUSceneData));
  vmaUnmapMemory(_allocator, _sceneParameterBuffer._allocation);

  // Clear depth at 1
  VkClearValue depthClear;
  depthClear.depthStencil.depth = 1.F;

  // Start the main renderpass.
  // We will use the clear color from above, and the framebuffer of the
  // index the swapchain gave us
  auto rpInfo = vkinit::renderpass_begin_info(
      _renderPass, _windowExtent, _framebuffers[swapchainImageIndex]);

  // Connect clear values
  auto clearValues = std::array<VkClearValue, 2>{clearValue, depthClear};
  rpInfo.clearValueCount = static_cast<std::uint32_t>(clearValues.size());
  rpInfo.pClearValues = clearValues.data();

  vkCmdBeginRenderPass(cmd, &rpInfo, VK_SUBPASS_CONTENTS_INLINE);

  draw_objects(cmd, _renderables.data(), _renderables.size());

  // Finalize the render pass
  vkCmdEndRenderPass(cmd);
  // Finalize the comamnd buffer(we can no longer add commands, but it can
  // now be executed)
  VK_CHECK(vkEndCommandBuffer(cmd));

  // Prepare the submission to the queue
  // We want to wait on the _presentSemaphore, as that semaphore is signaled
  // when the swapchain is ready. We will signal the _renderSemaphore, to
  // signal that rendering has finished
  VkSubmitInfo submit = {};
  submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit.pNext = nullptr;

  VkPipelineStageFlags waitStage =
      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  submit.pWaitDstStageMask = &waitStage;

  submit.waitSemaphoreCount = 1;
  submit.pWaitSemaphores = &get_current_frame()._presentSemaphore;

  submit.signalSemaphoreCount = 1;
  submit.pSignalSemaphores = &get_current_frame()._renderSemaphore;

  submit.commandBufferCount = 1;
  submit.pCommandBuffers = &cmd;

  // Submit command buffer to the queue and execute it.
  // _renderFence will now block until the graphic commands finish execution
  VK_CHECK(vkQueueSubmit(_graphicsQueue, 1, &submit,
                         get_current_frame()._renderFence));

  // This will put the image we just rendered into the visible window
  // We want to wait on the _renderSemaphore for that,
  // as it's necessary that drawing commands have finished before the image
  // is displayed to the user
  VkPresentInfoKHR presentInfo = {};
  presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
  presentInfo.pNext = nullptr;

  presentInfo.pSwapchains = &_swapchain;
  presentInfo.swapchainCount = 1;
  presentInfo.pWaitSemaphores = &get_current_frame()._renderSemaphore;
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
      } else if (e.type == SDL_KEYDOWN) {
        if (e.key.keysym.sym == SDLK_SPACE) {
          _selectedShader += 1;
          if (_selectedShader > 1) {
            _selectedShader = 0;
          }
        }
      }
    }

    draw();
  }
}

auto PipelineBuilder::build_pipeline(VkDevice device, VkRenderPass pass)
    -> VkPipeline {
  // Make viewport state from our stored viewpor and scissor.
  // At the moment we won't support multiple viewports or scissors.
  VkPipelineViewportStateCreateInfo viewportState = {};
  viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
  viewportState.pNext = nullptr;

  viewportState.viewportCount = 1;
  viewportState.pViewports = &_viewport;
  viewportState.scissorCount = 1;
  viewportState.pScissors = &_scissor;

  // Setup dummy color blending. We aren't using transparent objects yet the
  // blending is just "no blend", but we do write to the color attachment
  VkPipelineColorBlendStateCreateInfo colorBlending = {};
  colorBlending.sType =
      VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
  colorBlending.pNext = nullptr;

  colorBlending.logicOpEnable = VK_FALSE;
  colorBlending.logicOp = VK_LOGIC_OP_COPY;
  colorBlending.attachmentCount = 1;
  colorBlending.pAttachments = &_colorBlendAttachment;

  // Build the actual pipeline
  // We now use all of the info structs we have been writing into this one
  // to create the pipeline
  VkGraphicsPipelineCreateInfo pipelineInfo = {};
  pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
  pipelineInfo.pNext = nullptr;

  pipelineInfo.stageCount = static_cast<std::uint32_t>(_shaderStages.size());
  pipelineInfo.pStages = _shaderStages.data();
  pipelineInfo.pVertexInputState = &_vertexInputInfo;
  pipelineInfo.pInputAssemblyState = &_inputAssembly;
  pipelineInfo.pViewportState = &viewportState;
  pipelineInfo.pRasterizationState = &_rasterizer;
  pipelineInfo.pMultisampleState = &_multisampling;
  pipelineInfo.pColorBlendState = &colorBlending;
  pipelineInfo.layout = _pipelineLayout;
  pipelineInfo.renderPass = pass;
  pipelineInfo.subpass = 0;
  pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
  pipelineInfo.pDepthStencilState = &_depthStencil;

  // It's easy to error out on create graphics pipeline, so we handle it a
  // bit better that the common VK_CHECK case
  VkPipeline newPipeline;
  if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo,
                                nullptr, &newPipeline) != VK_SUCCESS) {
    std::cout << "Failed to create pipeline\n";
    return VK_NULL_HANDLE;
  }
  return newPipeline;
}
