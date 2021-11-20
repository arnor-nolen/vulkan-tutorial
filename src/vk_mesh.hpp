#pragma once

#include "vk_types.hpp"
#include <filesystem>
#include <glm/vec3.hpp>
#include <string_view>
#include <vector>

struct VertexInputDescription {
  std::vector<VkVertexInputBindingDescription> bindings;
  std::vector<VkVertexInputAttributeDescription> attributes;

  VkPipelineVertexInputStateCreateFlags flags = 0;
};

struct Vertex {
  glm::vec3 position;
  glm::vec3 normal;
  glm::vec3 color;

  static auto get_vertex_description() -> VertexInputDescription;
};

struct Mesh {
  // NOLINTNEXTLINE(misc-non-private-member-variables-in-classes)
  std::vector<Vertex> _vertices;
  // NOLINTNEXTLINE(misc-non-private-member-variables-in-classes)
  AllocatedBuffer _vertexBuffer;

  auto load_from_obj(const std::filesystem::path &filename) -> bool;
};
