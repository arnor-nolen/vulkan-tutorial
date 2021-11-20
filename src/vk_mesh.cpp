#include "vk_mesh.hpp"
#include <array>
#include <filesystem>
#include <iostream>
#include <string_view>
#include <tiny_obj_loader.h>

auto Vertex::get_vertex_description() -> VertexInputDescription {
  VertexInputDescription description;

  // We will have just 1 vertex buffer binding, with a per-vertex rate
  VkVertexInputBindingDescription mainBinding = {};
  mainBinding.binding = 0;
  mainBinding.stride = sizeof(Vertex);
  mainBinding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

  description.bindings.push_back(mainBinding);

  // Position will be stored at Location 0
  VkVertexInputAttributeDescription positionAttribute = {};
  positionAttribute.binding = 0;
  positionAttribute.location = 0;
  positionAttribute.format = VK_FORMAT_R32G32B32_SFLOAT;
  positionAttribute.offset = offsetof(Vertex, position);

  // Normal will be stored at Location 1
  VkVertexInputAttributeDescription normalAttribute = {};
  normalAttribute.binding = 0;
  normalAttribute.location = 1;
  normalAttribute.format = VK_FORMAT_R32G32B32_SFLOAT;
  normalAttribute.offset = offsetof(Vertex, normal);

  // Color will be stored at Location 2
  VkVertexInputAttributeDescription colorAttribute = {};
  colorAttribute.binding = 0;
  colorAttribute.location = 2;
  colorAttribute.format = VK_FORMAT_R32G32B32_SFLOAT;
  colorAttribute.offset = offsetof(Vertex, color);

  description.attributes.push_back(positionAttribute);
  description.attributes.push_back(normalAttribute);
  description.attributes.push_back(colorAttribute);

  return description;
}

auto Mesh::load_from_obj(const std::filesystem::path &filename) -> bool {

  auto material_path = std::filesystem::path(filename).remove_filename();

  // Attrib will contain the vertex arrays of the file
  tinyobj::attrib_t attrib;
  // Shapes contains the info for each separate object in the file
  std::vector<tinyobj::shape_t> shapes;
  // Materials contains the information about the material of each shape
  std::vector<tinyobj::material_t> materials;

  // Error and warning output from the load function
  std::string err;

  // Load the OBJ file
  tinyobj::LoadObj(&attrib, &shapes, &materials, &err,
                   filename.string().c_str(), material_path.string().c_str());

  // If we have any error, print it to the console, and break the mesh loading.
  // This happens if the file can't be found or is malformed
  if (!err.empty()) {
    std::cerr << err << '\n';
    return false;
  }

  // Loop over shapes
  for (auto &&shape : shapes) {
    // Loop over faces(polygon)
    size_t index_offset = 0;
    for (size_t f = 0; f != shape.mesh.num_face_vertices.size(); ++f) {
      // Hardcode loading to triangles
      size_t fv = 3;
      // Loop over vertices in the face
      for (size_t v = 0; v != fv; ++v) {
        // Access to vertex
        tinyobj::index_t idx = shape.mesh.indices[index_offset + v];

        // Vertex position
        auto vc = glm::vec3();
        for (int i = 0; i != 3; ++i) {
          vc[i] = attrib.vertices[3 * idx.vertex_index + i];
        }

        // Vertex normal
        auto nc = glm::vec3();
        for (int i = 0; i != 3; ++i) {
          nc[i] = attrib.normals[3 * idx.normal_index + i];
        }

        // We are setting the vertex color as the vertex normal. This is just
        // for display purposes
        Vertex new_vert = {.position = vc, .normal = nc, .color = nc};

        _vertices.push_back(new_vert);
      }
      index_offset += fv;
    }
  }
  return true;
}
