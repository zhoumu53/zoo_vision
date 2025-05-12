

// This file is part of zoo_vision.
//
// zoo_vision is free software: you can redistribute it and/or modify it under
// the terms of the GNU General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your option) any later
// version.
//
// zoo_vision is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
// A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License along with
// zoo_vision. If not, see <https://www.gnu.org/licenses/>.

#include "zoo_vision/image_embedder.hpp"
#include "zoo_vision/utils.hpp"

#include <torch/torch.h>

namespace zoo {

ImageEmbedder::ImageEmbedder() {
  at::InferenceMode inferenceGuard;

  readConfig(getConfig());
}

void ImageEmbedder::readConfig(const nlohmann::json &config) { // Load model
  const std::filesystem::path modelPath = std::filesystem::canonical(getDataPath() / config["models"]["embeddings"]);
  loadModel(modelPath);
}
void ImageEmbedder::loadModel(const std::filesystem::path &modelPath) {
  try {
    if (!std::filesystem::exists(modelPath)) {
      throw std::runtime_error("Model does not exist");
    }
    module_ = torch::jit::load(modelPath, torch::kCUDA);
    module_.eval();
  } catch (const std::exception &ex) {
    std::cout << "Error loading model from " << modelPath << std::endl;
    std::cout << "Exception: " << ex.what() << std::endl;
    std::terminate();
  }
}

at::Tensor ImageEmbedder::embed(const at::Tensor &images_f32) {
  at::InferenceMode inferenceGuard;
  c10::IValue modelResult = module_.forward({images_f32});
  return modelResult.toTensor();
}

} // namespace zoo