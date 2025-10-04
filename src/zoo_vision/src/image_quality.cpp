

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

#include "zoo_vision/image_quality.hpp"
#include "zoo_vision/utils.hpp"

#include <torch/torch.h>

#include <ranges>

namespace zoo {

ImageQualityNet::ImageQualityNet() { readConfig(getConfig()); }

void ImageQualityNet::readConfig(const nlohmann::json &config) { // Load model
  const auto configPath = config["models"]["quality"].get<std::string>();
  if (!configPath.empty()) {
    const std::filesystem::path modelPath = std::filesystem::canonical(getDataPath() / configPath);
    loadModel(modelPath);
  }
}
void ImageQualityNet::loadModel(const std::filesystem::path &modelPath) {
  try {
    if (!std::filesystem::exists(modelPath)) {
      throw ZooVisionError("Model does not exist");
    }
    module_.emplace(torch::jit::load(modelPath, torch::kCUDA));
    module_->eval();
  } catch (const std::exception &ex) {
    std::cout << "Error loading model from " << modelPath << std::endl;
    std::cout << "Exception: " << ex.what() << std::endl;
    std::terminate();
  }
}

std::vector<bool> ImageQualityNet::check(const at::Tensor &images_f32) {
  if (!module_.has_value()) {
    // No quality check
    return std::vector<bool>(images_f32.size(0), true);
  }

  at::InferenceMode inferenceGuard;
  const c10::IValue modelResult = module_->forward({images_f32});
  const at::Tensor logitsGpu = [&]() {
    if (modelResult.isTensor()) {
      return modelResult.toTensor();
    } else {
      auto dict = modelResult.toGenericDict();
      auto value = dict.at("logits").toTensor();
      return value.squeeze(1); // Remove dummy time dimension
    }
  }();
  const at::Tensor logits = logitsGpu.cpu();
  const at::Tensor probs = torch::nn::functional::softmax(logits, torch::nn::functional::SoftmaxFuncOptions(/*dim*/ 1));

  std::vector<bool> passesCheck;
  for (const auto i : std::views::iota(int64_t(0), probs.size(0))) {
    const auto pi = probs[i];
    const auto pBad = pi[0].item<float32_t>();
    const auto pGood = pi[1].item<float32_t>();
    passesCheck.push_back(pGood > pBad);
  }
  return passesCheck;
}

} // namespace zoo