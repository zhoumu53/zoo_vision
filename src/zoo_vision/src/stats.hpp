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
#pragma once

#include <cmath>
#include <deque>

namespace zoo {
class RunningStats {
public:
  float mean() const { return mean_; }
  float var() const {
    if (n_ <= 1) {
      return 0.0f;
    }
    return varN_ / (n_ - 1);
  }
  float std() const { return std::sqrt(var()); }

  void push(const float value) {
    n_ += 1;
    if (n_ == 1) {
      mean_ = value;
      varN_ = 0.0f;
    } else {
      float newMean = mean_ + (value - mean_) / n_;
      float newVarN = varN_ + (value - mean_) * (value - newMean);
      mean_ = newMean;
      varN_ = newVarN;
    }
  }

  void pop(const float value) {
    if (n_ == 1) {
      mean_ = 0.0;
      varN_ = 0.0f;
    } else {
      float newMean = (n_ * mean_ - value) / (n_ - 1);
      float newVarN = varN_ - (value - mean_) * (value - newMean);
      mean_ = newMean;
      varN_ = newVarN;
    }
    n_ -= 1;
  }

  void clear() {
    n_ = 0;
    mean_ = 0;
    varN_ = 0;
  }

private:
  size_t n_ = 0;
  float mean_ = 0;
  float varN_ = 0;
};

class BufferedStats {
public:
  BufferedStats(size_t capacity) : capacity_{capacity} {}

  float mean() const { return stats_.mean(); }
  float var() const { return stats_.var(); }
  float std() const { return stats_.std(); }

  void push(float value) {
    if (samples_.size() == capacity_) {
      stats_.pop(samples_.front());
      samples_.pop_front();
    }

    samples_.push_back(value);
    stats_.push(value);
  }

  size_t sampleCount() const { return samples_.size(); }

private:
  size_t capacity_;
  std::deque<float> samples_;
  RunningStats stats_;
};
} // namespace zoo