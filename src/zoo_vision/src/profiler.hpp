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

#include "zoo_vision/types.hpp"

#include <cassert>
#include <chrono>
#include <iomanip>
#include <map>
#include <memory>
#include <mutex>
#include <ostream>
#include <stack>
#include <string>
#include <thread>
#include <vector>

#define ENABLE_PROFILER

namespace zoo {

using ProfilerClock = std::chrono::high_resolution_clock;

class ProfilerSectionData {
public:
  inline ProfilerSectionData(ProfilerSectionData *parent, std::string key) : parent_{parent}, key_{std::move(key)} {}

  ProfilerSectionData *getParent() const { return parent_; }
  const std::string &getKey() const { return key_; }

  inline ProfilerClock::duration getDuration() const {
    if (sampleCount_ == 0) {
      return ProfilerClock::duration{0};
    } else {
      return totalDuration_ / sampleCount_;
    }
  }
  int getSampleCount() const { return sampleCount_; }

  void tictoc(ProfilerClock::duration duration) {
    totalDuration_ += duration;
    sampleCount_++;
  }

  const std::map<std::string, std::unique_ptr<ProfilerSectionData>> &getChildren() const { return childSections_; }
  std::map<std::string, std::unique_ptr<ProfilerSectionData>> &getChildren() { return childSections_; }

  inline ProfilerSectionData &getOrMakeChild(const std::string &subkey) {
    auto it = childSections_.find(subkey);
    if (it == childSections_.end()) {
      auto newIt = childSections_.emplace(subkey, std::make_unique<ProfilerSectionData>(this, subkey));
      return *newIt.first->second;
    } else
      return *it->second;
  }

  void reset() {
    totalDuration_ = std::chrono::nanoseconds{0};
    sampleCount_ = 0;
    for (auto &[subkey, child] : childSections_) {
      child->reset();
    }
  }

  template <typename T>
  inline T &logStats(T &stream, const std::string &prefix, const ProfilerClock::duration parentTime);

protected:
  ProfilerSectionData *parent_;
  std::string key_;
  std::optional<ProfilerClock::time_point> lastTic_;

  ProfilerClock::duration totalDuration_ = std::chrono::nanoseconds{0};
  int sampleCount_ = 0;

  std::map<std::string, std::unique_ptr<ProfilerSectionData>> childSections_;
};

/**
 * @brief Class to keep track of runtimes for different sections of code (i.e. runtime profiling).
 * Can be used directly with tic(string) and toc(string), or through the ProfileSection() class. All
 * profiling code is removed if ENABLE_PROFILER is not defined.
 */
class Profiler {
public:
  inline static Profiler &Instance();

  /// Needs to be called before any sections are pushed.
  void setActiveStack(std::stack<ProfilerSectionData *> *activeStack) { activeStack_ = activeStack; }
  std::stack<ProfilerSectionData *> *getActiveStack() const { return activeStack_; }

  inline ProfilerSectionData &pushActiveSection(const std::string &sectionKey);
  inline void popActiveSection(const ProfilerClock::duration duration);

  void showTotals(bool value) { showTotals_ = value; }
  bool isShowingTotals() const { return showTotals_; }

  template <typename T> inline T &logStats(T &stream);

  inline void reset();

protected:
  std::mutex mutex_;
  std::map<std::string, std::unique_ptr<ProfilerSectionData>> topSections_;

  bool showTotals_ = false;

  static thread_local std::stack<ProfilerSectionData *> *activeStack_;

  /// Singleton instance
  static std::unique_ptr<Profiler> gInstance;

  /// Singleton class, hidden constructor.
  Profiler() = default;
};

template <typename T> T &operator<<(T &stream, Profiler &profiler) { return profiler.logStats(stream); }

class ProfileStackGuard {
public:
  ProfileStackGuard(std::stack<ProfilerSectionData *> &stack) {
    oldStack_ = Profiler::Instance().getActiveStack();
    Profiler::Instance().setActiveStack(&stack);
  }
  ~ProfileStackGuard() { Profiler::Instance().setActiveStack(oldStack_); }

private:
  std::stack<ProfilerSectionData *> *oldStack_;
};

/**
 * @brief Used to profile a section by just constructing and destructing this object.
 */
class ProfileSection {
public:
#ifdef ENABLE_PROFILER
  ProfileSection(const std::string &sectionKey) {
    start_ = ProfilerClock::now();
    Profiler::Instance().pushActiveSection(sectionKey);
  }
  ~ProfileSection() {
    const auto duration = ProfilerClock::now() - start_;
    Profiler::Instance().popActiveSection(duration);
  }

private:
  ProfilerClock::time_point start_;
#else
  ProfileSection(const std::string & /*sectionKey*/) {}
#endif
};

class ProfileTicOnly {
public:
#ifdef ENABLE_PROFILER
  ProfileTicOnly(const std::string &sectionKey) : sectionKey_{sectionKey} {}

  void tic() {
    const auto now = ProfilerClock::now();
    if (lastTic_.has_value()) {
      const auto duration = now - *lastTic_;
      Profiler::Instance().popActiveSection(duration);
    }
    lastTic_ = now;
    Profiler::Instance().pushActiveSection(sectionKey_);
  }

private:
  std::string sectionKey_;
  std::optional<ProfilerClock::time_point> lastTic_;

#else
  ProfileSection(const std::string & /*sectionKey*/) {}
  void tic() {}
#endif
};

////////////////////////////////////////////////////////////////////////////////////////////////////
/// Implementation
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
T &ProfilerSectionData::logStats(T &stream, const std::string &prefix, const ProfilerClock::duration parentTime) {
  const bool showTotals = Profiler::Instance().isShowingTotals();

  if (showTotals) {
    const auto time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(totalDuration_).count();
    stream << prefix << key_ << ": " << time_ms << "ms";
    if (parentTime.count() > 0) {
      const int percent = static_cast<int>(100 * totalDuration_.count() / parentTime.count());
      stream << " (" << percent << "%)";
    }
    stream << "\n";

    ProfilerClock::duration totalChildrenTime{0};
    const std::string newPrefix = prefix + "-";
    for (const auto &[subkey, child] : childSections_) {
      child->logStats(stream, newPrefix, totalDuration_);
      totalChildrenTime += child->totalDuration_;
    }

    if (totalDuration_.count() > 0 && childSections_.size() > 1) {
      const ProfilerClock::duration otherTime = totalDuration_ - totalChildrenTime;
      const int otherPercent = 100 * otherTime.count() / totalDuration_.count();
      if (otherPercent > 0) {
        const auto otherTime_ms = std::chrono::duration_cast<std::chrono::milliseconds>(otherTime).count();
        stream << newPrefix << "other: " << otherTime_ms << "ms (" << otherPercent << "%)\n";
      }
    }
  } else {
    const auto time = getDuration();
    const int time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(time).count();
    stream << prefix << key_ << ": " << std::fixed << std::setprecision(2) << time_ms << "ms";
    stream << " x " << sampleCount_;
    stream << "\n";

    const std::string newPrefix = prefix + "-";
    for (const auto &[subkey, child] : childSections_) {
      child->logStats(stream, newPrefix, time);
    }
  }
  return stream;
}

///////////////////////////////////////////////////////////////////////////////////////

ProfilerSectionData &Profiler::pushActiveSection(const std::string &sectionKey) {
  // Need to lock all profiling because two threads might log the same section
  std::lock_guard<std::mutex> guard{mutex_};

  CHECK_NOT_NULL(activeStack_);
  auto &activeSections = *activeStack_;

  ProfilerSectionData &section = [&]() -> ProfilerSectionData & {
    if (activeSections.empty()) {
      // Nothing active, try to find the section in the top levels
      auto it = topSections_.find(sectionKey);
      if (it == topSections_.end()) {
        // Doesn't exist, add a new top level section
        auto [newIt, isNew] =
            topSections_.emplace(sectionKey, std::make_unique<ProfilerSectionData>(nullptr, sectionKey));
        CHECK_TRUE(isNew);
        return *newIt->second;
      } else {
        return *it->second;
      }
    } else {
      // Try to find the section as a child of the active section
      ProfilerSectionData &activeSection = *activeSections.top();
      return activeSection.getOrMakeChild(sectionKey);
    }
  }();

  activeSections.push(&section);
  return section;
}

void Profiler::popActiveSection(const ProfilerClock::duration duration) {
  // Need to lock all profiling because two threads might log the same section
  std::lock_guard<std::mutex> guard{mutex_};

  CHECK_NOT_NULL(activeStack_);
  auto &activeSections = *activeStack_;

  // Get tic time
  CHECK_FALSE(activeSections.empty());
  activeSections.top()->tictoc(duration);
  activeSections.pop();
}

Profiler &Profiler::Instance() {
  if (!gInstance.get()) {
    gInstance.reset(new Profiler());
  }
  return *gInstance;
}

template <typename T> T &Profiler::logStats(T &stream) {
#ifdef ENABLE_PROFILER
  std::lock_guard<std::mutex> lock(mutex_);

  for (auto &[key, section] : topSections_) {
    section->logStats(stream, "", ProfilerClock::duration{0});
  }
#endif
  return stream;
}

void Profiler::reset() {
  std::lock_guard<std::mutex> lock(mutex_);

  for (auto &[key, section] : topSections_) {
    section->reset();
  }
}

} // namespace zoo
