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
/*
 * Profiler.cpp
 *
 *  Created on: May 9, 2013
 *      Author: danielh
 */

#include "zoo_vision/profiler.hpp"

namespace zoo {

std::unique_ptr<Profiler> Profiler::gInstance;
thread_local std::stack<ProfilerSectionData *> *Profiler::activeStack_ = nullptr;

} // namespace zoo