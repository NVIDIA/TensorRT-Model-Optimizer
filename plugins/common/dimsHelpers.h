/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#ifndef TRT_PLUGIN_DIMS_HELPERS_H
#define TRT_PLUGIN_DIMS_HELPERS_H

#include "common/plugin.h" // purely for assertions

#include <algorithm> // all of
#include <functional>
#include <numeric>

namespace nvinfer1 {

namespace pluginInternal {

//! Return number of elements in the given dimensions in the range [start,
//! stop). Does not include padding added for vectorized formats.
//!
//! \param dims dimensions whose partial volume needs to be computed
//! \param start inclusive start axis
//! \param stop exclusive stop axis
//!
//! Expects 0 <= start <= stop <= dims.nbDims.
//! For i in the range [start,stop), dims.d[i] must be non-negative.
//!
inline int64_t volume(Dims const &dims, int32_t start, int32_t stop) {
  // The signature is int32_t start (and not uint32_t start) because int32_t is
  // used for indexing everywhere
  ASSERT_PARAM(start >= 0);
  ASSERT_PARAM(start <= stop);
  ASSERT_PARAM(stop <= dims.nbDims);
  ASSERT_PARAM(std::all_of(dims.d + start, dims.d + stop, [](int32_t x) { return x >= 0; }));
  return std::accumulate(dims.d + start, dims.d + stop, int64_t{1}, std::multiplies<int64_t>{});
}

//! Shorthand for volume(dims, 0, dims.nbDims).
inline int64_t volume(Dims const &dims) { return volume(dims, 0, dims.nbDims); }

} // namespace pluginInternal

} // namespace nvinfer1

#endif // TRT_PLUGIN_DIMS_HELPERS_H
