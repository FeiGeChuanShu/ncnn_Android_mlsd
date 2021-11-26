// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#ifndef NANODET_H
#define NANODET_H

#include <opencv2/core/core.hpp>

#include <net.h>

struct LinePt
{
    int x_start;
    int x_end;
    int y_start;
    int y_end;
};

class NanoDet
{
public:
    NanoDet();

    int load(const char* modeltype, int _target_size, const float* mean_vals, const float* norm_vals, bool use_gpu = false);

    int load(AAssetManager* mgr, const char* modeltype, int _target_size, const float* mean_vals, const float* norm_vals, bool use_gpu = false);

    int detect(const cv::Mat& rgb, std::vector<LinePt>& linepoints, int topk = 200, float score_threshold = 0.1f, float dist_threshold = 20.0f);
    int draw(cv::Mat& rgb, const std::vector<LinePt>& linepoints);

private:

    ncnn::Net lsdnet;
    int target_size;
    float mean_vals[3];
    float norm_vals[3];


    ncnn::UnlockedPoolAllocator blob_pool_allocator;
    ncnn::PoolAllocator workspace_pool_allocator;
};

#endif // NANODET_H
