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

#include "nanodet.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "cpu.h"

NanoDet::NanoDet()
{
    blob_pool_allocator.set_size_compare_ratio(0.f);
    workspace_pool_allocator.set_size_compare_ratio(0.f);
}

int NanoDet::load(const char* modeltype, int _target_size, const float* _mean_vals, const float* _norm_vals, bool use_gpu)
{
    lsdnet.clear();
    blob_pool_allocator.clear();
    workspace_pool_allocator.clear();

    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

    lsdnet.opt = ncnn::Option();

#if NCNN_VULKAN
    lsdnet.opt.use_vulkan_compute = use_gpu;
#endif

    lsdnet.opt.num_threads = ncnn::get_big_cpu_count();
    lsdnet.opt.blob_allocator = &blob_pool_allocator;
    lsdnet.opt.workspace_allocator = &workspace_pool_allocator;

    char parampath[256];
    char modelpath[256];
    sprintf(parampath, "%s.param", modeltype);
    sprintf(modelpath, "%s.bin", modeltype);

    lsdnet.load_param(parampath);
    lsdnet.load_model(modelpath);

    mean_vals[0] = _mean_vals[0];
    mean_vals[1] = _mean_vals[1];
    mean_vals[2] = _mean_vals[2];
    norm_vals[0] = _norm_vals[0];
    norm_vals[1] = _norm_vals[1];
    norm_vals[2] = _norm_vals[2];

    return 0;
}

int NanoDet::load(AAssetManager* mgr, const char* modeltype, int _target_size, const float* _mean_vals, const float* _norm_vals, bool use_gpu)
{
    lsdnet.clear();
    blob_pool_allocator.clear();
    workspace_pool_allocator.clear();

    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

    lsdnet.opt = ncnn::Option();

#if NCNN_VULKAN
    lsdnet.opt.use_vulkan_compute = use_gpu;
#endif

    lsdnet.opt.num_threads = ncnn::get_big_cpu_count();
    lsdnet.opt.blob_allocator = &blob_pool_allocator;
    lsdnet.opt.workspace_allocator = &workspace_pool_allocator;
    char parampath[256];
    char modelpath[256];
    sprintf(parampath, "%s.param", modeltype);
    sprintf(modelpath, "%s.bin", modeltype);

    lsdnet.load_param(mgr,parampath);
    lsdnet.load_model(mgr,modelpath);
    target_size = _target_size;
    mean_vals[0] = _mean_vals[0];
    mean_vals[1] = _mean_vals[1];
    mean_vals[2] = _mean_vals[2];
    norm_vals[0] = _norm_vals[0];
    norm_vals[1] = _norm_vals[1];
    norm_vals[2] = _norm_vals[2];

    return 0;
}

int NanoDet::detect(const cv::Mat& rgb, std::vector<LinePt>& linepoints, int topk, float score_threshold, float dist_threshold)
{
    int out_size = target_size / 2;
    ncnn::Extractor ex = lsdnet.create_extractor();
    ncnn::Mat ncnn_in = ncnn::Mat::from_pixels_resize(rgb.data,ncnn::Mat::PIXEL_RGB2BGRA, rgb.cols, rgb.rows, target_size, target_size);
    ncnn_in.substract_mean_normalize(0, norm_vals);

    ex.input("input", ncnn_in);

    ncnn::Mat org_disp_map, max_map, center_map;

    ex.extract("out1", org_disp_map);
    ex.extract("Decoder/Sigmoid_4:0", center_map);
    ex.extract("out2", max_map);

    float* max_map_data = (float*)max_map.data;
    float* center_map_data = (float*)center_map.data;
    std::vector<std::pair<float, int>> sort_result(max_map.total());
    for (int i = 0; i < max_map.total(); i++)
    {
        if (max_map_data[i] == center_map_data[i])
        {
            sort_result[i] = std::pair<float, int>(max_map_data[i],i);
        }
    }
    std::partial_sort(sort_result.begin(), sort_result.begin() + topk, sort_result.end(), std::greater<std::pair<float, int> >());

    std::vector<std::pair<int, int>>topk_pts;
    
    for (int i = 0; i < topk; i++)
    {
        int x = sort_result[i].second % out_size;
        int y = sort_result[i].second / out_size;
        topk_pts.push_back(std::pair<int, int>(x, y));
    }

    ncnn::Mat start_map = org_disp_map.channel_range(0, 2).clone();
    ncnn::Mat end_map = org_disp_map.channel_range(2, 2).clone();
    ncnn::Mat dist_map = ncnn::Mat(out_size, out_size, 1);
    float* start_map_data = (float*)start_map.data;
    float* end_map_data = (float*)end_map.data;
    for (int i = 0; i < start_map.total(); i++)
    {
        start_map_data[i] = (start_map_data[i] - end_map_data[i]) * (start_map_data[i] - end_map_data[i]);
    }
    float* dist_map_data = (float*)dist_map.data;
    for (int i = 0; i < start_map.total()/2; i++)
    {
        dist_map_data[i] = std::sqrt(start_map_data[i] + start_map_data[i + start_map.channel(0).total()]);
        
    }

    for (int i = 0; i < topk_pts.size(); i++)
    {
        int x = topk_pts[i].first;
        int y = topk_pts[i].second;

        float distance = dist_map_data[y * out_size + x];

        if (sort_result[i].first > score_threshold && distance > dist_threshold)
        {
            int disp_x_start = org_disp_map.channel(0)[y * out_size + x];
            int disp_y_start = org_disp_map.channel(1)[y * out_size + x];
            int disp_x_end = org_disp_map.channel(2)[y * out_size + x];
            int disp_y_end = org_disp_map.channel(3)[y * out_size + x];

            int x_start = std::max(std::min((int)((x + disp_x_start) * 2), target_size), 0);
            int y_start = std::max(std::min((int)((y + disp_y_start) * 2), target_size), 0);
            int x_end = std::max(std::min((int)((x + disp_x_end ) * 2), target_size), 0);
            int y_end = std::max(std::min((int)((y + disp_y_end ) * 2), target_size), 0);
            linepoints.push_back(LinePt{ x_start, x_end, y_start, y_end });
        }
    }
    return 0;
}

int NanoDet::draw(cv::Mat& rgb, const std::vector<LinePt>& linepoints)
{
    float h_ratio = (float)rgb.rows / target_size;
    float w_ratio = (float)rgb.cols / target_size;
    for (int i = 0; i < linepoints.size(); i++)
    {
        cv::line(rgb, cv::Point(linepoints[i].x_start*w_ratio, linepoints[i].y_start*h_ratio),
                 cv::Point(linepoints[i].x_end*w_ratio, linepoints[i].y_end*h_ratio), cv::Scalar(0, 0, 255), 2, 8);
    }
    return 0;
}
