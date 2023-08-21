/*!
* Modified from Deformable DETR
*/

#include <cstdio>
#include <algorithm>
#include <cstring>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <THC/THCAtomics.cuh>

#define CUDA_KERNEL_LOOP(i, n)                          \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;   \
      i < (n);                                          \
      i += blockDim.x * gridDim.x)

#define CUDA_NUM_THREADS 512
#define MAX_POINT 32

inline int GET_BLOCKS(const int N, const int num_threads) {
    return (N + num_threads - 1) / num_threads;
}

__device__ float ms_deform_attn_im2col_bilinear(
    const float*& bottom_data,
    const int& height, const int& width, const int& channels,
    const float& h, const float& w, const int& c) {

    const int h_low = floor(h);
    const int w_low = floor(w);
    const int h_high = h_low + 1;
    const int w_high = w_low + 1;

    const float lh = h - h_low;
    const float lw = w - w_low;
    const float hh = 1 - lh, hw = 1 - lw;

    const int w_stride = channels;
    const int h_stride = width * w_stride;
    const int h_low_ptr_offset = h_low * h_stride;
    const int h_high_ptr_offset = h_low_ptr_offset + h_stride;
    const int w_low_ptr_offset = w_low * w_stride;
    const int w_high_ptr_offset = w_low_ptr_offset + w_stride;

    float v1 = 0;
    if (h_low >= 0 && w_low >= 0) {
        const int ptr1 = h_low_ptr_offset + w_low_ptr_offset + c;
        v1 = bottom_data[ptr1];
    }
    float v2 = 0;
    if (h_low >= 0 && w_high <= width - 1) {
        const int ptr2 = h_low_ptr_offset + w_high_ptr_offset + c;
        v2 = bottom_data[ptr2];
    }
    float v3 = 0;
    if (h_high <= height - 1 && w_low >= 0) {
        const int ptr3 = h_high_ptr_offset + w_low_ptr_offset + c;
        v3 = bottom_data[ptr3];
    }
    float v4 = 0;
    if (h_high <= height - 1 && w_high <= width - 1) {
        const int ptr4 = h_high_ptr_offset + w_high_ptr_offset + c;
        v4 = bottom_data[ptr4];
    }

    const float w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;
    const float val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

    return val;
}

__global__ void ms_deformable_im2col_gpu_kernel_c2345(
    const float* feat_c2,
    const float* feat_c3,
    const float* feat_c4,
    const float* feat_c5,
    const int h_c2, const int w_c2,
    const int h_c3, const int w_c3,
    const int h_c4, const int w_c4,
    const int h_c5, const int w_c5,
    const float* data_sampling_loc,
    const float* data_attn_weight,
    const int batch_size,
    const int channels,
    const int num_views,
    const int num_query,
    const int num_point,
    float* data_col) {

    float res[MAX_POINT];

    CUDA_KERNEL_LOOP(index, batch_size * num_query * channels) {  // n: bs x query x channels
        int _temp = index;
        const int c_col = _temp % channels;
        _temp /= channels;
        const int sampling_index = _temp;
        _temp /= num_query;
        const int b_col = _temp;

        for (int p_col = 0; p_col < num_point; ++p_col) { res[p_col] = 0; }

        for (int p_col = 0; p_col < num_point; ++p_col) {
            // Sampling location in range [0, 1]
            int data_loc_ptr = sampling_index * num_point * 3 + p_col * 3;
            const float loc_w = data_sampling_loc[data_loc_ptr];
            const float loc_h = data_sampling_loc[data_loc_ptr + 1];
            const int loc_v = round(data_sampling_loc[data_loc_ptr + 2] * (num_views - 1));

            // Attn weights
            int data_weight_ptr = sampling_index * num_point * 4 + p_col * 4;
            const float weight_c2 = data_attn_weight[data_weight_ptr];
            const float weight_c3 = data_attn_weight[data_weight_ptr + 1];
            const float weight_c4 = data_attn_weight[data_weight_ptr + 2];
            const float weight_c5 = data_attn_weight[data_weight_ptr + 3];

            //const float h_im = loc_h * spatial_h - 0.5;  // align_corners = False
            //const float w_im = loc_w * spatial_w - 0.5;

            // C2 Feature
            float h_im = loc_h * (h_c2 - 1);  // align_corners = True
            float w_im = loc_w * (w_c2 - 1);

            if (h_im > -1 && w_im > -1 && h_im < h_c2 && w_im < w_c2) {
                const float* feat_c2_ptr = feat_c2 + b_col * num_views * h_c2 * w_c2 * channels + loc_v * h_c2 * w_c2 * channels;
                res[p_col] += ms_deform_attn_im2col_bilinear(feat_c2_ptr, h_c2, w_c2, channels, h_im, w_im, c_col) * weight_c2;
            }

            // C3 Feature
            h_im = loc_h * (h_c3 - 1);  // align_corners = True
            w_im = loc_w * (w_c3 - 1);

            if (h_im > -1 && w_im > -1 && h_im < h_c3 && w_im < w_c3) {
                const float* feat_c3_ptr = feat_c3 + b_col * num_views * h_c3 * w_c3 * channels + loc_v * h_c3 * w_c3 * channels;
                res[p_col] += ms_deform_attn_im2col_bilinear(feat_c3_ptr, h_c3, w_c3, channels, h_im, w_im, c_col) * weight_c3;
            }

            // C4 Feature
            h_im = loc_h * (h_c4 - 1);  // align_corners = True
            w_im = loc_w * (w_c4 - 1);

            if (h_im > -1 && w_im > -1 && h_im < h_c4 && w_im < w_c4) {
                const float* feat_c4_ptr = feat_c4 + b_col * num_views * h_c4 * w_c4 * channels + loc_v * h_c4 * w_c4 * channels;
                res[p_col] += ms_deform_attn_im2col_bilinear(feat_c4_ptr, h_c4, w_c4, channels, h_im, w_im, c_col) * weight_c4;
            }

            // C5 Feature
            h_im = loc_h * (h_c5 - 1);  // align_corners = True
            w_im = loc_w * (w_c5 - 1);

            if (h_im > -1 && w_im > -1 && h_im < h_c5 && w_im < w_c5) {
                const float* feat_c5_ptr = feat_c5 + b_col * num_views * h_c5 * w_c5 * channels + loc_v * h_c5 * w_c5 * channels;
                res[p_col] += ms_deform_attn_im2col_bilinear(feat_c5_ptr, h_c5, w_c5, channels, h_im, w_im, c_col) * weight_c5;
            }
        }

        for (int p_col = 0; p_col < num_point; ++p_col) {
            float* data_col_ptr = data_col + index * num_point + p_col;
            *data_col_ptr = res[p_col];
        }
    }
}

__global__ void ms_deformable_im2col_gpu_kernel_c23456(
    const float* feat_c2,
    const float* feat_c3,
    const float* feat_c4,
    const float* feat_c5,
    const float* feat_c6,
    const int h_c2, const int w_c2,
    const int h_c3, const int w_c3,
    const int h_c4, const int w_c4,
    const int h_c5, const int w_c5,
    const int h_c6, const int w_c6,
    const float* data_sampling_loc,
    const float* data_attn_weight,
    const int batch_size,
    const int channels,
    const int num_views,
    const int num_query,
    const int num_point,
    float* data_col) {

    float res[MAX_POINT];

    CUDA_KERNEL_LOOP(index, batch_size * num_query * channels) {  // n: bs x query x channels
        int _temp = index;
        const int c_col = _temp % channels;
        _temp /= channels;
        const int sampling_index = _temp;
        _temp /= num_query;
        const int b_col = _temp;

        for (int p_col = 0; p_col < num_point; ++p_col) { res[p_col] = 0; }

        for (int p_col = 0; p_col < num_point; ++p_col) {
            // Sampling location in range [0, 1]
            int data_loc_ptr = sampling_index * num_point * 3 + p_col * 3;
            const float loc_w = data_sampling_loc[data_loc_ptr];
            const float loc_h = data_sampling_loc[data_loc_ptr + 1];
            const int loc_v = round(data_sampling_loc[data_loc_ptr + 2] * (num_views - 1));

            // Attn weights
            int data_weight_ptr = sampling_index * num_point * 5 + p_col * 5;
            const float weight_c2 = data_attn_weight[data_weight_ptr];
            const float weight_c3 = data_attn_weight[data_weight_ptr + 1];
            const float weight_c4 = data_attn_weight[data_weight_ptr + 2];
            const float weight_c5 = data_attn_weight[data_weight_ptr + 3];
            const float weight_c6 = data_attn_weight[data_weight_ptr + 4];

            //const float h_im = loc_h * spatial_h - 0.5;  // align_corners = False
            //const float w_im = loc_w * spatial_w - 0.5;

            // C2 Feature
            float h_im = loc_h * (h_c2 - 1);  // align_corners = True
            float w_im = loc_w * (w_c2 - 1);

            if (h_im > -1 && w_im > -1 && h_im < h_c2 && w_im < w_c2) {
                const float* feat_c2_ptr = feat_c2 + b_col * num_views * h_c2 * w_c2 * channels + loc_v * h_c2 * w_c2 * channels;
                res[p_col] += ms_deform_attn_im2col_bilinear(feat_c2_ptr, h_c2, w_c2, channels, h_im, w_im, c_col) * weight_c2;
            }

            // C3 Feature
            h_im = loc_h * (h_c3 - 1);  // align_corners = True
            w_im = loc_w * (w_c3 - 1);

            if (h_im > -1 && w_im > -1 && h_im < h_c3 && w_im < w_c3) {
                const float* feat_c3_ptr = feat_c3 + b_col * num_views * h_c3 * w_c3 * channels + loc_v * h_c3 * w_c3 * channels;
                res[p_col] += ms_deform_attn_im2col_bilinear(feat_c3_ptr, h_c3, w_c3, channels, h_im, w_im, c_col) * weight_c3;
            }

            // C4 Feature
            h_im = loc_h * (h_c4 - 1);  // align_corners = True
            w_im = loc_w * (w_c4 - 1);

            if (h_im > -1 && w_im > -1 && h_im < h_c4 && w_im < w_c4) {
                const float* feat_c4_ptr = feat_c4 + b_col * num_views * h_c4 * w_c4 * channels + loc_v * h_c4 * w_c4 * channels;
                res[p_col] += ms_deform_attn_im2col_bilinear(feat_c4_ptr, h_c4, w_c4, channels, h_im, w_im, c_col) * weight_c4;
            }

            // C5 Feature
            h_im = loc_h * (h_c5 - 1);  // align_corners = True
            w_im = loc_w * (w_c5 - 1);

            if (h_im > -1 && w_im > -1 && h_im < h_c5 && w_im < w_c5) {
                const float* feat_c5_ptr = feat_c5 + b_col * num_views * h_c5 * w_c5 * channels + loc_v * h_c5 * w_c5 * channels;
                res[p_col] += ms_deform_attn_im2col_bilinear(feat_c5_ptr, h_c5, w_c5, channels, h_im, w_im, c_col) * weight_c5;
            }

            // C6 Feature
            h_im = loc_h * (h_c6 - 1);  // align_corners = True
            w_im = loc_w * (w_c6 - 1);

            if (h_im > -1 && w_im > -1 && h_im < h_c6 && w_im < w_c6) {
                const float* feat_c6_ptr = feat_c6 + b_col * num_views * h_c6 * w_c6 * channels + loc_v * h_c6 * w_c6 * channels;
                res[p_col] += ms_deform_attn_im2col_bilinear(feat_c6_ptr, h_c6, w_c6, channels, h_im, w_im, c_col) * weight_c6;
            }
        }

        for (int p_col = 0; p_col < num_point; ++p_col) {
            float* data_col_ptr = data_col + index * num_point + p_col;
            *data_col_ptr = res[p_col];
        }
    }
}

void ms_deformable_im2col_cuda_c2345(
    const float* feat_c2,
    const float* feat_c3,
    const float* feat_c4,
    const float* feat_c5,
    const int h_c2, const int w_c2,
    const int h_c3, const int w_c3,
    const int h_c4, const int w_c4,
    const int h_c5, const int w_c5,
    const float* data_sampling_loc,
    const float* data_attn_weight,
    const int batch_size,
    const int channels,
    const int num_views,
    const int num_query,
    const int num_point,
    float* data_col) {

    const int num_kernels = batch_size * num_query * channels;
    const int num_threads = CUDA_NUM_THREADS;

    ms_deformable_im2col_gpu_kernel_c2345 <<<GET_BLOCKS(num_kernels, num_threads), num_threads>>> (
        feat_c2, feat_c3, feat_c4, feat_c5, h_c2, w_c2, h_c3, w_c3, h_c4, w_c4, h_c5, w_c5,
        data_sampling_loc, data_attn_weight, batch_size, channels, num_views, num_query, num_point, data_col
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("error in ms_deformable_im2col_cuda_c2345: %s\n", cudaGetErrorString(err));
    }
}

void ms_deformable_im2col_cuda_c23456(
    const float* feat_c2,
    const float* feat_c3,
    const float* feat_c4,
    const float* feat_c5,
    const float* feat_c6,
    const int h_c2, const int w_c2,
    const int h_c3, const int w_c3,
    const int h_c4, const int w_c4,
    const int h_c5, const int w_c5,
    const int h_c6, const int w_c6,
    const float* data_sampling_loc,
    const float* data_attn_weight,
    const int batch_size,
    const int channels,
    const int num_views,
    const int num_query,
    const int num_point,
    float* data_col) {

    const int num_kernels = batch_size * num_query * channels;
    const int num_threads = CUDA_NUM_THREADS;

    ms_deformable_im2col_gpu_kernel_c23456 <<<GET_BLOCKS(num_kernels, num_threads), num_threads>>> (
        feat_c2, feat_c3, feat_c4, feat_c5, feat_c6, h_c2, w_c2, h_c3, w_c3, h_c4, w_c4, h_c5, w_c5, h_c6, w_c6,
        data_sampling_loc, data_attn_weight, batch_size, channels, num_views, num_query, num_point, data_col
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("error in ms_deformable_im2col_cuda_c23456: %s\n", cudaGetErrorString(err));
    }
}
