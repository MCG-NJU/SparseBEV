#include "msmv_sampling.h"

#define MAX_POINT 32

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
    float* data_col
);

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
    float* data_col
);

void ms_deformable_col2im_cuda_c2345(
    const float* grad_col,
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
    float* grad_value_c2,
    float* grad_value_c3,
    float* grad_value_c4,
    float* grad_value_c5,
    float* grad_sampling_loc,
    float* grad_attn_weight
);

void ms_deformable_col2im_cuda_c23456(
    const float *grad_col,
    const float *feat_c2,
    const float *feat_c3,
    const float *feat_c4,
    const float *feat_c5,
    const float *feat_c6,
    const int h_c2, const int w_c2,
    const int h_c3, const int w_c3,
    const int h_c4, const int w_c4,
    const int h_c5, const int w_c5,
    const int h_c6, const int w_c6,
    const float *data_sampling_loc,
    const float *data_attn_weight,
    const int batch_size,
    const int channels,
    const int num_views,
    const int num_query,
    const int num_point,
    float *grad_value_c2,
    float *grad_value_c3,
    float *grad_value_c4,
    float *grad_value_c5,
    float *grad_value_c6,
    float *grad_sampling_loc,
    float *grad_attn_weight
);

at::Tensor ms_deform_attn_cuda_c2345_forward(
    const at::Tensor& feat_c2,  // [B, N, H, W, C]
    const at::Tensor& feat_c3,  // [B, N, H, W, C]
    const at::Tensor& feat_c4,  // [B, N, H, W, C]
    const at::Tensor& feat_c5,  // [B, N, H, W, C]
    const at::Tensor& sampling_loc,  // [B, Q, P, 3]
    const at::Tensor& attn_weight  // [B, Q, P, 4]
    ) {
    AT_ASSERTM(feat_c2.is_contiguous(), "value tensor has to be contiguous");
    AT_ASSERTM(feat_c3.is_contiguous(), "value tensor has to be contiguous");
    AT_ASSERTM(feat_c4.is_contiguous(), "value tensor has to be contiguous");
    AT_ASSERTM(feat_c5.is_contiguous(), "value tensor has to be contiguous");
    AT_ASSERTM(sampling_loc.is_contiguous(), "sampling_loc tensor has to be contiguous");
    AT_ASSERTM(attn_weight.is_contiguous(), "attn_weight tensor has to be contiguous");

    AT_ASSERTM(feat_c2.is_cuda(), "value must be a CUDA tensor");
    AT_ASSERTM(feat_c3.is_cuda(), "value must be a CUDA tensor");
    AT_ASSERTM(feat_c4.is_cuda(), "value must be a CUDA tensor");
    AT_ASSERTM(feat_c5.is_cuda(), "value must be a CUDA tensor");
    AT_ASSERTM(sampling_loc.is_cuda(), "sampling_loc must be a CUDA tensor");
    AT_ASSERTM(attn_weight.is_cuda(), "attn_weight must be a CUDA tensor");

    const int batch_size = feat_c2.size(0);
    const int num_views = feat_c2.size(1);
    const int channels = feat_c2.size(4);
    const int num_query = sampling_loc.size(1);
    const int num_point = sampling_loc.size(2);
    AT_ASSERTM(num_point <= MAX_POINT, "num_point exceed limits");

    const int h_c2 = feat_c2.size(2);
    const int w_c2 = feat_c2.size(3);
    const int h_c3 = feat_c3.size(2);
    const int w_c3 = feat_c3.size(3);
    const int h_c4 = feat_c4.size(2);
    const int w_c4 = feat_c4.size(3);
    const int h_c5 = feat_c5.size(2);
    const int w_c5 = feat_c5.size(3);

    auto output = at::zeros({ batch_size, num_query, channels, num_point }, feat_c2.options());
    ms_deformable_im2col_cuda_c2345(
        feat_c2.data_ptr<float>(),
        feat_c3.data_ptr<float>(),
        feat_c4.data_ptr<float>(),
        feat_c5.data_ptr<float>(),
        h_c2, w_c2, h_c3, w_c3, h_c4, w_c4, h_c5, w_c5,
        sampling_loc.data_ptr<float>(),
        attn_weight.data_ptr<float>(),
        batch_size, channels, num_views, num_query, num_point,
        output.data_ptr<float>()
    );

    return output;
}

at::Tensor ms_deform_attn_cuda_c23456_forward(
    const at::Tensor& feat_c2,  // [B, N, H, W, C]
    const at::Tensor& feat_c3,  // [B, N, H, W, C]
    const at::Tensor& feat_c4,  // [B, N, H, W, C]
    const at::Tensor& feat_c5,  // [B, N, H, W, C]
    const at::Tensor& feat_c6,  // [B, N, H, W, C]
    const at::Tensor& sampling_loc,  // [B, Q, P, 3]
    const at::Tensor& attn_weight  // [B, Q, P, 4]
    ) {
    AT_ASSERTM(feat_c2.is_contiguous(), "value tensor has to be contiguous");
    AT_ASSERTM(feat_c3.is_contiguous(), "value tensor has to be contiguous");
    AT_ASSERTM(feat_c4.is_contiguous(), "value tensor has to be contiguous");
    AT_ASSERTM(feat_c5.is_contiguous(), "value tensor has to be contiguous");
    AT_ASSERTM(feat_c6.is_contiguous(), "value tensor has to be contiguous");
    AT_ASSERTM(sampling_loc.is_contiguous(), "sampling_loc tensor has to be contiguous");
    AT_ASSERTM(attn_weight.is_contiguous(), "attn_weight tensor has to be contiguous");

    AT_ASSERTM(feat_c2.is_cuda(), "value must be a CUDA tensor");
    AT_ASSERTM(feat_c3.is_cuda(), "value must be a CUDA tensor");
    AT_ASSERTM(feat_c4.is_cuda(), "value must be a CUDA tensor");
    AT_ASSERTM(feat_c5.is_cuda(), "value must be a CUDA tensor");
    AT_ASSERTM(feat_c6.is_cuda(), "value must be a CUDA tensor");
    AT_ASSERTM(sampling_loc.is_cuda(), "sampling_loc must be a CUDA tensor");
    AT_ASSERTM(attn_weight.is_cuda(), "attn_weight must be a CUDA tensor");

    const int batch_size = feat_c2.size(0);
    const int num_views = feat_c2.size(1);
    const int channels = feat_c2.size(4);
    const int num_query = sampling_loc.size(1);
    const int num_point = sampling_loc.size(2);
    AT_ASSERTM(num_point <= MAX_POINT, "num_point exceed limits");

    const int h_c2 = feat_c2.size(2);
    const int w_c2 = feat_c2.size(3);
    const int h_c3 = feat_c3.size(2);
    const int w_c3 = feat_c3.size(3);
    const int h_c4 = feat_c4.size(2);
    const int w_c4 = feat_c4.size(3);
    const int h_c5 = feat_c5.size(2);
    const int w_c5 = feat_c5.size(3);
    const int h_c6 = feat_c6.size(2);
    const int w_c6 = feat_c6.size(3);

    auto output = at::zeros({ batch_size, num_query, channels, num_point }, feat_c2.options());
    ms_deformable_im2col_cuda_c23456(
        feat_c2.data_ptr<float>(),
        feat_c3.data_ptr<float>(),
        feat_c4.data_ptr<float>(),
        feat_c5.data_ptr<float>(),
        feat_c6.data_ptr<float>(),
        h_c2, w_c2, h_c3, w_c3, h_c4, w_c4, h_c5, w_c5, h_c6, w_c6,
        sampling_loc.data_ptr<float>(),
        attn_weight.data_ptr<float>(),
        batch_size, channels, num_views, num_query, num_point,
        output.data_ptr<float>()
    );

    return output;
}

std::vector<at::Tensor> ms_deform_attn_cuda_c2345_backward(
    const at::Tensor& grad_output,
    const at::Tensor& feat_c2,  // [B, N, H, W, C]
    const at::Tensor& feat_c3,  // [B, N, H, W, C]
    const at::Tensor& feat_c4,  // [B, N, H, W, C]
    const at::Tensor& feat_c5,  // [B, N, H, W, C]
    const at::Tensor& sampling_loc,  // [B, Q, P, 3]
    const at::Tensor& attn_weight  // [B, Q, P, 4]
    ) {
    AT_ASSERTM(feat_c2.is_contiguous(), "value tensor has to be contiguous");
    AT_ASSERTM(feat_c3.is_contiguous(), "value tensor has to be contiguous");
    AT_ASSERTM(feat_c4.is_contiguous(), "value tensor has to be contiguous");
    AT_ASSERTM(feat_c5.is_contiguous(), "value tensor has to be contiguous");
    AT_ASSERTM(sampling_loc.is_contiguous(), "sampling_loc tensor has to be contiguous");
    AT_ASSERTM(attn_weight.is_contiguous(), "attn_weight tensor has to be contiguous");
    AT_ASSERTM(grad_output.is_contiguous(), "grad_output tensor has to be contiguous");

    AT_ASSERTM(feat_c2.is_cuda(), "value must be a CUDA tensor");
    AT_ASSERTM(feat_c3.is_cuda(), "value must be a CUDA tensor");
    AT_ASSERTM(feat_c4.is_cuda(), "value must be a CUDA tensor");
    AT_ASSERTM(feat_c5.is_cuda(), "value must be a CUDA tensor");
    AT_ASSERTM(sampling_loc.is_cuda(), "sampling_loc must be a CUDA tensor");
    AT_ASSERTM(attn_weight.is_cuda(), "attn_weight must be a CUDA tensor");
    AT_ASSERTM(grad_output.is_cuda(), "grad_output must be a CUDA tensor");

    const int batch_size = feat_c2.size(0);
    const int num_views = feat_c2.size(1);
    const int channels = feat_c2.size(4);
    const int num_query = sampling_loc.size(1);
    const int num_point = sampling_loc.size(2);
    AT_ASSERTM(num_point <= MAX_POINT, "num_point exceed limits");

    auto grad_value_c2 = at::zeros_like(feat_c2);
    auto grad_value_c3 = at::zeros_like(feat_c3);
    auto grad_value_c4 = at::zeros_like(feat_c4);
    auto grad_value_c5 = at::zeros_like(feat_c5);
    auto grad_sampling_loc = at::zeros_like(sampling_loc);
    auto grad_attn_weight = at::zeros_like(attn_weight);

    const int h_c2 = feat_c2.size(2);
    const int w_c2 = feat_c2.size(3);
    const int h_c3 = feat_c3.size(2);
    const int w_c3 = feat_c3.size(3);
    const int h_c4 = feat_c4.size(2);
    const int w_c4 = feat_c4.size(3);
    const int h_c5 = feat_c5.size(2);
    const int w_c5 = feat_c5.size(3);

    ms_deformable_col2im_cuda_c2345(
        grad_output.data_ptr<float>(),
        feat_c2.data_ptr<float>(),
        feat_c3.data_ptr<float>(),
        feat_c4.data_ptr<float>(),
        feat_c5.data_ptr<float>(),
        h_c2, w_c2, h_c3, w_c3, h_c4, w_c4, h_c5, w_c5,
        sampling_loc.data_ptr<float>(),
        attn_weight.data_ptr<float>(),
        batch_size, channels, num_views, num_query, num_point,
        grad_value_c2.data_ptr<float>(),
        grad_value_c3.data_ptr<float>(),
        grad_value_c4.data_ptr<float>(),
        grad_value_c5.data_ptr<float>(),
        grad_sampling_loc.data_ptr<float>(),
        grad_attn_weight.data_ptr<float>()
    );

    return {
        grad_value_c2, grad_value_c3, grad_value_c4, grad_value_c5, grad_sampling_loc, grad_attn_weight
    };
}

std::vector<at::Tensor> ms_deform_attn_cuda_c23456_backward(
    const at::Tensor& grad_output,
    const at::Tensor& feat_c2,  // [B, N, H, W, C]
    const at::Tensor& feat_c3,  // [B, N, H, W, C]
    const at::Tensor& feat_c4,  // [B, N, H, W, C]
    const at::Tensor& feat_c5,  // [B, N, H, W, C]
    const at::Tensor& feat_c6,  // [B, N, H, W, C]
    const at::Tensor& sampling_loc,  // [B, Q, P, 3]
    const at::Tensor& attn_weight  // [B, Q, P, 4]
    ) {
    AT_ASSERTM(feat_c2.is_contiguous(), "value tensor has to be contiguous");
    AT_ASSERTM(feat_c3.is_contiguous(), "value tensor has to be contiguous");
    AT_ASSERTM(feat_c4.is_contiguous(), "value tensor has to be contiguous");
    AT_ASSERTM(feat_c5.is_contiguous(), "value tensor has to be contiguous");
    AT_ASSERTM(feat_c6.is_contiguous(), "value tensor has to be contiguous");
    AT_ASSERTM(sampling_loc.is_contiguous(), "sampling_loc tensor has to be contiguous");
    AT_ASSERTM(attn_weight.is_contiguous(), "attn_weight tensor has to be contiguous");
    AT_ASSERTM(grad_output.is_contiguous(), "grad_output tensor has to be contiguous");

    AT_ASSERTM(feat_c2.is_cuda(), "value must be a CUDA tensor");
    AT_ASSERTM(feat_c3.is_cuda(), "value must be a CUDA tensor");
    AT_ASSERTM(feat_c4.is_cuda(), "value must be a CUDA tensor");
    AT_ASSERTM(feat_c5.is_cuda(), "value must be a CUDA tensor");
    AT_ASSERTM(feat_c6.is_cuda(), "value must be a CUDA tensor");
    AT_ASSERTM(sampling_loc.is_cuda(), "sampling_loc must be a CUDA tensor");
    AT_ASSERTM(attn_weight.is_cuda(), "attn_weight must be a CUDA tensor");
    AT_ASSERTM(grad_output.is_cuda(), "grad_output must be a CUDA tensor");

    const int batch_size = feat_c2.size(0);
    const int num_views = feat_c2.size(1);
    const int channels = feat_c2.size(4);
    const int num_query = sampling_loc.size(1);
    const int num_point = sampling_loc.size(2);
    AT_ASSERTM(num_point <= MAX_POINT, "num_point exceed limits");

    auto grad_value_c2 = at::zeros_like(feat_c2);
    auto grad_value_c3 = at::zeros_like(feat_c3);
    auto grad_value_c4 = at::zeros_like(feat_c4);
    auto grad_value_c5 = at::zeros_like(feat_c5);
    auto grad_value_c6 = at::zeros_like(feat_c6);
    auto grad_sampling_loc = at::zeros_like(sampling_loc);
    auto grad_attn_weight = at::zeros_like(attn_weight);

    const int h_c2 = feat_c2.size(2);
    const int w_c2 = feat_c2.size(3);
    const int h_c3 = feat_c3.size(2);
    const int w_c3 = feat_c3.size(3);
    const int h_c4 = feat_c4.size(2);
    const int w_c4 = feat_c4.size(3);
    const int h_c5 = feat_c5.size(2);
    const int w_c5 = feat_c5.size(3);
    const int h_c6 = feat_c6.size(2);
    const int w_c6 = feat_c6.size(3);

    ms_deformable_col2im_cuda_c23456(
        grad_output.data_ptr<float>(),
        feat_c2.data_ptr<float>(),
        feat_c3.data_ptr<float>(),
        feat_c4.data_ptr<float>(),
        feat_c5.data_ptr<float>(),
        feat_c6.data_ptr<float>(),
        h_c2, w_c2, h_c3, w_c3, h_c4, w_c4, h_c5, w_c5, h_c6, w_c6,
        sampling_loc.data_ptr<float>(),
        attn_weight.data_ptr<float>(),
        batch_size, channels, num_views, num_query, num_point,
        grad_value_c2.data_ptr<float>(),
        grad_value_c3.data_ptr<float>(),
        grad_value_c4.data_ptr<float>(),
        grad_value_c5.data_ptr<float>(),
        grad_value_c6.data_ptr<float>(),
        grad_sampling_loc.data_ptr<float>(),
        grad_attn_weight.data_ptr<float>()
    );

    return {
        grad_value_c2, grad_value_c3, grad_value_c4, grad_value_c5, grad_value_c6, grad_sampling_loc, grad_attn_weight
    };
}

#ifdef TORCH_EXTENSION_NAME
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("_ms_deform_attn_cuda_c2345_forward", &ms_deform_attn_cuda_c2345_forward, "pass");
    m.def("_ms_deform_attn_cuda_c2345_backward", &ms_deform_attn_cuda_c2345_backward, "pass");
    m.def("_ms_deform_attn_cuda_c23456_forward", &ms_deform_attn_cuda_c23456_forward, "pass");
    m.def("_ms_deform_attn_cuda_c23456_backward", &ms_deform_attn_cuda_c23456_backward, "pass");
}
#endif