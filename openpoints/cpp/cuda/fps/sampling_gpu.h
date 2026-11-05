#pragma once
#include <torch/extension.h>
int furthest_point_sampling_wrapper(int b, int n, int m,
    at::Tensor points, at::Tensor temp, at::Tensor indices);
void furthest_point_sampling_kernel_launcher(int b, int n, int m,
    const float *points, float *temp, int *indices);
