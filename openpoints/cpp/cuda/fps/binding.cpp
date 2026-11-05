#include <torch/extension.h>
int furthest_point_sampling_wrapper(int, int, int, at::Tensor, at::Tensor, at::Tensor);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fps", &furthest_point_sampling_wrapper);
}
