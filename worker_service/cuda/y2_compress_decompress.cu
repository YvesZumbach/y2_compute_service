
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

std::vector<torch::Tensor> y2_cuda_decompress_and_apply_deltas(
        std::vector<int> message,
        torch::Tensor weights,
        torch::Tensor bias
) {
    // TODO: Write the actual kernel
}

std::vector<int> y2_cuda_process_gradient_and_compute_message(
        torch::Tensor grad,
        torch::Tensor weights,
        torch::Tensor residuals
) {
    // TODO: Write the actual kernel
}