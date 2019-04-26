#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

std::vector<torch::Tensor> y2_cuda_decompress_and_apply_deltas(
        std::vector<int> message,
        torch::Tensor weights,
        torch::Tensor bias
);

std::vector<int> y2_cuda_process_gradient_and_compute_message(
        torch::Tensor grad,
        torch::Tensor weights,
        torch::Tensor residuals
);

// C++ interface

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")

std::vector<torch::Tensor> y2_decompress_and_apply_deltas(
        std::vector<int> message,
        torch::Tensor weights,
        torch::Tensor bias
) {
    CHECK_CUDA(message);
    CHECK_CUDA(weights);
    CHECK_CUDA(bias);

    return y2_cuda_decompress_and_apply_deltas(message, weights, bias);
}

std::vector<int> y2_process_gradient_and_compute_message(
        torch::Tensor grad,
        torch::Tensor weights,
        torch::Tensor residuals
) {
    CHECK_CUDA(grad);
    CHECK_CUDA(weights);
    CHECK_CUDA(residuals);

    return y2_cuda_process_gradient_and_compute_message(grad, weights, residuals);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("decompress", &y2_decompress_and_apply_deltas, "Decompress a message and apply the deltas to the weights of the NN (CUDA)");
  m.def("compress", &y2_process_gradient_and_compute_message, "Add gradient deltas to the residuals, compute message to send and apply the deltas above the threshold (CUDA)");
}
