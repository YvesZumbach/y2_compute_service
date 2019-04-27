#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

void y2_cuda_decompress_and_apply_deltas(
        std::vector<std::vector<int>> messages,
        std::vector<torch::Tensor> parameters
);

std::vector<int> y2_cuda_process_gradient_and_compute_message(
        std::vector<torch::Tensor> parameters,
        std::vector<torch::Tensor> residuals
);

// C++ interface

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")

void y2_decompress_and_apply_deltas(
        std::vector<std::vector<int>> messages,
        std::vector<torch::Tensor> parameters
) {
    CHECK_CUDA(messages);
    CHECK_CUDA(parameters);

    y2_cuda_decompress_and_apply_deltas(messages, parameters);
}

std::vector<int> y2_process_gradient_and_compute_message(
        std::vector<torch::Tensor> grad,
        std::vector<torch::Tensor> parameters,
        std::vector<torch::Tensor> residuals
) {
    CHECK_CUDA(parameters);
    CHECK_CUDA(residuals);

    return y2_cuda_process_gradient_and_compute_message(parameters, residuals);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("decompress", &y2_decompress_and_apply_deltas, "Decompress a message and apply the deltas to the weights of the NN (CUDA)");
  m.def("compress", &y2_process_gradient_and_compute_message, "Add gradient deltas to the residuals, compute message to send and apply the deltas above the threshold (CUDA)");
}
