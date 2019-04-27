
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

__global__ void y2_cuda_decompress_and_apply_deltas(
        std::vector<std::vector<int>> messages,
        std::vector<torch::Tensor> parameters
) {
    // TODO: Write the actual kernel

    // for message in messages
    //      for each integer in message
                    positive/negative, index : split(integer)
                    find id of parameter list to apply to
                    find id of parameter in list
                    apply + or - threshold

}

__global__ std::vector<int> y2_cuda_process_gradient_and_compute_message(
        std::vector<torch::Tensor> parameters,
        std::vector<torch::Tensor> residuals
) {
    // for each element in parameters gradients
            add value to associated residual in residuals vector
            if residual > threshold:
                   create compressed msg and add to msg map
                   substract/add threshold to residual
                   set gradient to threshold value
}