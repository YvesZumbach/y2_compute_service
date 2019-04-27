
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

__global__ void y2_cuda_decompress_and_apply_deltas(
        std::vector<std::vector<int>> messages,
        std::vector<torch::Tensor> parameters
) {
    // TODO: Write the actual kernel
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    for (unsigned i=0; i < messages.size(); i++) {
        std::vector<int> message = messages[i];
        for (unsigned j=threadId; j < message.size(); j += blockDim.x) {
            int encoded = message[j];
            int positive_threshold = encoded >> 31;
            int tensor_id = (encoded << 1) >> 29;
            int weight_id = (encoded << 4) >> 4;
            int is_weights = tensor_id % 2 == 0;
            torch::Tensor t;
            if (is_weights) {
                t = parameters[tensor_id];
            } else {
                t = parameters[tensor_id - 1].&grad();
            }
            if (positive_threshold) {
                t[weight_id] += 5.0;
            } else {
                t[weight_id] -= 5.0;
            }
        }
    }
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