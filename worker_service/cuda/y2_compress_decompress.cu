
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

const int delta = 5.0;
const int weight_index_mask = 0b111111111111111111111;

__global__ void y2_cuda_decompress_and_apply_deltas(
        std::vector<std::vector<int>> messages,
        std::vector<torch::Tensor> parameters
) {
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    for (unsigned i=0; i < messages.size(); i++) {
        std::vector<int> message = messages[i];
        for (unsigned j=threadId; j < message.size(); j += blockDim.x) {
            int encoded = message[j];
            // Whether the threshold is positive or negative is encoded in the least significant bit
            const int positive_delta = encoded & 0b1;
            // Keep the 21 least significant bits for the index of the weight
            const unsigned int weight_index = index & weight_index_mask;
            encoded = encoded >> 21;
            // Whether we are changing a bias or a weight
            const unsigned int is_weight = encoded & 0b1;
            encoded = encoded >> 1;
            // The remaining bits encode the tensor index
            const unsigned int tensor_index = encoded;
            torch::Tensor t;
            if (is_weights) {
                t = parameters[tensor_index];
            } else {
                t = parameters[tensor_index - 1].&grad();
            }
            t[weight_index] += positive_delta ? delta : -delta;
        }
    }

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