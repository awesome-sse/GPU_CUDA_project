import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from server import kernel_injection
import pytest

def apply_convolution_true(input_data, kernel):
    pad_width = 1
    padded_input = np.pad(input_data, pad_width, mode='constant')
    
    height, width = input_data.shape
    output_data = np.zeros((height, width), dtype=np.float32)
    
    for y in range(height):
        for x in range(width):
            sum_val = 0.0
            
            for i in range(-1, 2):
                for j in range(-1, 2):
                    sum_val += padded_input[y + 1 + i, x + 1 + j] * kernel[i + 1][j + 1]
            
            output_data[y, x] = max(min(sum_val, 255), 0)
    
    return output_data


with open("kernel.cu", "r") as f:
    cuda_code = f.read()


def test_apply_convolution():
    width = 256
    height = 256
    input_data = np.random.rand(width * height).astype(np.float32)
    output_data = np.zeros_like(input_data)

    block_size = (16, 16, 1) 
    grid_size = ((width + block_size[0] - 1) // block_size[0], 
                 (height + block_size[1] - 1) // block_size[1], 1)

    kernel = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]

    kernel_code = kernel_injection(kernel)
    cuda_kernel_code = kernel_code + cuda_code
    mod = SourceModule(cuda_kernel_code)
    apply_convolution = mod.get_function("applyConvolution")

    apply_convolution(cuda.In(input_data), cuda.Out(output_data), np.int32(width), np.int32(height), block=block_size, grid=grid_size)

    for i in range(width * height):
        assert output_data[i] >= 0 and output_data[i] <= 255

    output_data_true = apply_convolution_true(input_data.reshape((height, width)), np.array(kernel))

    for i in range(width * height):
        assert np.isclose(output_data[i], output_data_true.flatten()[i], atol=1e-4) == True


def test_blur_filter():
    blur_kernel = [[1/9, 1/9, 1/9], [1/9, 1/9, 1/9], [1/9, 1/9, 1/9]]
    
    kernel_code = kernel_injection(blur_kernel)
    cuda_kernel_code = kernel_code + cuda_code
    
    mod = SourceModule(cuda_kernel_code)
    apply_convolution = mod.get_function("applyConvolution")
    
    width = 256
    height = 256
    input_data = np.random.rand(width * height).astype(np.float32)
    output_data = np.zeros_like(input_data)

    block_size = (16, 16, 1)
    grid_size = ((width + block_size[0] - 1) // block_size[0], 
                 (height + block_size[1] - 1) // block_size[1], 1)

    apply_convolution(cuda.In(input_data), cuda.Out(output_data), np.int32(width), np.int32(height), block=block_size, grid=grid_size)
    for i in range(width * height):
        assert output_data[i] >= 0 and output_data[i] <= 255

    output_data_true = apply_convolution_true(input_data.reshape((height, width)), np.array(blur_kernel))

    for i in range(width * height):
        assert np.isclose(output_data[i], output_data_true.flatten()[i], atol=1e-4) == True


def test_sharpen_filter():
    sharpen_kernel = [[0, -1, 0], [-1, 5, -1], [0, -1, 0]]
    
    kernel_code = kernel_injection(sharpen_kernel)
    cuda_kernel_code = kernel_code + cuda_code
    
    mod = SourceModule(cuda_kernel_code)
    apply_convolution = mod.get_function("applyConvolution")
    
    width = 256
    height = 256
    input_data = np.random.rand(width * height).astype(np.float32)
    output_data = np.zeros_like(input_data)

    block_size = (16, 16, 1)
    grid_size = ((width + block_size[0] - 1) // block_size[0], 
                 (height + block_size[1] - 1) // block_size[1], 1)

    apply_convolution(cuda.In(input_data), cuda.Out(output_data), np.int32(width), np.int32(height), block=block_size, grid=grid_size)
    for i in range(width * height):
        assert output_data[i] >= 0 and output_data[i] <= 255

    output_data_true = apply_convolution_true(input_data.reshape((height, width)), np.array(sharpen_kernel))

    for i in range(width * height):
        assert np.isclose(output_data[i], output_data_true.flatten()[i], atol=1e-4) == True


def test_zero_kernel():
    zero_kernel = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    
    kernel_code = kernel_injection(zero_kernel)
    cuda_kernel_code = kernel_code + cuda_code
    
    mod = SourceModule(cuda_kernel_code)
    apply_convolution = mod.get_function("applyConvolution")
    
    width = 256
    height = 256
    input_data = np.random.rand(width * height).astype(np.float32)
    output_data = np.zeros_like(input_data)

    block_size = (16, 16, 1)
    grid_size = ((width + block_size[0] - 1) // block_size[0], 
                 (height + block_size[1] - 1) // block_size[1], 1)

    apply_convolution(cuda.In(input_data), cuda.Out(output_data), np.int32(width), np.int32(height), block=block_size, grid=grid_size)
    for i in range(width * height):
        assert output_data[i] == 0

    output_data_true = apply_convolution_true(input_data.reshape((height, width)), np.array(zero_kernel))

    for i in range(width * height):
        assert np.isclose(output_data[i], output_data_true.flatten()[i], atol=1e-4) == True


if __name__ == '__main__':
    pytest.main()