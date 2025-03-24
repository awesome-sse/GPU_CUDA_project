import numpy as np
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
from flask import Flask, request, jsonify
from contextlib import contextmanager
import io

# CUDA код для применения свертки
with open("kernel.cu", "r") as f:
    cuda_code = f.read()

# Инициализация CUDA и Flask
cuda.init()
device = cuda.Device(0)
app = Flask(__name__)

def kernel_injection(kernel_matrix):
    kernel_code = "__constant__ float kernel[3][3] = {\n"
    for i in range(3):
        kernel_code += "\t{"
        for j in range(2):
            kernel_code += f"{kernel_matrix[i][j]},"
        kernel_code += f"{kernel_matrix[i][2]}"
        kernel_code += "}"
        if (i < 2):
            kernel_code += ",\n"
        else:
            kernel_code += "\n};"
    return kernel_code

@contextmanager
def gpu_ctx():
    ctx = device.make_context()
    try:
        yield ctx
    finally:
        ctx.pop()

@app.route("/convolve", methods=["POST"])
def convolve():
    try:
        data = request.json
        image_matrix = np.array(data['image'], dtype=np.float32)
        kernel_matrix = np.array(data['kernel'], dtype=np.float32)

        h, w = image_matrix.shape

        out = np.zeros_like(image_matrix, dtype=np.float32)

        block = (16, 16, 1)
        grid = (
            (w + block[0] - 1) // block[0],
            (h + block[1] - 1) // block[1]
        )
        with gpu_ctx():
            start_event = cuda.Event()
            end_event = cuda.Event()
            start_event.record()

            kernel_code = kernel_injection(kernel_matrix)

            d_src = cuda.mem_alloc(image_matrix.nbytes)
            d_dst = cuda.mem_alloc(out.nbytes)
            cuda.memcpy_htod(d_src, image_matrix)

            mod = SourceModule(kernel_code + cuda_code, options=['-use_fast_math'])

            cuda_kernel = mod.get_function("applyConvolution")

            cuda_kernel(d_src, d_dst, np.int32(w), np.int32(h), block=block, grid=grid)
            cuda.memcpy_dtoh(out, d_dst)

            d_src.free()
            d_dst.free()

            end_event.record()
            end_event.synchronize()

            total_gpu_time_ms = start_event.time_till(end_event)

        return jsonify({
            "result": out.tolist(),
            "total_gpu_time_ms": total_gpu_time_ms
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)