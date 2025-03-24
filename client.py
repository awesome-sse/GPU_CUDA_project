import argparse
import numpy as np
import aiohttp
import asyncio
import os
from PIL import Image
import io

def get_kernel(kernel_type, custom_kernel=""):
    if kernel_type == "blur":
        return np.array([[1/9, 1/9, 1/9],
                         [1/9, 1/9, 1/9],
                         [1/9, 1/9, 1/9]], dtype=np.float32)  # Ядро размытия 3x3
    elif kernel_type == "sharpness":
        return np.array([[0, -1, 0],
                         [-1, 5, -1],
                         [0, -1, 0]], dtype=np.float32)  # Ядро повышения резкости 3x3

    elif kernel_type == "custom":
        if custom_kernel != "":
            custom_values = list(map(float, custom_kernel.split(',')))
            kernel_size = int(len(custom_values) ** 0.5)  # Предполагаем квадратное ядро
            kernel_matrix = np.array(custom_values).reshape((kernel_size, kernel_size)).astype(np.float32)

            return kernel_matrix
        else:
            raise ValueError("Not entered value custom_kernel")

    else:
        raise ValueError("Unknown kernel type. Use 'blur' or 'sharpness'.")
     

async def send_to_server(session, server_url, image_matrix, kernel_matrix):
    payload = {
        "image": image_matrix.tolist(),  # Преобразуем NumPy массив в список для JSON
        "kernel": kernel_matrix.tolist()   # То же самое для ядра
    }
    async with session.post(server_url + "/convolve", json=payload) as response:
        if response.status != 200:
            error = await response.text()
            print(f"Error from server {server_url}: {error}")
            return None
        return await response.json()

async def process_image(session, server_url, image_path, output_dir, kernel_matrix):
    image = Image.open(image_path).convert("L")  # Конвертируем в градации серого
    image_matrix = np.array(image, dtype=np.float32)

    result = await send_to_server(session, server_url, image_matrix, kernel_matrix)
    if result is not None:
        blurred_image = np.array(result["result"], dtype=np.uint8).reshape(image_matrix.shape)
        output_path = os.path.join(output_dir, os.path.basename(image_path))
        Image.fromarray(blurred_image).save(output_path)
        print(f"Processed and saved: {output_path}, GPU time: {np.round(result.get('total_gpu_time_ms', 0), 2)} ms")
        return result.get("total_gpu_time_ms", 0)
    return 0

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--servers", nargs="+", required=True)
    parser.add_argument("--images", default="images")
    parser.add_argument("--output", default="results")
    parser.add_argument("--kernel_type", choices=["blur", "sharpness", "custom"], required=True)
    parser.add_argument("--custom_kernel", type=str, help="Custom kernel as a comma-separated string (e.g., '0,-1,0,-1,5,-1,0,-1,0' for sharpness)", default="")
    
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Получаем ядро свертки на основе типа
    kernel_matrix = get_kernel(args.kernel_type, args.custom_kernel)

    image_files = [os.path.join(args.images, f) for f in os.listdir(args.images) if f.endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        print(f"No images found in directory: {args.images}")
        return

    start_time = asyncio.get_event_loop().time()
    total_gpu_time_ms = 0
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        for image_path in image_files:
            server_url = args.servers[len(tasks) % len(args.servers)]
            task = asyncio.create_task(process_image(session, server_url, image_path, args.output, kernel_matrix))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        total_gpu_time_ms = sum(results)
    
    end_time = asyncio.get_event_loop().time()
    total_time = end_time - start_time
    print(f"All images processed. Total time: {total_time:.6f} seconds")
    print(f"Total GPU time: {total_gpu_time_ms:.2f} ms")

if __name__ == "__main__":
    asyncio.run(main())