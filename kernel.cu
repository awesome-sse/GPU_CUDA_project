extern "C" {
    #define BLOCK_SIZE 16
    
    __global__ void applyConvolution(float *input, float *output, int width, int height) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        
        // Копирование в разделяемую память для более быстрого доступа нитей к памяти внутри блока
        __shared__ float sharedInput[BLOCK_SIZE + 2][BLOCK_SIZE + 2];

        if (x < width && y < height) {
            sharedInput[threadIdx.y + 1][threadIdx.x + 1] = input[y * width + x];
        }
        
        // Обработка границ: если нить относится к границе блока, то берется либо
        // значение матрицы соседнего блока, либо значение остается 0 (padding = 1, который заполняется нулями)
        if (threadIdx.x == 0 && x > 0) {
            sharedInput[threadIdx.y + 1][0] = input[y * width + (x - 1)];
        }
        if (threadIdx.x == BLOCK_SIZE - 1 && x < width - 1) {
            sharedInput[threadIdx.y + 1][BLOCK_SIZE + 1] = input[y * width + (x + 1)];
        }
        if (threadIdx.y == 0 && y > 0) {
            sharedInput[0][threadIdx.x + 1] = input[(y - 1) * width + x];
        }
        if (threadIdx.y == BLOCK_SIZE - 1 && y < height - 1) {
            sharedInput[BLOCK_SIZE + 1][threadIdx.x + 1] = input[(y + 1) * width + x];
        }
        if (threadIdx.x == 0 && x > 0 && threadIdx.y == 0 && y > 0) {
            sharedInput[0][0] = input[(y - 1) * width + (x - 1)];
        }
        if (threadIdx.x == BLOCK_SIZE - 1 && x < width - 1 && threadIdx.y == 0 && y > 0) {
            sharedInput[0][BLOCK_SIZE + 1] = input[(y - 1) * width + (x + 1)];
        }
        if (threadIdx.y == BLOCK_SIZE - 1 && y < height - 1 && threadIdx.x == 0 && x > 0) {
            sharedInput[BLOCK_SIZE + 1][0] = input[(y + 1) * width + (x - 1)];
        }
        if (threadIdx.x == BLOCK_SIZE - 1 && x < width - 1 && threadIdx.y == BLOCK_SIZE - 1 && y < height - 1) {
            sharedInput[BLOCK_SIZE + 1][BLOCK_SIZE + 1] = input[(y + 1) * width + (x + 1)];
        }
        
        __syncthreads();
        
        // Применение свертки (1 нить - 1 операция применения свертки)
        if (x < width && y < height) {
            float sum = 0.0f;
            for (int i = -1; i <= 1; ++i) {
                for (int j = -1; j <= 1; ++j) {
                    sum += sharedInput[threadIdx.y + i + 1][threadIdx.x + j + 1] * kernel[i + 1][j + 1];
                }
            }
            // Запись результата (так как значения в картинке от 0 до 255, то срезаем результаты по указанному диапазону)
            output[y * width + x] = max(min(sum, float(255)), float(0));
        }
    }
}
