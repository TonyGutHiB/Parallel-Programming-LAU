#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define BLOCK_SIZE 16
_global_ void matrixMultiplication(int* matrixA, int* matrixB, int* matrixC, int rowsA, int columnsA, int columnsB) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rowsA && col < columnsB) {
        int sum = 0;
        for (int k = 0; k < columnsA; ++k) {
            sum += matrixA[row * columnsA + k] * matrixB[k * columnsB + col];
        }
        matrixC[row * columnsB + col] = sum;
    }
}
int main() {
    int  rowsA , columnsA, columnsB;
     printf("Enter the dimensions of matrix A (rowsA): ");
     scanf("%d", &rowsA);
     printf ("Enter the dimensions of matrix B (columnsA x columnsB): ");
    
  scanf("%d %d", &columnsA, &columnsB);

  
  
  int* matrixA, * matrixB, * matrixC;
  
  int* dev_matrixA, * dev_matrixB, * dev_matrixC;

  
  
  size_t sizeA = rowsA * columnsA * sizeof(int);
    size_t sizeB = columnsA * columnsB * sizeof(int);
  
  size_t sizeC = rowsA * columnsB * sizeof(int);

    matrixA = (int*)malloc(sizeA);
  
  matrixB = (int*)malloc(sizeB);
    matrixC = (int*)malloc(sizeC);

  
  srand(time(NULL));
    for (int i = 0; i < rowsA * columnsA; ++i) {
        matrixA[i] = rand() % 10;
  
    }
    for (int i = 0; i < columnsA * columnsB; ++i) {
        matrixB[i] = rand() % 10;
    
    }
    cudaMalloc((void**)&dev_matrixA, sizeA);
    cudaMalloc((void**)&dev_matrixB, sizeB);
    
  cudaMalloc((void**)&dev_matrixC, sizeC);
    cudaMemcpy(dev_matrixA, matrixA, sizeA, cudaMemcpyHostToDevice);
       cudaMemcpy(dev_matrixB, matrixB, sizeB, cudaMemcpyHostToDevice);
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
      dim3 gridDim((columnsB + blockDim.x - 1) / blockDim.x, (rowsA + blockDim.y - 1) / blockDim.y);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
     cudaEventCreate(&stop);

    cudaEventRecord(start);

    matrixMultiplication<<<gridDim, blockDim>>>(dev_matrixA, dev_matrixB, dev_matrixC, rowsA, columnsA, columnsB);

  
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(matrixC, dev_matrixC, sizeC, cudaMemcpyDeviceToHost);

    printf("%f milliseconds", milliseconds);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(dev_matrixA);
    cudaFree(dev_matrixB);
    cudaFree(dev_matrixC);
    free(matrixA);
    free(matrixB);
    free(matrixC);
    return 0;
}
