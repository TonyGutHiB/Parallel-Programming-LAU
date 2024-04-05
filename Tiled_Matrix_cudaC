#include <stdio.h>
#include <stdlib.h>
#include <time.h>


#define BLOCK_SIZE 16

#define TILE_SIZE 16


_global_ void matrixMultiplication(int* matrixA, int* matrixB, int* matrixC, int rowsA, int columnsA, int columnsB) {

  int row = blockIdx.y * blockDim.y + threadIdx.y;
  
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  
    _shared_ int tileA[TILE_SIZE][TILE_SIZE];
  
  _shared_ int tileB[TILE_SIZE][TILE_SIZE];

  
  int tileRow, tileCol;
    int result = 0;
  
  for (int tile = 0; tile < (columnsA + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        tileRow = threadIdx.y;
  
    tileCol = threadIdx.x;
        if (tile * TILE_SIZE + tileCol < columnsA && row < rowsA)
    
          tileA[tileRow][tileCol] = matrixA[row * columnsA + tile * TILE_SIZE + tileCol];
        else
            tileA[tileRow][tileCol] = 0;

        if (tile * TILE_SIZE + tileRow < columnsA && col < columnsB)
            tileB[tileRow][tileCol] = matrixB[(tile * TILE_SIZE + tileRow) * columnsB + col];
        else
            tileB[tileRow][tileCol] = 0;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k) {
            result += tileA[tileRow][k] * tileB[k][tileCol];
        }

        __syncthreads();
    }
    if (row < rowsA && col < columnsB) {
        matrixC[row * columnsB + col] = result;
    }
}

int main() {
    
  int rowsA, columnsA, columnsB;
       printf("Enter the dimensions of matrix A (rowsA): ");
    scanf  ("%d", &rowsA);
    
  printf ("Enter the dimensions of matrix B (columnsA x columnsB): ");
  
  scanf("%d %d", &columnsA, &columnsB);

      int* matrixA, * matrixB, * matrixC;
      int* dev_matrixA, * dev_matrixB, * dev_matrixC;

      size_t sizeA = rowsA * columnsA * sizeof(int);
      
  size_t sizeB = columnsA * columnsB *sizeof(int);
    size_t sizeC = rowsA * columnsB *sizeof(int);  

    
  
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
