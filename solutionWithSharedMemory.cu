
// System includes
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>


// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include "device_functions.h"
#include <helper_cuda.h>

#define MAX_PIXEL_INTENSITY          63
#define IMAGE_MATRIX_DIMENSION       15
#define RANDOM_SEED                  5

#define WINDOW_DIMENSION             5
#define WINDOW_COUNT_PER_FRAME       11

#define FRAME_DIMENSION_WIDTH        15
#define FRAME_DIMENSION_HEIGHT       5        
#define FRAME_COUNT                  11

#define THREADS_IN_BLOCK_WIDTH       WINDOW_COUNT_PER_FRAME
#define THREADS_IN_BLOCK_HEIGHT      1

#define BLOCKS_IN_GRID_WIDTH         1
#define BLOCKS_IN_GRID_HEIGHT        1

// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.stride + col)
typedef struct {
    int width;
    int height;
    int stride;
    unsigned int* elements;
} Matrix;


// To measure time
cudaEvent_t start, stop;
float elapsed_time_ms;


__device__ unsigned int GetElement(const Matrix* A, int row, int col)
{
    return A->elements[row * A->stride + col];
}

__device__ void SetElement(Matrix* A, int row, int col, unsigned int value)
{
    A->elements[row * A->stride + col] = value;
}

// Get the WINDOW_DIMENSIONxWINDOW_DIMENSION sub-matrix Asub of A that is
// located col sub-matrices to the right and row sub-matrices down
// from the upper-left corner of A
__device__ void GetWindowFromGlobal(Matrix globalMatrix, Matrix *matrixInWindow, int frameNumber, int windowNumber)
{    
    unsigned int element;
    int startIndex = globalMatrix.stride * frameNumber + windowNumber;

    for (int i = 0; i < matrixInWindow->height; i++)
    {
        for (int j = 0; j < matrixInWindow->width; j++)
        {
            element = 
                globalMatrix.elements[startIndex + i * globalMatrix.stride + j];

            matrixInWindow->elements[i * matrixInWindow->stride + j] = element;
        }
    }
}

__device__ void GetWindowFromShared(Matrix sharedMatrix, Matrix* matrixInWindow, int windowNumber)
{
    unsigned int element;
    int startIndex = windowNumber;

    for (int i = 0; i < matrixInWindow->height; i++)
    {
        for (int j = 0; j < matrixInWindow->width; j++)
        {
            element =
                sharedMatrix.elements[startIndex + i * sharedMatrix.stride + j];

            matrixInWindow->elements[i * matrixInWindow->stride + j] = element;
        }
    }
}

__device__ void SetWindowToGlobal(Matrix* matrixInWindow, Matrix* globalMatrix,
    int frameNumber, int windowNumber)
{
    unsigned int value;
    int startIndex = globalMatrix->stride * frameNumber + windowNumber;

    for (int i = 0; i < matrixInWindow->height; i++)
    {
        for (int j = 0; j < matrixInWindow->width; j++)
        {
            value = GetElement(matrixInWindow, i, j);

            atomicAdd(&(globalMatrix->elements[startIndex + i * globalMatrix->stride + j]), value);
        }
    }
}

__device__ void SetWindowToShared(Matrix* matrixInWindow, Matrix* sharedMatrix, int windowNumber)
{
    unsigned int value;
    int startIndex = windowNumber;

    for (int i = 0; i < matrixInWindow->height; i++)
    {
        for (int j = 0; j < matrixInWindow->width; j++)
        {
            value = GetElement(matrixInWindow, i, j);

            sharedMatrix->elements[startIndex + i * sharedMatrix->stride + j] += value;
        }
    }
}


__global__ void MatMulKernel(Matrix A, Matrix C, Matrix matrixInWindow, Matrix resultMatrixInWindow, int frameIndex)
{    
    int frameNumber = frameIndex;
    int windowNumber = threadIdx.x;

    __shared__ Matrix A_shared;
    A_shared.width = FRAME_DIMENSION_WIDTH;
    A_shared.height = FRAME_DIMENSION_HEIGHT;
    A_shared.stride = A_shared.width;
    A_shared.elements = 
        (unsigned int*)malloc(FRAME_DIMENSION_WIDTH *
                              FRAME_DIMENSION_HEIGHT * 
                              sizeof(unsigned int));

    if (windowNumber % WINDOW_DIMENSION == 0)
    {
        GetWindowFromGlobal(A, &matrixInWindow, frameNumber, windowNumber);
        SetWindowToShared(&matrixInWindow, &A_shared, windowNumber);
    }

    __syncthreads();


    GetWindowFromShared(A_shared, &matrixInWindow, windowNumber);
    GetWindowFromGlobal(C, &resultMatrixInWindow, frameNumber, windowNumber);

    unsigned int accumulator;

    for (int i = 0; i < matrixInWindow.height; i++)
    {
        for (int j = 0; j < matrixInWindow.width; j++)
        {
            accumulator = 0;
            for (int k = 0; k < WINDOW_DIMENSION; k++)
            {
                accumulator += GetElement(&matrixInWindow, i, k) *
                    GetElement(&matrixInWindow, k, j);
            }

            SetElement(&resultMatrixInWindow, i, j, accumulator);
        }
    }

    __syncthreads();

    SetWindowToGlobal(&resultMatrixInWindow, &C, frameNumber, windowNumber);
}

static void generateMatrixElements(Matrix* A)
{
    size_t size = A->width * A->height;

    srand(RANDOM_SEED);

    for (int i = 0; i < size; i++)
    {
        A->elements[i] = rand() % MAX_PIXEL_INTENSITY;
    }
}

static void printMatrixOnConsole(Matrix* matrix, char* matrixName)
{
    printf(" %s\n", matrixName);

    for (int row = 0; row < matrix->height; row++)
    {
        for (int column = 0; column < matrix->width; column++)
        {
            printf(" %d\t", matrix->elements[row * matrix->width + column]);
        }
        printf("\n");
    }

    printf("\n\n");
}


int main(int argc, char** argv)
{
    Matrix A;
    Matrix C;
    int elementCount;
    size_t size;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    A.width = C.width = IMAGE_MATRIX_DIMENSION;
    A.height = C.height = IMAGE_MATRIX_DIMENSION;
    A.stride = C.stride = IMAGE_MATRIX_DIMENSION;

    elementCount = IMAGE_MATRIX_DIMENSION * IMAGE_MATRIX_DIMENSION;
    size = elementCount * sizeof(unsigned int);

    A.elements = (unsigned int*)malloc(size);
    C.elements = (unsigned int*)calloc(elementCount, sizeof(unsigned int));

    generateMatrixElements(&A);
    printMatrixOnConsole(&A, "Matrix A - Initial");
    printMatrixOnConsole(&C, "Matrix C - Initial");

    for (int frameIndex = 0; frameIndex < FRAME_COUNT; frameIndex++)
    {
        Matrix frameMatrix;
        frameMatrix.height = FRAME_DIMENSION_HEIGHT;
        frameMatrix.width = FRAME_DIMENSION_WIDTH;
        frameMatrix.stride = frameMatrix.width;

        // Load A to device memory
        Matrix d_A;
        d_A.width = A.width;
        d_A.height = A.height;
        d_A.stride = A.stride;

        cudaMalloc(&d_A.elements, size);
        cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);

        // Allocate C in device memory
        Matrix d_C;
        d_C.width = C.width;
        d_C.height = C.height;
        d_C.stride = C.stride;

        cudaMalloc(&d_C.elements, size);
        cudaMemcpy(d_C.elements, C.elements, size, cudaMemcpyHostToDevice);

        // Allocate matrix for computations is device
        // Note: It makes easier to debug with NSight when allocating it from the host
        Matrix matrixInWindow;
        matrixInWindow.width = WINDOW_DIMENSION;
        matrixInWindow.height = WINDOW_DIMENSION;
        matrixInWindow.stride = matrixInWindow.width;
        cudaMalloc(&matrixInWindow.elements, WINDOW_DIMENSION * WINDOW_DIMENSION * sizeof(unsigned int));

        // Allocate matrix for computations is device
        // Note: It makes easier to debug with NSight when allocating it from the host
        Matrix resultMatrixInWindow;
        resultMatrixInWindow.width = WINDOW_DIMENSION;
        resultMatrixInWindow.height = WINDOW_DIMENSION;
        resultMatrixInWindow.stride = matrixInWindow.width;
        cudaMalloc(&resultMatrixInWindow.elements, WINDOW_DIMENSION * WINDOW_DIMENSION * sizeof(unsigned int));

        // Start to measure time
        cudaEventRecord(start, 0);

        // Invoke kernel
        dim3 dimBlock(THREADS_IN_BLOCK_WIDTH, THREADS_IN_BLOCK_HEIGHT);
        dim3 dimGrid(BLOCKS_IN_GRID_WIDTH, BLOCKS_IN_GRID_HEIGHT);
        MatMulKernel <<< dimGrid, dimBlock >>> (d_A, d_C, matrixInWindow, resultMatrixInWindow, frameIndex);

        // Wait for GPU to finish before accessing on host
        cudaDeviceSynchronize();

        // Read C from device memory
        cudaMemcpy(C.elements, d_C.elements, size, cudaMemcpyDeviceToHost);

        // Stop to measure time
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed_time_ms, start, stop);

        // Free device memory
        cudaFree(d_A.elements);
        cudaFree(d_C.elements);
        cudaFree(matrixInWindow.elements);
        cudaFree(resultMatrixInWindow.elements);

        printf("After Frame %d Run\n", frameIndex);
        printf(" Elapsed Time (ms): %f \n", elapsed_time_ms);
        printMatrixOnConsole(&C, "Matrix C");
    }


    return 0;
}