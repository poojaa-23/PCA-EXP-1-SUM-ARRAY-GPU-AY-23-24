# PCA: EXP-1  SUM ARRAY GPU
<h3>NAME: s.pooja abirami
<h3>REGISTER NO.:212223100041
<h3>EX. NO</h3>
<h3>DATE: 30/07/26</h3>
<h1> <align=center> SUM ARRAY ON HOST AND DEVICE </h3>
PCA-GPU-based-vector-summation.-Explore-the-differences.
i) Using the program sumArraysOnGPU-timer.cu, set the block.x = 1023. Recompile and run it. Compare the result with the execution configuration of block.x = 1024. Try to explain the difference and the reason.      
ii) Refer to sumArraysOnGPU-timer.cu, and let block.x = 256. Make a new kernel to let each thread handle two elements. Compare the results with other execution confi gurations.

## AIM:

To perform vector addition on host and device.

## EQUIPMENTS REQUIRED:

Hardware – PCs with NVIDIA GPU & CUDA NVCC
Google Colab with NVCC Compiler




## PROCEDURE:

1. Initialize the device and set the device properties.
2. Allocate memory on the host for input and output arrays.
3. Initialize input arrays with random values on the host.
4. Allocate memory on the device for input and output arrays, and copy input data from host to device.
5. Launch a CUDA kernel to perform vector addition on the device.
6. Copy output data from the device to the host and verify the results against the host's sequential vector addition. Free memory on the host and the device.

## PROGRAM:

```
%%cuda
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>

#define CHECK(call) \
{ \
    const cudaError_t error = call; \
    if (error != cudaSuccess) \
    { \
        fprintf(stderr, "Error: %s:%d, code:%d, reason:%s\n", \
                __FILE__, __LINE__, error, cudaGetErrorString(error)); \
        exit(EXIT_FAILURE); \
    } \
}

inline double seconds()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.0e-6);
}

void initialData(float *ip, int size)
{
    srand((unsigned)time(NULL));

    for (int i = 0; i < size; i++)
    {
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
    }
}

void sumArraysOnHost(float *A, float *B, float *C, const int N)
{
    for (int i = 0; i < N; i++)
    {
        C[i] = A[i] + B[i];
    }
}

void checkResult(float *hostRef, float *gpuRef, const int N)
{
    double epsilon = 1.0E-8;
    bool match = true;

    for (int i = 0; i < N; i++)
    {
        if (fabs(hostRef[i] - gpuRef[i]) > epsilon)
        {
            match = false;
            printf("Arrays do not match!\n");
            printf("host %5.2f gpu %5.2f at index %d\n",
                   hostRef[i], gpuRef[i], i);
            break;
        }
    }

    if (match)
        printf("Arrays match.\n");
}

// CUDA Kernel
__global__ void sumArraysOnGPU(float *A, float *B, float *C, int N)
{
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 2;

    if (i < N)
        C[i] = A[i] + B[i];

    if (i + 1 < N)
        C[i + 1] = A[i + 1] + B[i + 1];
}

int main()
{
    printf("CUDA Vector Addition Starting...\n");

    int dev = 0;
    cudaDeviceProp deviceProp;

    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    CHECK(cudaSetDevice(dev));

    printf("Using Device %d: %s\n", dev, deviceProp.name);

    int nElem = 1 << 24;
    size_t nBytes = nElem * sizeof(float);

    printf("Vector Size = %d\n", nElem);

    // Allocate Host Memory
    float *h_A = (float *)malloc(nBytes);
    float *h_B = (float *)malloc(nBytes);
    float *hostRef = (float *)malloc(nBytes);
    float *gpuRef = (float *)malloc(nBytes);

    double start, elapsed;

    // Initialize Data
    start = seconds();
    initialData(h_A, nElem);
    initialData(h_B, nElem);
    elapsed = seconds() - start;

    printf("Initial Data Time = %f sec\n", elapsed);

    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    // CPU Computation
    start = seconds();
    sumArraysOnHost(h_A, h_B, hostRef, nElem);
    elapsed = seconds() - start;

    printf("CPU Time = %f sec\n", elapsed);

    // Allocate Device Memory
    float *d_A, *d_B, *d_C;

    CHECK(cudaMalloc((void **)&d_A, nBytes));
    CHECK(cudaMalloc((void **)&d_B, nBytes));
    CHECK(cudaMalloc((void **)&d_C, nBytes));

    // Copy Input Data
    CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice));

    // Initialize Output
    CHECK(cudaMemset(d_C, 0, nBytes));

    // Launch Configuration
    int iLen = 1024;
    dim3 block(iLen);
    dim3 grid((nElem + block.x - 1) / block.x);

    // GPU Computation
    start = seconds();

    sumArraysOnGPU<<<grid, block>>>(d_A, d_B, d_C, nElem);

    CHECK(cudaDeviceSynchronize());

    elapsed = seconds() - start;

    printf("Grid (Blocks)      = %d\n", grid.x);
    printf("Threads per Block  = %d\n", block.x);
    printf("GPU Time = %f sec\n", elapsed);

    CHECK(cudaGetLastError());

    // Copy Result Back
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    // Verify Result
    checkResult(hostRef, gpuRef, nElem);

    // Free Memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);

    return 0;
}
```
## OUTPUT:

<img width="524" height="196" alt="image" src="https://github.com/user-attachments/assets/fa9b0a53-b673-469a-a0d2-2e4b7df41748" />     


<img width="430" height="192" alt="image" src="https://github.com/user-attachments/assets/c11f1cbd-7fbf-4ae5-9ce2-2caa7080cde2" />   


<img width="351" height="181" alt="image" src="https://github.com/user-attachments/assets/31e42b8b-76a0-49d5-b9d7-5cc38def0a7e" />

## RESULT:
Thus, Implementation of sum arrays on host and device is done in nvcc cuda using random number.
