#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <chrono>
#include <cstdlib>


__host__ __device__ double dot(double3 A, double3 B)
{
    return A.x*B.x + A.y*B.y + A.z*B.z;
}

__host__ __device__ double magSqr(double3 A)
{
    return (A.x*A.x + A.y*A.y + A.z*A.z);
}

__global__ void HelloFromGPU()
{
    printf("Hello from GPU\n");
}


__global__ void FarassatCUDAKernel(
	const double* srcTime, const size_t srcTimeSize,
    const double* srcArea, const size_t srcNum,
    const double3 M, const double C0,
    const double* srcQn, const double* srcQnDot,
    const double3* srcLi, const double3* srcLiDot,
    const double* R, const double3* Rhat, const double* timeDelay,
    const double obsTimeMin, const double deltaT, const size_t obsTimeNum,
    const size_t obsNum, double* pPrime
)
{
	size_t obsI = blockIdx.z * blockDim.z + threadIdx.z;
    size_t srcI = blockIdx.y * blockDim.y + threadIdx.y;
    size_t timeI = blockIdx.x * blockDim.x + threadIdx.x;
	if (obsI < obsNum && srcI < srcNum && timeI < srcTimeSize) {
        

		size_t SOIdx = srcI*obsNum + obsI;
		size_t TSIdx = timeI*srcNum + srcI;
		//size_t TOIdx = timeI*obsNum + obsI;
        double rMag = R[SOIdx];
        double Qn  = srcQn[TSIdx];
        double QnDot = srcQnDot[TSIdx];
        double3 Li = srcLi[TSIdx];
        double3 LiDot = srcLiDot[TSIdx];
        double3 rHat = Rhat[SOIdx];
        double dS = srcArea[srcI];

        double advTime = timeDelay[SOIdx] + srcTime[timeI];
        double Mr = dot(rHat, M);
        double MSqr = magSqr(M);
        double LrDot = dot(LiDot, rHat);
        double LM = dot(Li, M);
        double Lr = dot(Li, rHat);
        
        double S1 = QnDot / rMag / (1 - Mr) / (1 - Mr);
		double S2 = Qn * C0 * (Mr - MSqr) / pow(rMag, 2) / pow(1 - Mr, 3);
		double S3 = LrDot / C0 / rMag / pow(1 - Mr, 2);
		double S4 = (Lr - LM) / pow(rMag, 2) / pow(1 - Mr, 2);
		double S5 = Lr * (Mr - MSqr) / pow(rMag, 2) / pow(1 - Mr, 3);
        
        const double pi = 2. * acos(0.0);
        double pTotal = (S1 + S2 + S3 + S4 + S5) * dS / 4. / pi;
        size_t advTimeLeftIndex = (size_t)floor((advTime - obsTimeMin) / deltaT);
        
        if (advTimeLeftIndex < obsTimeNum) {
            double weight = 1.0 - (advTime - (obsTimeMin + advTimeLeftIndex * deltaT)) / deltaT;
            atomicAdd(&pPrime[obsI + obsNum * advTimeLeftIndex], weight * pTotal);

            size_t advTimeRightIndex = advTimeLeftIndex + 1;
            if (advTimeRightIndex < obsTimeNum) {
                atomicAdd(&pPrime[obsI + obsNum * advTimeRightIndex], (1.0 - weight) * pTotal);
            }
        }
	}
}

extern "C" void CalTimeSignalCu(double* srcTime, const int srcTimeSize,
    double* srcArea, const int srcNum,
    const double3 M, const double C0,
    const double* srcQn, const double* srcQnDot,
    const double3* srcLi, const double3* srcLiDot,
    const double* R,const double3* Rhat, const double* timeDelay,
    const double obsTimeMin, const double deltaT, const int obsTimeNum,
    const int obsNum, double* pPrime, int myRank, int commSize,
    double* h2dMs, double* d2hMs,
    unsigned long long* h2dBytes, unsigned long long* d2hBytes,
    double* kernelMs)
{

    int gpuCount;
    cudaGetDeviceCount(&gpuCount);

    //int deviceId = (myRank * gpuCount) / commSize;

    if (myRank >= gpuCount) {
    fprintf(stderr, "Error: myRank=%d >= gpuCount=%d, no GPU available.\n", myRank, gpuCount);
    std::exit(EXIT_FAILURE);
    }

    //int deviceId = myRank;  

    cudaSetDevice(myRank);

    if (h2dMs)    *h2dMs = 0.0;
    if (d2hMs)    *d2hMs = 0.0;
    if (h2dBytes) *h2dBytes = 0ULL;
    if (d2hBytes) *d2hBytes = 0ULL;
    if (kernelMs) *kernelMs = 0.0;

    unsigned long long h2dBytesLocal = 0ULL;
    unsigned long long d2hBytesLocal = 0ULL;


    int deviceId;
    cudaGetDevice(&deviceId);
    printf("Process (localRank %d) is using GPU %d in CalTimeSignalcu\n", myRank, deviceId);
    fflush(stdout);
    
    
    //Transform some  data to double3
    
    double *d_srcTime, *d_srcArea, *d_srcQn, *d_srcQnDot, *d_R, *d_timeDelay, *d_pPrime;
    double3 *d_srcLi, *d_srcLiDot, *d_Rhat;
    

    cudaMalloc((void**)&d_srcTime, srcTimeSize * sizeof(double));
    cudaMalloc((void**)&d_srcArea, srcNum * sizeof(double));
    cudaMalloc((void**)&d_srcQn, srcTimeSize * srcNum * sizeof(double));
    cudaMalloc((void**)&d_srcQnDot, srcTimeSize * srcNum * sizeof(double));
    cudaMalloc((void**)&d_srcLi, srcTimeSize * srcNum * sizeof(double3));
    cudaMalloc((void**)&d_srcLiDot, srcTimeSize * srcNum * sizeof(double3));
    cudaMalloc((void**)&d_R, srcNum* obsNum * sizeof(double));
    cudaMalloc((void**)&d_Rhat, srcNum* obsNum * sizeof(double3));
    cudaMalloc((void**)&d_timeDelay, srcNum * obsNum * sizeof(double));
    cudaMalloc((void**)&d_pPrime, obsTimeNum * obsNum * sizeof(double));

    
    auto h2dStart = std::chrono::high_resolution_clock::now();

    cudaMemcpy(d_srcTime, srcTime, srcTimeSize * sizeof(double), cudaMemcpyHostToDevice);
    h2dBytesLocal += (unsigned long long)srcTimeSize * sizeof(double);

    cudaMemcpy(d_srcArea, srcArea, srcNum * sizeof(double), cudaMemcpyHostToDevice);
    h2dBytesLocal += (unsigned long long)srcNum * sizeof(double);

    cudaMemcpy(d_srcQn, srcQn, srcTimeSize * srcNum * sizeof(double), cudaMemcpyHostToDevice);
    h2dBytesLocal += (unsigned long long)srcTimeSize * (unsigned long long)srcNum * sizeof(double);

    cudaMemcpy(d_srcQnDot, srcQnDot, srcTimeSize * srcNum * sizeof(double), cudaMemcpyHostToDevice);
    h2dBytesLocal += (unsigned long long)srcTimeSize * (unsigned long long)srcNum * sizeof(double);

    cudaMemcpy(d_srcLi, srcLi, srcTimeSize * srcNum * sizeof(double3), cudaMemcpyHostToDevice);
    h2dBytesLocal += (unsigned long long)srcTimeSize * (unsigned long long)srcNum * sizeof(double3);

    cudaMemcpy(d_srcLiDot, srcLiDot, srcTimeSize * srcNum * sizeof(double3), cudaMemcpyHostToDevice);
    h2dBytesLocal += (unsigned long long)srcTimeSize * (unsigned long long)srcNum * sizeof(double3);

    cudaMemcpy(d_R, R, srcNum * obsNum * sizeof(double), cudaMemcpyHostToDevice);
    h2dBytesLocal += (unsigned long long)srcNum * (unsigned long long)obsNum * sizeof(double);

    cudaMemcpy(d_Rhat, Rhat, srcNum * obsNum * sizeof(double3), cudaMemcpyHostToDevice);
    h2dBytesLocal += (unsigned long long)srcNum * (unsigned long long)obsNum * sizeof(double3);

    cudaMemcpy(d_timeDelay, timeDelay, srcNum * obsNum * sizeof(double), cudaMemcpyHostToDevice);
    h2dBytesLocal += (unsigned long long)srcNum * (unsigned long long)obsNum * sizeof(double);
    
    cudaDeviceSynchronize();
    auto h2dEnd = std::chrono::high_resolution_clock::now();

    if (h2dMs) {
        *h2dMs = std::chrono::duration<double, std::milli>(h2dEnd - h2dStart).count();
    }
    if (h2dBytes) {
        *h2dBytes = h2dBytesLocal;
    }
    
    
    //cudaMemcpy(d_pPrime, pPrime, obsTimeNum * obsNum * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemset(d_pPrime, 0, obsTimeNum * obsNum * sizeof(double));  


    dim3 threadsPerBlock(16, 16, 1);

    dim3 numBlocks((srcTimeSize + threadsPerBlock.x - 1) / threadsPerBlock.x,
               (srcNum + threadsPerBlock.y - 1) / threadsPerBlock.y,
               (obsNum + threadsPerBlock.z - 1) / threadsPerBlock.z);
    //printf("Start computing!\n");


    cudaEvent_t start, stop;
    float time;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);


    
    FarassatCUDAKernel<<<numBlocks,threadsPerBlock>>>(d_srcTime, srcTimeSize,
    d_srcArea, srcNum,
	M, C0,
    d_srcQn, d_srcQnDot,
    d_srcLi, d_srcLiDot,
    d_R, d_Rhat, d_timeDelay,
    obsTimeMin, deltaT, obsTimeNum,
    obsNum,  d_pPrime);
    cudaDeviceSynchronize(); 
    cudaError_t error = cudaGetLastError();
    if (error!= cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(1);
    }
    

    cudaEventRecord(stop, 0);


    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    if (kernelMs) {
    *kernelMs = (double)time;
    }
    
    printf("Execution time: %f ms\n", time);
    fflush(stdout);

    //printf("Process %d is using GPU %d in CalTimeSignalcu\n", myRank, deviceId);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    auto d2hStart = std::chrono::high_resolution_clock::now();

    cudaMemcpy(pPrime, d_pPrime,
               obsTimeNum * obsNum * sizeof(double),
               cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    auto d2hEnd = std::chrono::high_resolution_clock::now();

    d2hBytesLocal += (unsigned long long)obsTimeNum * (unsigned long long)obsNum * sizeof(double);

    if (d2hMs) {
        *d2hMs = std::chrono::duration<double, std::milli>(d2hEnd - d2hStart).count();
    }
    if (d2hBytes) {
        *d2hBytes = d2hBytesLocal;
    }
    

    cudaFree(d_srcTime);
    cudaFree(d_srcArea);
    cudaFree(d_srcQn);
    cudaFree(d_srcQnDot);
    cudaFree(d_srcLi);
    cudaFree(d_srcLiDot);
    cudaFree(d_R);
    cudaFree(d_Rhat);
    cudaFree(d_timeDelay);
    cudaFree(d_pPrime);
}

