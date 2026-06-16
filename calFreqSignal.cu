#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <thrust/complex.h>

__global__ void FarassatFreqCUDAKernel(
    double* srcFreq, const int srcFreqSize,
    double* srcArea, const int srcNum,
	const double3 M, const double C0,
    const thrust::complex<double>* srcQnCplx,
    const thrust::complex<double>* srcLi0Cplx,
	const thrust::complex<double>* srcLi1Cplx,
	const thrust::complex<double>* srcLi2Cplx,
    const double* R,const double3* Rhat, const double* timeDelay,
    const int obsNum, double* pPrimeCplxReal, double* pPrimeCplxImag)
{
    const size_t obsI = blockIdx.z * blockDim.z + threadIdx.z;
    const size_t srcI = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t FreqI = blockIdx.x * blockDim.x + threadIdx.x;
    thrust::complex<double> I(0.0,1.0);
    
	if (obsI < obsNum && srcI < srcNum && FreqI < srcFreqSize)
    {
        
        const double pi = 2. * acos(0.0);
        size_t SOIdx = srcI*obsNum + obsI;
		size_t TSIdx = FreqI*srcNum + srcI;
		size_t TOIdx = FreqI*obsNum + obsI;

        double rMag = R[SOIdx];
        
        double omega = 2*pi*srcFreq[FreqI];
        double td = timeDelay[SOIdx];
        thrust::complex<double> Qn  = srcQnCplx[TSIdx];
        thrust::complex<double> Li0 = srcLi0Cplx[TSIdx];
        thrust::complex<double> Li1 = srcLi1Cplx[TSIdx];
        thrust::complex<double> Li2 = srcLi2Cplx[TSIdx];
        thrust::complex<double> QnDot = I * omega * Qn;
		thrust::complex<double> Li0Dot = I * omega * Li0;
		thrust::complex<double> Li1Dot = I * omega * Li1;
		thrust::complex<double> Li2Dot = I * omega * Li2;
        thrust::complex<double> LM = M.x * Li0 + M.y * Li1 + M.z * Li2;
        double3 rHat = Rhat[SOIdx];
        double dS = srcArea[srcI];
        double MSqr = pow(M.x, 2) + pow(M.y,2) + pow(M.z,2);
        double Mr = M.x*rHat.x + M.y*rHat.y + M.z*rHat.z;
        thrust::complex<double> LrDot = Li0Dot * rHat.x + Li1Dot * rHat.y + Li2Dot * rHat.z;
		thrust::complex<double> Lr = Li0 * rHat.x + Li1 * rHat.y + Li2 * rHat.z;

				// these did not include moving surface terms, e.g. MDot, rDot, nDot, vDot?
		thrust::complex<double> S1 = QnDot / rMag / (1 - Mr) / (1 - Mr);
		thrust::complex<double> S2 = Qn * C0 * (Mr - MSqr) / pow(rMag, 2) / pow(1 - Mr, 3);
		thrust::complex<double> S3 = LrDot / C0 / rMag / pow(1 - Mr, 2);
		thrust::complex<double> S4 = (Lr - LM) / pow(rMag, 2) / pow(1 - Mr, 2);
		thrust::complex<double> S5 = Lr * (Mr - MSqr) / pow(rMag, 2) / pow(1 - Mr, 3);
		thrust::complex<double> pTotal = (S1 + S2 + S3 + S4 + S5) * dS / 4. / pi;
        pTotal = pTotal * thrust::exp(-I * omega * td);
        atomicAdd(&pPrimeCplxReal[TOIdx], pTotal.real());
        atomicAdd(&pPrimeCplxImag[TOIdx], pTotal.imag());
    }
}


extern "C" void CalFreqSignalCu(double* srcFreq, const int srcFreqSize,
    double* srcArea, const int srcNum,
	const double3 M, const double C0,
    const thrust::complex<double>* srcQnCplx,
    const thrust::complex<double>* srcLi0Cplx,
	const thrust::complex<double>* srcLi1Cplx,
	const thrust::complex<double>* srcLi2Cplx,
    const double* R,const double3* Rhat, const double* timeDelay,
    const int obsNum, double* pPrimeCplxReal, double* pPrimeCplxImag,int myRank,int commSize)
    {

    //按进程数量划分GPU块
    int gpuCount;
    cudaGetDeviceCount(&gpuCount);

    //int deviceId = (myRank * gpuCount) / commSize;

    if (myRank >= gpuCount) {
    fprintf(stderr, "Error: myRank=%d >= gpuCount=%d, no GPU available.\n", myRank, gpuCount);
    std::exit(EXIT_FAILURE);
    }

    //int deviceId = myRank;  // 保证 myRank 与 GPU 数量一致
    cudaSetDevice(myRank);

    int deviceId;
    cudaGetDevice(&deviceId);
    printf("Process (localRank %d) is using GPU %d in CalFreqSignalCu\n", myRank, deviceId);
    fflush(stdout);


    double *d_srcFreq, *d_srcArea,  *d_R, *d_timeDelay, *d_pPrimeCplxReal, *d_pPrimeCplxImag;
    double3  *d_Rhat;
    thrust::complex<double> *d_srcLi0Cplx, *d_srcLi1Cplx, *d_srcLi2Cplx, *d_srcQnCplx;


    cudaMalloc((void**)&d_srcFreq, srcFreqSize * sizeof(double));
    cudaMalloc((void**)&d_srcArea, srcNum * sizeof(double));
    cudaMalloc((void**)&d_srcQnCplx, srcFreqSize * srcNum * sizeof(thrust::complex<double>));
    cudaMalloc((void**)&d_srcLi0Cplx, srcFreqSize * srcNum * sizeof(thrust::complex<double>));
    cudaMalloc((void**)&d_srcLi1Cplx, srcFreqSize * srcNum * sizeof(thrust::complex<double>));
    cudaMalloc((void**)&d_srcLi2Cplx, srcFreqSize * srcNum * sizeof(thrust::complex<double>));
    cudaMalloc((void**)&d_R, srcNum* obsNum * sizeof(double));
    cudaMalloc((void**)&d_timeDelay, srcNum* obsNum * sizeof(double));
    cudaMalloc((void**)&d_Rhat, srcNum* obsNum * sizeof(double3));
    cudaMalloc((void**)&d_pPrimeCplxReal, srcFreqSize * obsNum * sizeof(double));
    cudaMalloc((void**)&d_pPrimeCplxImag, srcFreqSize * obsNum * sizeof(double));
    // 复制数据到 GPU
    cudaMemcpy(d_srcFreq, srcFreq, srcFreqSize * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_srcArea, srcArea, srcNum * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_srcQnCplx, srcQnCplx, srcFreqSize * srcNum * sizeof(thrust::complex<double>), cudaMemcpyHostToDevice); 
    cudaMemcpy(d_srcLi0Cplx, srcLi0Cplx, srcFreqSize * srcNum * sizeof(thrust::complex<double>), cudaMemcpyHostToDevice);
    cudaMemcpy(d_srcLi1Cplx, srcLi1Cplx, srcFreqSize * srcNum * sizeof(thrust::complex<double>), cudaMemcpyHostToDevice);
    cudaMemcpy(d_srcLi2Cplx, srcLi2Cplx, srcFreqSize * srcNum * sizeof(thrust::complex<double>), cudaMemcpyHostToDevice);
    cudaMemcpy(d_R, R, srcNum* obsNum * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Rhat, Rhat, srcNum* obsNum * sizeof(double3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_timeDelay, timeDelay, srcNum* obsNum * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pPrimeCplxReal, pPrimeCplxReal, srcFreqSize * obsNum * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pPrimeCplxImag, pPrimeCplxImag, srcFreqSize * obsNum * sizeof(double), cudaMemcpyHostToDevice);
    dim3 threadsPerBlock(16, 16, 1);

    dim3 numBlocks((srcFreqSize + threadsPerBlock.x - 1) / threadsPerBlock.x,
                    (srcNum + threadsPerBlock.y - 1) / threadsPerBlock.y,
                    (obsNum + threadsPerBlock.z -1)/threadsPerBlock.z);
    printf("Start computing!\n");


    cudaEvent_t start, stop;
    float time;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);



    FarassatFreqCUDAKernel<<<numBlocks,threadsPerBlock>>>(d_srcFreq, srcFreqSize,
        d_srcArea, srcNum,
        M, C0,
        d_srcQnCplx, 
        d_srcLi0Cplx, d_srcLi1Cplx, d_srcLi2Cplx, 
        d_R, d_Rhat, d_timeDelay,
        obsNum,  d_pPrimeCplxReal, d_pPrimeCplxImag);
        cudaDeviceSynchronize();
        cudaError_t error = cudaGetLastError();
    if (error!= cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(1);
    }

    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    printf("Execution time: %f ms\n", time);
    fflush(stdout);

    //printf("Process %d is using GPU %d in CalFreqSignalCu\n", myRank, deviceId);


    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaMemcpy(pPrimeCplxReal, d_pPrimeCplxReal, srcFreqSize * obsNum * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(pPrimeCplxImag, d_pPrimeCplxImag, srcFreqSize * obsNum * sizeof(double), cudaMemcpyDeviceToHost);


    cudaFree(d_srcFreq);
    cudaFree(d_srcArea);
    cudaFree(d_srcQnCplx);
    cudaFree(d_srcLi0Cplx);
    cudaFree(d_srcLi1Cplx);
    cudaFree(d_srcLi2Cplx);
    cudaFree(d_R);
    cudaFree(d_Rhat);
    cudaFree(d_pPrimeCplxReal);
    cudaFree(d_pPrimeCplxImag);
    
}