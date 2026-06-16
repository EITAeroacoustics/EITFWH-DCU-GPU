#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <thrust/complex.h>
#include <chrono>
#include <cstdlib>



__host__ __device__ thrust::complex<double> Green (double3 x, double3 y, double omega, double M, double C)
{
	thrust::complex<double> I(0.0,1.0);
	double betasqr = 1.0-M*M;
    double r1 = x.x - y.x;
    double r2 = x.y - y.y;
    double R2D = sqrt(pow(r1, 2) + betasqr * pow(r2, 2));

		double Hk = omega / C / betasqr * R2D;
		double H1 = j0(Hk);
		double H2 = y0(Hk);
		//printf("%f ; %f ; %f\n", omega, M, C);
		//printf("%f ; %f ; %f ; %f ; %f ; %f\n",  r1 , r2 , R2D , Hk , H1 , H2);
		thrust::complex<double> Hkel(H1,-H2);
		//printf("%f + %f i\n",Hkel.real(), Hkel.imag());
		return I/4.0/sqrt(betasqr)*thrust::exp(I*M*omega/C/betasqr*r1)*Hkel;
	
}

__host__ __device__ thrust::complex<double> dGdy (double3 x, double3 y, double omega, double M, double C)
{
	double h = 1e-6;
    double3 ytmp1 = {y.x,y.y+h/2,0.0};
    double3 ytmp2 = {y.x,y.y-h/2,0.0};
	thrust::complex<double> G1 = Green(x, ytmp1, omega, M, C);
	thrust::complex<double> G2 = Green(x, ytmp2, omega, M, C);
	return (G1-G2)/h; 
}


__host__ __device__ thrust::complex<double> dGdx (double3 x, double3 y,  double omega, double M, double C)
{
	double h = 1e-6;
    double3 ytmp1 = {y.x+h/2,y.y,0.0};
    double3 ytmp2 = {y.x-h/2,y.y,0.0};
	thrust::complex<double> G1 = Green(x, ytmp1, omega, M, C);
	thrust::complex<double> G2 = Green(x, ytmp2, omega, M, C);
	return (G1-G2)/h; 
}




__global__ void FarassatFreq2DCUDAKernel(
    double* srcFreq, const int srcFreqSize,
    double srcArea, const int srcNum,
	const double M, const double C0,
    const thrust::complex<double>* srcQnCplx,
    const thrust::complex<double>* srcLi0Cplx,
	const thrust::complex<double>* srcLi1Cplx,
    const double3* xPos,const double3* yPos,
    const int obsNum, double* pPrimeCplxReal, double* pPrimeCplxImag)
{
    const size_t obsI = blockIdx.z * blockDim.z + threadIdx.z;
    const size_t srcI = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t FreqI = blockIdx.x * blockDim.x + threadIdx.x;
    thrust::complex<double> I(0.0,1.0);
    
	if (obsI < obsNum && srcI < srcNum && FreqI < srcFreqSize)
    {
        
        const double pi = 2. * acos(0.0);
		size_t TSIdx = FreqI*srcNum + srcI;
		size_t TOIdx = FreqI*obsNum + obsI;
        const double3 x = xPos[obsI];
        const double3 y = yPos[srcI];
        double omega = 2*pi*srcFreq[FreqI];
        thrust::complex<double> G1 = Green(x, y, omega, M, C0);
		thrust::complex<double> dGx = dGdx(x, y, omega, M, C0);
		thrust::complex<double> dGy = dGdy(x, y, omega, M, C0);
        double dS = srcArea;
        
        thrust::complex<double> Qn  = srcQnCplx[TSIdx];
        thrust::complex<double> Li0 = srcLi0Cplx[TSIdx];
        thrust::complex<double> Li1 = srcLi1Cplx[TSIdx];

        thrust::complex<double> S1 = I * omega * Qn * G1;
		thrust::complex<double> S2 = Li0 * dGx;
		thrust::complex<double> S3 = Li1 * dGy;
		thrust::complex<double> pTotal = - (S1 + S2 + S3) * dS;
        atomicAdd(&pPrimeCplxReal[TOIdx], pTotal.real());
        atomicAdd(&pPrimeCplxImag[TOIdx], pTotal.imag());
    }
}






extern "C" void CalFreqSignalCu2D(double* srcFreq, const int srcFreqSize,
    double srcArea, const int srcNum,
    const double M, const double C0,
    const thrust::complex<double>* srcQnCplx,
    const thrust::complex<double>* srcLi0Cplx,
    const thrust::complex<double>* srcLi1Cplx,
    const double3* xPos,const double3* yPos,
    const int obsNum, double* pPrimeCplxReal, double* pPrimeCplxImag,
    int myRank,int commSize,
    double* h2dMs, double* d2hMs,
    unsigned long long* h2dBytes, unsigned long long* d2hBytes,
    double* kernelMs)
    {

    int gpuCount;
    cudaGetDeviceCount(&gpuCount);

    //int deviceId = (myRank * gpuCount) / commSize;

    if (myRank >= gpuCount) {
    fprintf(stderr, "Error: localRank=%d >= gpuCount=%d, no GPU available.\n", myRank, gpuCount);
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
    printf("Process (localRank %d) is using GPU %d in CalFreqSignalCu2D\n", myRank, deviceId);
    fflush(stdout);
    

    double *d_srcFreq, *d_pPrimeCplxReal, *d_pPrimeCplxImag;
    double3  *d_xPos, *d_yPos;
    thrust::complex<double> *d_srcLi0Cplx, *d_srcLi1Cplx, *d_srcQnCplx;

    cudaMalloc((void**)&d_srcFreq, srcFreqSize * sizeof(double));
    //cudaMalloc((void**)&d_srcArea, srcNum * sizeof(double));
    cudaMalloc((void**)&d_srcQnCplx, srcFreqSize * srcNum * sizeof(thrust::complex<double>));
    cudaMalloc((void**)&d_srcLi0Cplx, srcFreqSize * srcNum * sizeof(thrust::complex<double>));
    cudaMalloc((void**)&d_srcLi1Cplx, srcFreqSize * srcNum * sizeof(thrust::complex<double>));

    cudaMalloc((void**)&d_xPos, obsNum * sizeof(double3));
    cudaMalloc((void**)&d_yPos, srcNum * sizeof(double3));
    cudaMalloc((void**)&d_pPrimeCplxReal, srcFreqSize * obsNum * sizeof(double));
    cudaMalloc((void**)&d_pPrimeCplxImag, srcFreqSize * obsNum * sizeof(double));
    
    
    auto h2dStart = std::chrono::high_resolution_clock::now();

    cudaMemcpy(d_srcFreq, srcFreq, srcFreqSize * sizeof(double), cudaMemcpyHostToDevice);
    h2dBytesLocal += (unsigned long long)srcFreqSize * sizeof(double);

    cudaMemcpy(d_srcQnCplx, srcQnCplx, srcFreqSize * srcNum * sizeof(thrust::complex<double>), cudaMemcpyHostToDevice);
    h2dBytesLocal += (unsigned long long)srcFreqSize * (unsigned long long)srcNum * sizeof(thrust::complex<double>);

    cudaMemcpy(d_srcLi0Cplx, srcLi0Cplx, srcFreqSize * srcNum * sizeof(thrust::complex<double>), cudaMemcpyHostToDevice);
    h2dBytesLocal += (unsigned long long)srcFreqSize * (unsigned long long)srcNum * sizeof(thrust::complex<double>);

    cudaMemcpy(d_srcLi1Cplx, srcLi1Cplx, srcFreqSize * srcNum * sizeof(thrust::complex<double>), cudaMemcpyHostToDevice);
    h2dBytesLocal += (unsigned long long)srcFreqSize * (unsigned long long)srcNum * sizeof(thrust::complex<double>);

    cudaMemcpy(d_xPos, xPos, obsNum * sizeof(double3), cudaMemcpyHostToDevice);
    h2dBytesLocal += (unsigned long long)obsNum * sizeof(double3);

    cudaMemcpy(d_yPos, yPos, srcNum * sizeof(double3), cudaMemcpyHostToDevice);
    h2dBytesLocal += (unsigned long long)srcNum * sizeof(double3);

    cudaMemcpy(d_pPrimeCplxReal, pPrimeCplxReal, srcFreqSize * obsNum * sizeof(double), cudaMemcpyHostToDevice);
    h2dBytesLocal += (unsigned long long)srcFreqSize * (unsigned long long)obsNum * sizeof(double);

    cudaMemcpy(d_pPrimeCplxImag, pPrimeCplxImag, srcFreqSize * obsNum * sizeof(double), cudaMemcpyHostToDevice);
    h2dBytesLocal += (unsigned long long)srcFreqSize * (unsigned long long)obsNum * sizeof(double);

    cudaDeviceSynchronize();
    auto h2dEnd = std::chrono::high_resolution_clock::now();

    if (h2dMs) {
        *h2dMs = std::chrono::duration<double, std::milli>(h2dEnd - h2dStart).count();
    }
    if (h2dBytes) {
        *h2dBytes = h2dBytesLocal;
    }

    dim3 threadsPerBlock(16, 16, 1);

    dim3 numBlocks((srcFreqSize + threadsPerBlock.x - 1) / threadsPerBlock.x,
                    (srcNum + threadsPerBlock.y - 1) / threadsPerBlock.y,
                    (obsNum + threadsPerBlock.z -1)/threadsPerBlock.z);
    //printf("Start computing!\n");

    cudaEvent_t start, stop;
    float time;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);




    FarassatFreq2DCUDAKernel<<<numBlocks,threadsPerBlock>>>(d_srcFreq, srcFreqSize,
        srcArea, srcNum,
        M, C0,
        d_srcQnCplx, 
        d_srcLi0Cplx, d_srcLi1Cplx,
        d_xPos, d_yPos,
        obsNum,  d_pPrimeCplxReal, d_pPrimeCplxImag);
        cudaDeviceSynchronize(); 
        cudaError_t error = cudaGetLastError();
    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    if (kernelMs) {
    *kernelMs = (double)time;
    }


    
    printf("Execution time: %f ms\n", time);
    fflush(stdout);

    //printf("Process %d is using GPU %d in CalFreqSignalCu2D\n", myRank, deviceId);


    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    if (error!= cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(1);
    }

    auto d2hStart = std::chrono::high_resolution_clock::now();

    cudaMemcpy(pPrimeCplxReal, d_pPrimeCplxReal,
            srcFreqSize * obsNum * sizeof(double),
            cudaMemcpyDeviceToHost);
    d2hBytesLocal += (unsigned long long)srcFreqSize * (unsigned long long)obsNum * sizeof(double);

    cudaMemcpy(pPrimeCplxImag, d_pPrimeCplxImag,
            srcFreqSize * obsNum * sizeof(double),
            cudaMemcpyDeviceToHost);
    d2hBytesLocal += (unsigned long long)srcFreqSize * (unsigned long long)obsNum * sizeof(double);

    cudaDeviceSynchronize();
    auto d2hEnd = std::chrono::high_resolution_clock::now();

    if (d2hMs) {
        *d2hMs = std::chrono::duration<double, std::milli>(d2hEnd - d2hStart).count();
    }
    if (d2hBytes) {
        *d2hBytes = d2hBytesLocal;
    }
    

    cudaFree(d_srcFreq);
    //cudaFree(d_srcArea);
    cudaFree(d_srcQnCplx);
    cudaFree(d_srcLi0Cplx);
    cudaFree(d_srcLi1Cplx);

    cudaFree(d_xPos);
    cudaFree(d_yPos);
    cudaFree(d_pPrimeCplxReal);
    cudaFree(d_pPrimeCplxImag);
    
}