#include "Farrasat1A.h"
#include <cfloat>
#include <cuda_runtime.h>
#include <thrust/complex.h>
#include <iomanip>
#include <cstdio>


Farrasat1A::Farrasat1A(
	double MxIn,
	double C0In,
	double rhoIn,
	double pIn,
	bool isPenetrableIn,
	int rankIn) :
	M0(MxIn),
	C0(C0In),
	rho0(rhoIn),
	p0(pIn),
	isPenetrable(isPenetrableIn),
	isMaster(!rankIn)
{
	// Initialize();
};

Farrasat1A::~Farrasat1A()
{
}

extern "C" void CalTimeSignalCu(double* srcTime, const int srcTimeSize,
    double* srcArea, const int srcNum,
    const double3 M, const double C0,
    const double* srcQn, const double* srcQnDot,
    const double3* srcLi, const double3* srcLiDot,
    const double* R,const double3* Rhat, const double* timeDelay,
    const double obsTimeMin, const double deltaT, const int obsTimeNum,
    const int obsLocationSize, double* pPrime, int myRank, int commSize,
    double* h2dMs, double* d2hMs,
    unsigned long long* h2dBytes, unsigned long long* d2hBytes,
    double* kernelMs);

extern "C" void CalFreqSignalCu(double* srcFreq, const int srcFreqSize,
    double* srcArea, const int srcNum,
	const double3 M, const double C0,
    const thrust::complex<double>* srcQn,
    const thrust::complex<double>* srcLi0Cplx,
	const thrust::complex<double>* srcLi1Cplx,
	const thrust::complex<double>* srcLi2Cplx,
    const double* R,const double3* Rhat, const double* timeDelay,
    const int obsLocationSize, double* pPrimeCplxReal, double* pPrimeCplxImag,int myRank,int commSize);

extern "C" void CalFreqSignalCu2D(double* srcFreq, const int srcFreqSize,
    double srcArea, const int srcNum,
    const double M, const double C0,
    const thrust::complex<double>* srcQn,
    const thrust::complex<double>* srcLi0Cplx,
    const thrust::complex<double>* srcLi1Cplx,
    const double3* xPos,const double3* yPos,
    const int obsLocationSize, double* pPrimeCplxReal, double* pPrimeCplxImag,
    int myRank,int commSize,
    double* h2dMs, double* d2hMs,
    unsigned long long* h2dBytes, unsigned long long* d2hBytes,
    double* kernelMs);

// Define the time values for observers
void Farrasat1A::InitializeObserverTime(double tMin, double tMax, size_t tNum)
{
	obsTimeMin = tMin;
	obsTimeMax = tMax;
	obsTimeNum = tNum;
	double dt = (obsTimeMax - obsTimeMin) / (obsTimeNum - 1.0);
	for (size_t i = 0; i < obsTimeNum; i++)
		obsTime.push_back(double(i) * dt + obsTimeMin);
}

double Farrasat1A::GetObserverTimeValue(size_t iStep) {
	return obsTime[iStep];
}

double Farrasat1A::GetMaximumObserverTime() {
	return obsTimeMax;
}

double Farrasat1A::GetMinimumObserverTime() {
	return obsTimeMin;
}

//*********************************************
size_t Farrasat1A::GetNumberOfObserverTime() {
	return obsTimeNum;
}


void Farrasat1A::CheckIntegralSurface() {
	//
	vec3 avgNormal, avgCenter;
	// 
	double minArea = *min_element(srcArea.begin(), srcArea.end());
	double maxArea = *max_element(srcArea.begin(), srcArea.end());
	double allArea = accumulate(srcArea.begin(), srcArea.end(), 0.0);
	for (size_t ip = 0; ip < srcNum; ip++)
	{
		allArea += srcArea[ip];
		avgNormal += srcNormVec[0][ip] * srcArea[ip];
		avgCenter += srcLocation[0][ip] * srcArea[ip];
	}
	avgNormal /= allArea;
	avgCenter /= allArea;

	if (isMaster) fcout << "Number of surface panels " << srcNum << endl;
	if (isMaster) fcout << "Averaged normal vector " << avgNormal << endl;
	if (isMaster) fcout << "Averaged panel  center " << avgCenter << endl;
	if (isMaster) fcout << "Min / max panel area " << minArea << " / " << maxArea << endl;
	if (isMaster) fcout << "Total aera " << allArea << endl;
	if (isMaster) fcout << "Minumum / maximum length " << sqrt(minArea) << " / " << sqrt(maxArea) << endl;
}

void Farrasat1A::ReadSurface(
	vector<double>srcAreaIn,
	vector<vector<vec3>>  srcLocationIn,
	vector<vector<vec3>>  srcNormVecIn,
	bool isMovingIn) {
	srcArea = srcAreaIn;
	srcLocation = srcLocationIn;
	srcNormVec = srcNormVecIn;
	isMoving = isMovingIn;
	srcNum = srcArea.size();
}

void Farrasat1A::ReadSurface2D(
	double AreaIn2D,
	vector<vector<vec3>>  LocationIn2D,
	vector<vector<vec3>>  NormVecIn2D,
	bool isMovingIn2D) 
{
	
	srcArea2D = AreaIn2D;
	srcLocation2D = LocationIn2D;
	srcNormVec2D = NormVecIn2D;
	isMoving2D = isMovingIn2D;
	
}

void Farrasat1A::SetFlowData(vector<double> srcTimeIn,
	vector<vector<double>> srcDensityIn,
	vector<vector<double>> srcPressureIn,
	vector<vector<vec3>> srcVelocityIn
) {

	srcTime = srcTimeIn;
	srcDensity = srcDensityIn;
	srcPressure = srcPressureIn;
	srcVelocity = srcVelocityIn;
	/*****************************************************************/
}

void Farrasat1A::SetflowData2D(vector<double> flowTimeIn2D,
	vector<vector<double>> flowDensityIn2D,
	vector<vector<double>> flowPressureIn2D,
	vector<vector<vec3>> flowVelocityIn2D
) {

	srcTime2D = flowTimeIn2D;
	srcDensity2D = flowDensityIn2D;
	srcPressure2D = flowPressureIn2D;
	srcVelocity2D = flowVelocityIn2D;
	
}

void Farrasat1A::CalTimeDelayAtStep(size_t it) {

	// TODO: to put it into a initialize function
	// allocate memory space for radiation radius/vector
	timeDelay.resize(srcNum); 
	R.resize(srcNum);
	Rhat.resize(srcNum);
	for (size_t i = 0; i < srcNum; i++)
	{
		timeDelay[i].resize(obsLocation.size()); 
		R[i].resize(obsLocation.size());
		Rhat[i].resize(obsLocation.size());
	}

	const vec3 Mo = { -M0, 0., 0. };
	const double betaSqr = 1.0 - M0 * M0;

	for (size_t i = 0; i < srcNum; i++) 
	{
		for (size_t j = 0; j < obsLocation.size(); j++) 
		{
			const vec3& xPos = obsLocation[j];
			const vec3& yPos = srcLocation[0][i];
			vec3 rHat;
			rHat = xPos - yPos;
			double r = rHat.mag();
			double Mor = dot(Mo, rHat) / r;
			double td = r / C0 * (Mor + sqrt(Mor * Mor + betaSqr)) / betaSqr;

			// Ref: REF D. Casalino 2003 JSV Eq.(39)
			//  from the source location at emission time to the observer location at the advanced time 
			// 
			//	Below formulation assumes a static observer, but it could be easily extened to consider
			//	any subsonic motion
			rHat += Mo * C0 * td;
			r = rHat.mag();

			R[i][j] = r;
			Rhat[i][j] = rHat / r;
			timeDelay[i][j] = r / C0;
		}
	}

}

// This function computes the maximum/minimum time delay, and it
// works for sources in subsonic motion.
void Farrasat1A::CalTimeDelayMinMax() {

	CalTimeDelayAtStep((size_t)0); 

	timeDelayMin = DBL_MAX;
	timeDelayMax = 0.0;
	for (size_t i = 0; i < srcNum; i++)
	{
		timeDelayMin = min(timeDelayMin, *min_element(timeDelay[i].begin(), timeDelay[i].end()));
		timeDelayMax = max(timeDelayMax, *max_element(timeDelay[i].begin(), timeDelay[i].end()));
	}
}

void Farrasat1A::CalSourceTerms() {
	const double deltaT = obsTime[1] - obsTime[0];

	srcQn.resize(srcTime.size());
	srcLi.resize(srcTime.size());
	for (size_t i = 0; i < srcTime.size(); i++)
	{
		srcQn[i].resize(srcNum);
		srcLi[i].resize(srcNum);
	}

	for (size_t i = 0; i < srcTime.size(); i++)
	{
		if(isMaster)   cout << "\r";
		if(isMaster)  fcout << "Progress: " << floor(100 * (i + 1) / srcTime.size()) << " % " << flush;
		for (size_t j = 0; j < srcNum; j++)
		{
			const double& rho = srcDensity[i][j];
			const vec3& n = srcNormVec[0][j];
			const vec3& u = srcVelocity[i][j]+ vec3(-C0 * M0, 0.0, 0.0);
			double p = srcPressure[i][j] - p0;
			// relative to a static medium
			// TODO: consider on-body fw-h: v=u+vec3(-C0 * M0, 0.0, 0.0);
			vec3 v = vec3(-C0 * M0, 0.0, 0.0);
			if (!isPenetrable) v = u;

			// compute source terms
			vec3 Qi = rho0 * v + rho * (u - v);
			srcQn[i][j] = dot(Qi, n);

			// We have neglected the viscous shear force over the data surface 
			// acting on the flId exterior to the surface
			// TODO:
			// The viscous term, I*uj, needs to be included.
			srcLi[i][j] = p * n + rho * u * dot(u - v, n);
		}
	}
	if (isMaster)   cout << endl;
}

void Farrasat1A::CalSourceTerms2D(void)
{
	srcQn2D.resize(srcTime2D.size());
	srcLi2D.resize(srcTime2D.size());
	
	size_t srcNum2D = srcLocation2D[0].size();
	
	for (size_t i = 0; i < srcTime2D.size(); i++)
	{
		srcQn2D[i].resize(srcNum2D);
		srcLi2D[i].resize(srcNum2D);
	}

	for (size_t i = 0; i < srcTime2D.size(); i++)
	{
		if (isMaster)   cout << "\r";
		if (isMaster)  fcout << "Progress: " << floor(100 * (i + 1) / srcTime2D.size()) << " % " << flush;
		for (size_t j = 0; j < srcNum2D; j++)
		{
			
			const double& rho = srcDensity2D[i][j];
			const vec3& n = srcNormVec2D[i][j];
			const vec3& u = srcVelocity2D[i][j];
			double p = srcPressure2D[i][j]; 
			
			// relative to a uniform x0 medium
			vec3 U0= vec3(C0 * M0, 0.0, 0.0);
			
			// compute source terms
			vec3 Qi = rho * u - rho0 * U0;
			srcQn2D[i][j] = dot(Qi, n);

			// We have neglected the viscous shear force over the data surface 
			
			srcLi2D[i][j] = (p-p0) * n + rho * (u - 2 * U0) * dot(u, n)+ rho0 * U0 * dot(U0, n);

		
		}

	}
	if (isMaster)   cout << endl;

}

void Farrasat1A::InterpTimeDerivative() {
	// 1. this assumes a uniform sampling from cfd, i.e., constant time-step
	// 2. this will produce numerical errors, epsecially at high-frequencies
	const double deltaT = obsTime[1] - obsTime[0];
	srcQnDot.resize(srcTime.size());
	srcLiDot.resize(srcTime.size());
	for (size_t i = 0; i < srcTime.size(); i++)
	{
		srcQnDot[i].resize(srcNum);
		srcLiDot[i].resize(srcNum);
	}

	for (size_t i = 0; i < srcTime.size(); i++)
	{
		if (isMaster)   cout << "\r";
		if (isMaster)  fcout << "Progress: " << floor(100 * (i + 1) / srcTime.size()) << " % " << flush;
		if (i == 0) // first step
		{
			// first-order forward difference
			for (size_t j = 0; j < srcNum; j++)
			{
				srcQnDot[i][j] = 0.0;
				srcLiDot[i][j] = vec3(0, 0., 0.);
			}
		}
		else if (i == 1) // second step
		{
			// first-order backward difference
			double oneByDt = 1.0 / deltaT;
			const double bd1w0 = 1.0 * oneByDt;
			const double bd1w1 = -1.0 * oneByDt;
			for (size_t j = 0; j < srcNum; j++)
			{
				srcQnDot[i][j] = bd1w0 * srcQn[i][j] + bd1w1 * srcQn[i - 1][j];
				srcLiDot[i][j] = bd1w0 * srcLi[i][j] + bd1w1 * srcLi[i - 1][j];
			}
		}
		else if (i == 2) // third step
		{
			// second-order backward difference
			double oneByDt = 1.0 / deltaT;
			const double bd2w0 = 1.5 * oneByDt;
			const double bd2w1 = -2.0 * oneByDt;
			const double bd2w2 = 0.5 * oneByDt;
			for (size_t j = 0; j < srcNum; j++)
			{
				srcQnDot[i][j] =
					bd2w0 * srcQn[i][j]
					+ bd2w1 * srcQn[i - 1][j]
					+ bd2w2 * srcQn[i - 2][j];
				srcLiDot[i][j] =
					bd2w0 * srcLi[i][j]
					+ bd2w1 * srcLi[i - 1][j]
					+ bd2w2 * srcLi[i - 2][j];
			}
		}
		else if (i == 3) // forth step
		{
			// third-order backward difference
			double oneByDt = 1.0 / deltaT;
			const double bd3w0 = 11. / 6. * oneByDt;
			const double bd3w1 = -3.0 * oneByDt;
			const double bd3w2 = 1.5 * oneByDt;
			const double bd3w3 = -1. / 3. * oneByDt;
			for (size_t j = 0; j < srcNum; j++)
			{
				srcQnDot[i][j] =
					bd3w0 * srcQn[i][j] + bd3w1 * srcQn[i - 1][j]
					+ bd3w2 * srcQn[i - 2][j]
					+ bd3w3 * srcQn[i - 3][j];
				srcLiDot[i][j] =
					bd3w0 * srcLi[i][j]
					+ bd3w1 * srcLi[i - 1][j]
					+ bd3w2 * srcLi[i - 2][j]
					+ bd3w3 * srcLi[i - 3][j];
			}
		}
		else if (i == 4) // fifth step
		{
			// forth-order backward difference
			double oneByDt = 1.0 / deltaT;
			const double bd4w0 = 25. / 12. * oneByDt;
			const double bd4w1 = -4.0 * oneByDt;
			const double bd4w2 = 3.0 * oneByDt;
			const double bd4w3 = -4. / 3. * oneByDt;
			const double bd4w4 = 1. / 4. * oneByDt;
			for (size_t j = 0; j < srcNum; j++)
			{
				srcQnDot[i][j] =
					bd4w0 * srcQn[i][j]
					+ bd4w1 * srcQn[i - 1][j]
					+ bd4w2 * srcQn[i - 2][j]
					+ bd4w3 * srcQn[i - 3][j]
					+ bd4w4 * srcQn[i - 4][j];
				srcLiDot[i][j] =
					bd4w0 * srcLi[i][j]
					+ bd4w1 * srcLi[i - 1][j]
					+ bd4w2 * srcLi[i - 2][j]
					+ bd4w3 * srcLi[i - 3][j]
					+ bd4w4 * srcLi[i - 4][j];
			}
		}
		else if (i == 5) // sixth step
		{
			// fifth-order backward difference
			double oneByDt = 1.0 / deltaT;
			const double bd5w0 = 137. / 60. * oneByDt;
			const double bd5w1 = -5.0 * oneByDt;
			const double bd5w2 = 5.0 * oneByDt;
			const double bd5w3 = -10. / 3. * oneByDt;
			const double bd5w4 = 5. / 4. * oneByDt;
			const double bd5w5 = -1. / 5. * oneByDt;
			for (size_t j = 0; j < srcNum; j++)
			{
				srcQnDot[i][j] =
					bd5w0 * srcQn[i][j]
					+ bd5w1 * srcQn[i - 1][j]
					+ bd5w2 * srcQn[i - 2][j]
					+ bd5w3 * srcQn[i - 3][j]
					+ bd5w4 * srcQn[i - 4][j]
					+ bd5w5 * srcQn[i - 5][j];
				srcLiDot[i][j] =
					bd5w0 * srcLi[i][j]
					+ bd5w1 * srcLi[i - 1][j]
					+ bd5w2 * srcLi[i - 2][j]
					+ bd5w3 * srcLi[i - 3][j]
					+ bd5w4 * srcLi[i - 4][j]
					+ bd5w5 * srcLi[i - 5][j];
			}
		}
		else
		{
			// sixth-order backward difference
			double oneByDt = 1.0 / deltaT;
			const double bd6w0 = 49. / 20. * oneByDt;
			const double bd6w1 = -6. * oneByDt;
			const double bd6w2 = 15. / 2. * oneByDt;
			const double bd6w3 = -20. / 3. * oneByDt;
			const double bd6w4 = 15. / 4. * oneByDt;
			const double bd6w5 = -6. / 5. * oneByDt;
			const double bd6w6 = 1. / 6. * oneByDt;
			for (size_t j = 0; j < srcNum; j++)
			{
				srcQnDot[i][j] =
					bd6w0 * srcQn[i][j]
					+ bd6w1 * srcQn[i - 1][j]
					+ bd6w2 * srcQn[i - 2][j]
					+ bd6w3 * srcQn[i - 3][j]
					+ bd6w4 * srcQn[i - 4][j]
					+ bd6w5 * srcQn[i - 5][j]
					+ bd6w6 * srcQn[i - 6][j];
				srcLiDot[i][j] =
					bd6w0 * srcLi[i][j]
					+ bd6w1 * srcLi[i - 1][j]
					+ bd6w2 * srcLi[i - 2][j]
					+ bd6w3 * srcLi[i - 3][j]
					+ bd6w4 * srcLi[i - 4][j]
					+ bd6w5 * srcLi[i - 5][j]
					+ bd6w6 * srcLi[i - 6][j];
			}
		}
	}
	if (isMaster)   cout << endl;
}

/*
void Farrasat1A::CalTimeSignal(int myRank,int commSize) {
	
	double CalSourceStartTime = MPI_Wtime();
	CalSourceTerms();
	double CalSourceEndTime = MPI_Wtime();

	if (isMaster)
	{
		cout << "*************************************************" << endl;
		cout << "Calculate 3D-Time Source term Time cost: " << fixed << setprecision(2);
		cout << CalSourceEndTime - CalSourceStartTime << " s" << endl;
		cout << "*************************************************" << endl;
	}
	
	
	InterpTimeDerivative();
	if (isMaster)   cout << " Transform Source Information \r";
	double* d_srcQn = (double*)malloc(sizeof(double)*srcNum*srcTime.size());
    double* d_srcQnDot = (double*)malloc(sizeof(double)*srcNum*srcTime.size());
    double3* d_srcLi = (double3*)malloc(sizeof(double3)*srcNum*srcTime.size());
    double3* d_srcLiDot = (double3*)malloc(sizeof(double3)*srcNum*srcTime.size());
	double* d_pPrime = (double*)malloc(sizeof(double)*obsTimeNum*obsLocation.size());
	double3 d_M = {-M0,0,0};
    for (int i = 0; i<srcTime.size(); ++i)
    {
        for (int j = 0; j<srcNum; ++j)
        {
            d_srcQn[j+i*srcNum] = srcQn[i][j];
            d_srcQnDot[j+i*srcNum] = srcQnDot[i][j];
            d_srcLi[j+i*srcNum].x = srcLi[i][j][0];
            d_srcLi[j+i*srcNum].y = srcLi[i][j][1];
            d_srcLi[j+i*srcNum].z = srcLi[i][j][2];
            d_srcLiDot[j+i*srcNum].x = srcLiDot[i][j][0];
            d_srcLiDot[j+i*srcNum].y = srcLiDot[i][j][1];
            d_srcLiDot[j+i*srcNum].z = srcLiDot[i][j][2];
        }
    }
	if (isMaster)   cout << " Transform Distance Information \r";
	double* d_R = (double*)malloc(sizeof(double)*srcNum*obsLocation.size());
	double3* d_Rhat = (double3*)malloc(sizeof(double3)*srcNum*obsLocation.size());
	double* d_TimeDelay = (double*)malloc(sizeof(double)*srcNum*obsLocation.size());
	for (int i=0; i<srcNum;++i)
	{
		for(int j=0;j<obsLocation.size();++j)
		{
			d_R[j+i*obsLocation.size()] = R[i][j];
			d_Rhat[j+i*obsLocation.size()].x = Rhat[i][j][0];
			d_Rhat[j+i*obsLocation.size()].y = Rhat[i][j][1];
			d_Rhat[j+i*obsLocation.size()].z = Rhat[i][j][2];
			d_TimeDelay[j+i*obsLocation.size()] = timeDelay[i][j];
		}
	}
	double* d_srcArea = (double*)malloc(sizeof(double)*srcNum);
	for(int i = 0;i<srcNum;++i)
	{
		d_srcArea[i] = srcArea[i];
	}
	double* d_srcTime = (double*)malloc(sizeof(double)*srcTime.size());
	for(int i = 0;i<srcTime.size();++i)
	{
		d_srcTime[i] = srcTime[i];
	}
	
	for (int i = 0;i<obsTimeNum;++i)
	{
		for(int j = 0;j<obsLocation.size();++j)
		{
			d_pPrime[j+i*obsLocation.size()] = 0.0;//Initiallize
		}
	}
	// allocate memory space for pressure at observers
	pPrime.resize(obsTimeNum, vector<double>(obsLocation.size(), 0.0));

	if (isMaster)   cout << " Deliver observer to Device \r";
	//double3* obsLocationCUDA = (double3*)malloc(sizeof(double3)*obsLocation.size());
	//cudaMallocManaged(&obsLocationCUDA, sizeof(double3)*obsLocation.size());


	//deliver the data to cuda
	//deliver flow information
	//deliver source terms


	//cuda并行计算
	// this implies a uniform pressure sampling at observers
	// it is ok, for the convenience of signal processing, eg. fft.
	const double deltaT = obsTime[1] - obsTime[0];

	vec3 M(-M0, 0, 0);
	double MSqr = M.magSqr();
	
	CalTimeSignalCu(d_srcTime, srcTime.size(),
    d_srcArea, srcNum,
	d_M, C0,
    d_srcQn, d_srcQnDot,
    d_srcLi, d_srcLiDot,
    d_R, d_Rhat, d_TimeDelay,
    obsTimeMin, deltaT, obsTimeNum,
    obsLocation.size(), d_pPrime,myRank,commSize);
	
	MPI_Barrier(MPI_COMM_WORLD);

	if (isMaster)   cout << "Finish computing! Copy the pressure from GPU to CPU";
	for (int i = 0;i<obsTimeNum;++i)
	{
		for(int j = 0;j<obsLocation.size();++j)
		{
			pPrime[i][j] = d_pPrime[j+i*obsLocation.size()];
		}
	}

	if (isMaster) cout << endl;
}
*/



void Farrasat1A::CalTimeSignal(int myRank, int commSize)
{

    double t0 = MPI_Wtime();
    CalSourceTerms();
    InterpTimeDerivative();
    double t1 = MPI_Wtime();
    if (isMaster) {
        std::cout << "*************************************************\n";
        std::cout << "Calculate 3D-Time Source+Derivative Time cost: "
                  << std::fixed << std::setprecision(2) << (t1 - t0) << " s\n";
        std::cout << "*************************************************\n";
    }

    const int nObs = (int)obsLocation.size();
    const int nSrc = (int)srcNum;
    const int T    = (int)srcTime.size();
	
	pureFWHKernelTime_s = 0.0;
	internalMPITime_s   = 0.0;

	cudaSetDevice(myRank);   

    const double deltaT_obs = (obsTimeNum > 1) ? (obsTime[1] - obsTime[0]) : 1.0;


    size_t freeDev = 0, totalDev = 0;
    cudaError_t ce = cudaMemGetInfo(&freeDev, &totalDev);
    if (ce != cudaSuccess) {
        freeDev = (size_t)2ULL * 1024ULL * 1024ULL * 1024ULL;
        totalDev = freeDev;
    }

    const double DEV_SAFETY = 0.70;
    size_t devBudget = (size_t)(freeDev * DEV_SAFETY);


    auto mem_dev_bytes = [&](int len, int obsTile)->size_t {
        if (len <= 0 || obsTile <= 0) return (size_t)0;
        const size_t a = (size_t)len * (size_t)nSrc * (size_t)64;  // 2*double + 2*double3 = 16 + 48
        const size_t b = (size_t)nSrc * (size_t)obsTile * (size_t)40; // R(8)+Rhat(24)+td(8)
        const size_t c = (size_t)obsTimeNum * (size_t)obsTile * (size_t)8; // pPrime
        const size_t d = (size_t)nSrc * (size_t)8;  // srcArea
        const size_t e = (size_t)len * (size_t)8;   // srcTime
        return a + b + c + d + e;
    };


    const size_t HOST_TILE_BUDGET = (size_t)1ULL * 1024ULL * 1024ULL * 1024ULL;
    auto host_tile_bytes = [&](int len, int obsTile)->size_t {
        const size_t geom = (size_t)nSrc * (size_t)obsTile * (size_t)40;
        const size_t out1 = (size_t)obsTimeNum * (size_t)obsTile * (size_t)8; // tile 
        const size_t out2 = (size_t)obsTimeNum * (size_t)obsTile * (size_t)8; // 
        return geom + out1 + out2;
    };


    int OBS_TILE = nObs;
    int BLOCK_T  = T;

    auto clamp_ge1 = [](long long v)->int { return (int)std::max(1LL, v); };


    if (mem_dev_bytes(T, nObs) <= devBudget && host_tile_bytes(T, nObs) <= HOST_TILE_BUDGET) {
        OBS_TILE = nObs;
        BLOCK_T  = T;
    } else {
        long long numer_dev = (long long)devBudget
            - (long long)((size_t)T * (size_t)nSrc * (size_t)64 + (size_t)nSrc * (size_t)8 + (size_t)T * (size_t)8);
        long long denom_dev = (long long)((size_t)nSrc * (size_t)40 + (size_t)obsTimeNum * (size_t)8);
        long long obsTile_max_dev = (denom_dev > 0) ? numer_dev / denom_dev : 0;

        // perObs_host = nSrc*40 + 2*obsTimeNum*8
        long long perObs_host = (long long)((size_t)nSrc * (size_t)40 + (size_t)2 * (size_t)obsTimeNum * (size_t)8);
        long long obsTile_max_host = (perObs_host > 0) ? (long long)HOST_TILE_BUDGET / perObs_host : 0;

        long long obsTile_try = std::min<long long>((long long)nObs, std::max(0LL, std::min(obsTile_max_dev, obsTile_max_host)));

        if (obsTile_try >= 1 && mem_dev_bytes(T, (int)obsTile_try) <= devBudget
            && host_tile_bytes(T, (int)obsTile_try) <= HOST_TILE_BUDGET) {
            OBS_TILE = (int)obsTile_try;
            BLOCK_T  = T;
        } else {
            long long obsTile_host_only = std::max(1LL, std::min<long long>((long long)nObs, obsTile_max_host));
            OBS_TILE = (int)obsTile_host_only;

            long long numer_len = (long long)devBudget
                - (long long)((size_t)nSrc * (size_t)OBS_TILE * (size_t)40
                              + (size_t)obsTimeNum * (size_t)OBS_TILE * (size_t)8
                              + (size_t)nSrc * (size_t)8);
            long long denom_len = (long long)((size_t)nSrc * (size_t)64 + (size_t)8);
            long long len_max = (denom_len > 0) ? numer_len / denom_len : 0;

            BLOCK_T = clamp_ge1(std::min<long long>((long long)T, len_max));

            while ((mem_dev_bytes(BLOCK_T, OBS_TILE) > devBudget
                    || host_tile_bytes(BLOCK_T, OBS_TILE) > HOST_TILE_BUDGET)
                   && OBS_TILE > 1)
            {
                OBS_TILE = std::max(1, OBS_TILE / 2);
            }
            while ((mem_dev_bytes(BLOCK_T, OBS_TILE) > devBudget) && BLOCK_T > 1) {
                BLOCK_T = std::max(1, BLOCK_T / 2);
            }
        }
    }

    if (isMaster) {
        std::cout << "[Auto Tiling] GPU free " << (freeDev >> 20) << " MB, use <= "
                  << (devBudget >> 20) << " MB\n";
        std::cout << "Chosen BLOCK_T=" << BLOCK_T << " / " << T
                  << ", OBS_TILE=" << OBS_TILE << " / " << nObs << "\n";
    }

	// ===== instrumentation: chunk / transfer statistics =====
    const int numObsTilesPlan   = (nObs + OBS_TILE - 1) / OBS_TILE;
    const int numTimeChunksPlan = (T + BLOCK_T - 1) / BLOCK_T;

    int totalChunks = 0;

    unsigned long long nominalWorksetBytes =
        (unsigned long long)mem_dev_bytes(BLOCK_T, OBS_TILE);
    unsigned long long maxChunkWorksetBytes = 0ULL;
    unsigned long long sumChunkWorksetBytes = 0ULL;

    double h2dMsAccum = 0.0;
    double d2hMsAccum = 0.0;
    unsigned long long h2dBytesAccum = 0ULL;
    unsigned long long d2hBytesAccum = 0ULL;

    pPrime.assign(obsTimeNum, std::vector<double>(nObs, 0.0));

    auto dot3 = [](const double3& a, const double3& b){ return a.x*b.x + a.y*b.y + a.z*b.z; };
    auto mag3 = [](const double3& a){ return std::sqrt(a.x*a.x + a.y*a.y + a.z*a.z); };

    std::vector<double>  h_srcArea(nSrc);
    for (int i = 0; i < nSrc; ++i) h_srcArea[i] = srcArea[i];
    double3 h_M = { -M0, 0.0, 0.0 };

    for (int j0 = 0; j0 < nObs; j0 += OBS_TILE)
    {
        const int jN = std::min(nObs, j0 + OBS_TILE);
        const int tileObs = jN - j0;

        std::vector<double>  h_R((size_t)nSrc * tileObs);
        std::vector<double3> h_Rhat((size_t)nSrc * tileObs);
        std::vector<double>  h_td((size_t)nSrc * tileObs);

        const double betaSqr = 1.0 - M0 * M0;
        const double3 Mo = { -M0, 0.0, 0.0 };

        for (int ip = 0; ip < nSrc; ++ip) {
            const vec3& yv = srcLocation[0][ip]; 
            const double3 yPos = { yv[0], yv[1], yv[2] };

            for (int jj = 0; jj < tileObs; ++jj) {
                const vec3& xv = obsLocation[j0 + jj];
                const double3 xPos = { xv[0], xv[1], xv[2] };

                double3 r0 = { xPos.x - yPos.x, xPos.y - yPos.y, xPos.z - yPos.z };
                double   r  = mag3(r0);
                double   Mor = dot3(Mo, r0) / std::max(r, 1e-300);

                double   td = r / C0 * (Mor + std::sqrt(Mor*Mor + betaSqr)) / betaSqr;
                double3  r1 = { r0.x + Mo.x * C0 * td,
                                r0.y + Mo.y * C0 * td,
                                r0.z + Mo.z * C0 * td };
                double   r1m = mag3(r1);

                const size_t idx = (size_t)jj + (size_t)ip * (size_t)tileObs;
                h_R[idx]    = r1m;
                h_Rhat[idx] = { r1.x / r1m, r1.y / r1m, r1.z / r1m };
                h_td[idx]   = r1m / C0;
            }
        }


        std::vector<double> h_pPrime_tile((size_t)obsTimeNum * tileObs, 0.0);

        auto flat_idx = [&](int itb, int ip) -> size_t { return (size_t)ip + (size_t)itb * (size_t)nSrc; };

        std::vector<double>  blk_Qn((size_t)BLOCK_T * nSrc);
        std::vector<double>  blk_QnDot((size_t)BLOCK_T * nSrc);
        std::vector<double3> blk_Li((size_t)BLOCK_T * nSrc);
        std::vector<double3> blk_LiDot((size_t)BLOCK_T * nSrc);
        std::vector<double>  blk_Time(BLOCK_T);

        for (int t = 0; t < T; t += BLOCK_T)
        {
			const int len = std::min(BLOCK_T, T - t);

    		totalChunks++;

    		unsigned long long thisChunkWorksetBytes =
        		(unsigned long long)mem_dev_bytes(len, tileObs);

			if (thisChunkWorksetBytes > maxChunkWorksetBytes)
				maxChunkWorksetBytes = thisChunkWorksetBytes;

			sumChunkWorksetBytes += thisChunkWorksetBytes;

			
            for (int itb = 0; itb < len; ++itb) {
                const int ig = t + itb;
                blk_Time[itb] = srcTime[ig];

                for (int ip = 0; ip < nSrc; ++ip) {
                    const size_t id = flat_idx(itb, ip);

                    blk_Qn    [id] = srcQn   [ig][ip];
                    blk_QnDot [id] = srcQnDot[ig][ip];

                    blk_Li    [id].x = srcLi   [ig][ip][0];
                    blk_Li    [id].y = srcLi   [ig][ip][1];
                    blk_Li    [id].z = srcLi   [ig][ip][2];

                    blk_LiDot [id].x = srcLiDot[ig][ip][0];
                    blk_LiDot [id].y = srcLiDot[ig][ip][1];
                    blk_LiDot [id].z = srcLiDot[ig][ip][2];
                }
            }

            std::vector<double> h_block_out((size_t)obsTimeNum * tileObs);
            
			double h2dMsThis = 0.0;
			double d2hMsThis = 0.0;
			double kernelMsThis = 0.0;
			unsigned long long h2dBytesThis = 0ULL;
			unsigned long long d2hBytesThis = 0ULL;



			CalTimeSignalCu(
				blk_Time.data(), len,
				h_srcArea.data(), nSrc,
				h_M, C0,
				blk_Qn.data(), blk_QnDot.data(),
				blk_Li.data(), blk_LiDot.data(),
				h_R.data(), h_Rhat.data(), h_td.data(),
				obsTimeMin, deltaT_obs, obsTimeNum,
				tileObs, h_block_out.data(),
				myRank, commSize,
				&h2dMsThis, &d2hMsThis,
				&h2dBytesThis, &d2hBytesThis,
				&kernelMsThis
			);

			pureFWHKernelTime_s += kernelMsThis * 1.0e-3;   // ms -> s

			h2dMsAccum    += h2dMsThis;
			d2hMsAccum    += d2hMsThis;
			h2dBytesAccum += h2dBytesThis;
			d2hBytesAccum += d2hBytesThis;

            double mpi_t0 = MPI_Wtime();
			MPI_Barrier(MPI_COMM_WORLD);
			internalMPITime_s += MPI_Wtime() - mpi_t0;

            for (size_t it = 0; it < (size_t)obsTimeNum; ++it) {
                double* dst = &h_pPrime_tile[it * (size_t)tileObs];
                const double* src = &h_block_out[it * (size_t)tileObs];
                for (int jj = 0; jj < tileObs; ++jj) dst[jj] += src[jj];
            }
        }

        for (size_t it = 0; it < (size_t)obsTimeNum; ++it) {
            for (int jj = 0; jj < tileObs; ++jj) {
                pPrime[it][j0 + jj] = h_pPrime_tile[it * (size_t)tileObs + jj];
            }
        }
    }

	// ===== write per-device statistics to .dat =====
    int globalRank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &globalRank);

    char statFileName[256];
    std::sprintf(statFileName, "dcu_chunk_stats_rank%d.dat", globalRank);

    std::ofstream statFile(statFileName, std::ios::out);
    statFile << "# globalRank localRank nSrc nObs T BLOCK_T OBS_TILE "
             << "numObsTiles numTimeChunks totalChunks "
             << "nominalWorksetBytes maxChunkWorksetBytes sumChunkWorksetBytes "
             << "H2D_ms D2H_ms H2D_bytes D2H_bytes\n";

    statFile << globalRank << " "
             << myRank << " "
             << nSrc << " "
             << nObs << " "
             << T << " "
             << BLOCK_T << " "
             << OBS_TILE << " "
             << numObsTilesPlan << " "
             << numTimeChunksPlan << " "
             << totalChunks << " "
             << nominalWorksetBytes << " "
             << maxChunkWorksetBytes << " "
             << sumChunkWorksetBytes << " "
             << std::fixed << std::setprecision(6)
             << h2dMsAccum << " "
             << d2hMsAccum << " "
             << h2dBytesAccum << " "
             << d2hBytesAccum << "\n";

    statFile.close();

    if (isMaster) std::cout << "Finish computing! (auto-tiling time-domain)\n";
}





void Farrasat1A::CalFreqSpectra2D(int myRank,int commSize)
{
	pureFWHKernelTime_s = 0.0;
	internalMPITime_s   = 0.0;

	cudaSetDevice(myRank); 
	
	double CalSourceStartTime = MPI_Wtime();
	
	CalSourceTerms2D();

	size_t srcTNum = srcTime2D.size();
	size_t srcFrequencyNum = (size_t)floor(srcTNum / 2) + 1;
	size_t srcNum2D = srcLocation2D[0].size();


	const int nObs = (int)obsLocation.size();
	const int nSrc = (int)srcNum2D;
	const int Nf   = (int)srcFrequencyNum;

	const int totalChunks       = 1;
	const int numObsTilesPlan   = 1;
	const int numTimeChunksPlan = 1;


	const int BLOCK_T  = Nf;
	const int OBS_TILE = nObs;

	// d_srcFreq           : Nf * 8
	// d_srcQnCplx         : Nf * nSrc * 16
	// d_srcLi0Cplx        : Nf * nSrc * 16
	// d_srcLi1Cplx        : Nf * nSrc * 16
	// d_xPos              : nObs * 24
	// d_yPos              : nSrc * 24
	// d_pPrimeCplxReal    : Nf * nObs * 8
	// d_pPrimeCplxImag    : Nf * nObs * 8
	auto mem_dev_bytes_2d = [&](int nf, int obsNum)->unsigned long long {
		const unsigned long long a = (unsigned long long)nf * sizeof(double);
		const unsigned long long b = (unsigned long long)nf * (unsigned long long)nSrc * sizeof(thrust::complex<double>) * 3ULL;
		const unsigned long long c = (unsigned long long)obsNum * sizeof(double3);
		const unsigned long long d = (unsigned long long)nSrc * sizeof(double3);
		const unsigned long long e = (unsigned long long)nf * (unsigned long long)obsNum * sizeof(double) * 2ULL;
		return a + b + c + d + e;
	};

	unsigned long long nominalWorksetBytes = mem_dev_bytes_2d(Nf, nObs);
	unsigned long long maxChunkWorksetBytes = nominalWorksetBytes;
	unsigned long long sumChunkWorksetBytes = nominalWorksetBytes;

	double h2dMsAccum = 0.0;
	double d2hMsAccum = 0.0;
	unsigned long long h2dBytesAccum = 0ULL;
	unsigned long long d2hBytesAccum = 0ULL;



	double* in_real;
	fftw_complex* dft_cplx;
	// allocate memory 
	in_real = (double*)fftw_malloc(sizeof(double) * srcTNum);
	dft_cplx = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * srcFrequencyNum);
	
	for (int i = 0; i < srcTNum; i++)
		in_real[i] = 0.;

	for (int i = 0; i < srcFrequencyNum; i++)
	{
		dft_cplx[i][0] = 0.;
		dft_cplx[i][1] = 0.;
	}
	fftw_plan plan = fftw_plan_dft_r2c_1d((int)srcTNum, in_real, dft_cplx, FFTW_ESTIMATE);
	fftw_execute(plan); // initialize the dft plan

	// allocate memory space for pressure at observers
	vector<vector<complex<double>>> QnCplx2D(srcFrequencyNum, vector<complex<double>>(srcNum2D, complex<double>(0., 0.)));
	vector<vector<complex<double>>> Li0Cplx2D(srcFrequencyNum, vector<complex<double>>(srcNum2D, complex<double>(0., 0.)));
	vector<vector<complex<double>>> Li1Cplx2D(srcFrequencyNum, vector<complex<double>>(srcNum2D, complex<double>(0., 0.)));
	vector<vector<complex<double>>> Li2Cplx2D(srcFrequencyNum, vector<complex<double>>(srcNum2D, complex<double>(0., 0.)));

	double scaleFactor = 2.0 / srcTNum;
	if (isMaster)  fcout << "Calculate 2D Fourier transform " << endl;
	for (int ip = 0; ip < srcNum2D; ip++) 
	{
		if (isMaster)   cout << "\r";
		if (isMaster)  fcout << "Progress: " << floor(100 * (ip + 1) / srcNum2D) << " % " << flush;

		// DFT for Qn
		for (int it = 0; it < srcTNum; it++)
		{
			in_real[it] = srcQn2D[it][ip];
		}
		fftw_execute(plan); // run dft
		for (int ik = 0; ik < srcFrequencyNum; ik++)
		{
			QnCplx2D[ik][ip] = complex<double>(dft_cplx[ik][0], dft_cplx[ik][1]) * scaleFactor;
		}

		// DFT for Li0
		for (int it = 0; it < srcTNum; it++)
		{
			in_real[it] = srcLi2D[it][ip][0];
		}
		fftw_execute(plan); // run dft
		for (int ik = 0; ik < srcFrequencyNum; ik++)
		{
			Li0Cplx2D[ik][ip] = complex<double>(dft_cplx[ik][0], dft_cplx[ik][1]) * scaleFactor;
		}

		// DFT for Li1
		for (int it = 0; it < srcTNum; it++)
		{
			in_real[it] = srcLi2D[it][ip][1];
		}
		fftw_execute(plan); // run dft
		for (int ik = 0; ik < srcFrequencyNum; ik++)
		{
			Li1Cplx2D[ik][ip] = complex<double>(dft_cplx[ik][0], dft_cplx[ik][1]) * scaleFactor;
		}


	}
	fftw_destroy_plan(plan);
	if (isMaster) cout << endl;
	//*******************************************//
	//*******************************************//
	const double pi = 2. * acos(0.0);
	//
	const double deltaT = srcTime2D[1] - srcTime2D[0];
	const double samplingFrequency = 1.0 / deltaT;
	srcFrequency2D.resize(srcFrequencyNum, 0.0);
	double deltaF = samplingFrequency / srcTNum;
	for (int i = 0; i < srcFrequencyNum; i++) {
		srcFrequency2D[i] = double(i) * deltaF;
	}
	srcFrequency2D[0] = 1E-8;
	thrust::complex<double>* d_QnCplx2D = (thrust::complex<double>*)malloc(sizeof(thrust::complex<double>)*srcFrequencyNum*srcNum2D);
	thrust::complex<double>* d_Li0Cplx2D = (thrust::complex<double>*)malloc(sizeof(thrust::complex<double>)*srcFrequencyNum*srcNum2D);
	thrust::complex<double>* d_Li1Cplx2D = (thrust::complex<double>*)malloc(sizeof(thrust::complex<double>)*srcFrequencyNum*srcNum2D);
	for (int i = 0; i < srcFrequencyNum;++i)
	{
		for(int j = 0; j < srcNum2D;++j)
		{
			d_QnCplx2D[j+i*srcNum2D] = thrust::complex<double>(QnCplx2D[i][j].real(), QnCplx2D[i][j].imag());
			d_Li0Cplx2D[j+i*srcNum2D] = thrust::complex<double>(Li0Cplx2D[i][j].real(), Li0Cplx2D[i][j].imag());
			d_Li1Cplx2D[j+i*srcNum2D] = thrust::complex<double>(Li1Cplx2D[i][j].real(), Li1Cplx2D[i][j].imag());
		}
	}
	double3* d_xPos = (double3*)malloc(sizeof(double3)*obsLocation.size());
	double3* d_yPos = (double3*)malloc(sizeof(double3)*srcNum2D);
	for (int i=0; i<obsLocation.size();++i)
	{
		d_xPos[i].x = obsLocation[i][0];
		d_xPos[i].y = obsLocation[i][1];	
		d_xPos[i].z = 0.0;
	}
	for(int j=0;j<srcNum2D;++j)
	{
		d_yPos[j].x = srcLocation2D[0][j][0];
		d_yPos[j].y = srcLocation2D[0][j][1];	
		d_yPos[j].z = 0.0;
	}
	//double* d_srcArea = (double*)malloc(sizeof(double)*srcNum2D);
	//for(int i = 0;i<srcNum2D;++i)
	//{
	//	d_srcArea[i] = srcArea2D[i];
	//}
	double* d_srcFreq = (double*)malloc(sizeof(double)*srcFrequencyNum);
	double* d_pPrimeCplxReal = (double*)malloc(sizeof(double) * srcFrequencyNum * obsLocation.size());
	double* d_pPrimeCplxImag = (double*)malloc(sizeof(double) * srcFrequencyNum * obsLocation.size());

	for(int i = 0;i<srcFrequencyNum;++i)
	{
		d_srcFreq[i] = srcFrequency2D[i];
	}
	for(int i = 0;i<srcFrequencyNum;++i)
	{
		for(int j = 0;j<obsLocation.size();++j)
		{
			d_pPrimeCplxReal[j+i*obsLocation.size()] = 0.0;
			d_pPrimeCplxImag[j+i*obsLocation.size()] = 0.0;
		}
	}
	
	
	// calculate time
	double CalSourceEndTime = MPI_Wtime();

	if (isMaster)
	{
		cout << "*************************************************" << endl;
		cout << "Calculate 2D-Frequency Source term Time cost: " << fixed << setprecision(2);
		cout << CalSourceEndTime - CalSourceStartTime << " s" << endl;
		cout << "*************************************************" << endl;
	}
	
	
	if (isMaster)  fcout << "Calculate 2D pressure spectrum" << endl;
	
	double h2dMsThis = 0.0;
	double d2hMsThis = 0.0;
	double kernelMsThis = 0.0;
	unsigned long long h2dBytesThis = 0ULL;
	unsigned long long d2hBytesThis = 0ULL;

	CalFreqSignalCu2D(
		d_srcFreq, srcFrequencyNum,
		srcArea2D, srcNum2D,
		M0, C0,
		d_QnCplx2D, d_Li0Cplx2D, d_Li1Cplx2D,
		d_xPos, d_yPos, obsLocation.size(),
		d_pPrimeCplxReal, d_pPrimeCplxImag,
		myRank, commSize,
		&h2dMsThis, &d2hMsThis,
		&h2dBytesThis, &d2hBytesThis,
		&kernelMsThis
	);

	pureFWHKernelTime_s += kernelMsThis * 1.0e-3;   // ms -> s

	h2dMsAccum    += h2dMsThis;
	d2hMsAccum    += d2hMsThis;
	h2dBytesAccum += h2dBytesThis;
	d2hBytesAccum += d2hBytesThis;
	
	pPrimeCplx2D.resize(srcFrequencyNum, vector<complex<double>>(obsLocation.size(), complex<double>(0., 0.)));

	double mpi_t0 = MPI_Wtime();
	MPI_Barrier(MPI_COMM_WORLD);
	internalMPITime_s += MPI_Wtime() - mpi_t0;
	
	if (isMaster)   cout << "Finish computing! Copy the pressure from GPU to CPU";
	for (int i = 0;i<srcFrequencyNum;++i)
	{
		for(int j = 0;j<obsLocation.size();++j)
		{
			pPrimeCplx2D[i][j] = complex<double>(d_pPrimeCplxReal[j+i*obsLocation.size()],d_pPrimeCplxImag[j+i*obsLocation.size()]);
		}
	}


	int globalRank = 0;
	MPI_Comm_rank(MPI_COMM_WORLD, &globalRank);

	char statFileName[256];
	std::sprintf(statFileName, "dcu_chunk_stats_rank%d.dat", globalRank);

	std::ofstream statFile(statFileName, std::ios::out);
	statFile << "# globalRank localRank nSrc nObs T BLOCK_T OBS_TILE "
			<< "numObsTiles numTimeChunks totalChunks "
			<< "nominalWorksetBytes maxChunkWorksetBytes sumChunkWorksetBytes "
			<< "H2D_ms D2H_ms H2D_bytes D2H_bytes\n";

	statFile << globalRank << " "
			<< myRank << " "
			<< nSrc << " "
			<< nObs << " "
			<< Nf << " "
			<< BLOCK_T << " "
			<< OBS_TILE << " "
			<< numObsTilesPlan << " "
			<< numTimeChunksPlan << " "
			<< totalChunks << " "
			<< nominalWorksetBytes << " "
			<< maxChunkWorksetBytes << " "
			<< sumChunkWorksetBytes << " "
			<< std::fixed << std::setprecision(6)
			<< h2dMsAccum << " "
			<< d2hMsAccum << " "
			<< h2dBytesAccum << " "
			<< d2hBytesAccum << "\n";

	statFile.close();
	

	
	if (isMaster)   cout << endl;


}



void Farrasat1A::RecoverSignals2D()
{
	size_t srcTNum = srcTime2D.size();
	size_t srcFrequencyNum = (size_t)floor(srcTNum / 2) + 1;

	// allocate memory space for pressure at observers
	if (isMaster)  fcout << "Calculate pressure signal" << endl;

	// 	declare real number arrays for 1d-fft real input / complex output
	double* out_real;
	fftw_complex* ift_cplx;
	// allocate memory
	out_real = (double*)fftw_malloc(sizeof(double) * srcTNum);
	ift_cplx = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * srcFrequencyNum);

	for (int i = 0; i < srcFrequencyNum; i++)
	{
		ift_cplx[i][0] = 0.;
		ift_cplx[i][1] = 0.;
	}
	fftw_plan ift_plan = fftw_plan_dft_c2r_1d((int)srcTNum, ift_cplx, out_real, FFTW_ESTIMATE);
	fftw_execute(ift_plan); // initialize the ift plan

	pPrime2D.resize(srcTNum, vector<double>(obsLocation.size(), 0.0));
	for (int io = 0; io < obsLocation.size(); io++)
	{
		if (isMaster)   cout << "\r";
		if (isMaster)  fcout << "Progress: " << floor(100 * (io + 1) / obsLocation.size()) << " % " << flush;

		for (int ik = 0; ik < srcFrequencyNum; ik++)
		{ 
			ift_cplx[ik][0] = pPrimeCplx2D[ik][io].real();
			ift_cplx[ik][1] = pPrimeCplx2D[ik][io].imag();
		}
		fftw_execute(ift_plan);
		for (int it = 0; it < srcTNum; it++)
		{
			pPrime2D[it][io] = out_real[it] / 2.0 ;
		}
	}
	fftw_destroy_plan(ift_plan);
	if (isMaster)   cout << endl;
}



void Farrasat1A::CalFreqSpectra(int myRank,int commSize) {
	CalSourceTerms();

	size_t srcTNum = srcTime.size();
	size_t srcFrequencyNum = (size_t) floor(srcTNum / 2) + 1;
	
	// declare real number arrays for 1d-fft real input / complex output
	double* in_real;
	fftw_complex* dft_cplx;
	// allocate memory 
	in_real = (double*)fftw_malloc(sizeof(double) * srcTNum);
	dft_cplx = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * srcFrequencyNum);

	for (int i = 0; i < srcTNum; i++)
		in_real[i] = 0.;

	for (int i = 0; i < srcFrequencyNum; i++)
	{
		dft_cplx[i][0] = 0.;
		dft_cplx[i][1] = 0.;
	}
	fftw_plan plan = fftw_plan_dft_r2c_1d((int) srcTNum, in_real, dft_cplx, FFTW_ESTIMATE);
	fftw_execute(plan); // initialize the dft plan

	// allocate memory space for pressure at observers 
	vector<vector<complex<double>>> QnCplx(srcFrequencyNum, vector<complex<double>>(srcNum, complex<double>(0., 0.)));
	vector<vector<complex<double>>> Li0Cplx(srcFrequencyNum, vector<complex<double>>(srcNum, complex<double>(0., 0.)));
	vector<vector<complex<double>>> Li1Cplx(srcFrequencyNum, vector<complex<double>>(srcNum, complex<double>(0., 0.)));
	vector<vector<complex<double>>> Li2Cplx(srcFrequencyNum, vector<complex<double>>(srcNum, complex<double>(0., 0.)));

	double scaleFactor = 2.0 / srcTNum;
	if (isMaster)  fcout << "Calculate Fourier transform " << endl;
	for (int ip = 0; ip < srcNum; ip++) 
	{
		if (isMaster)   cout << "\r";
		if (isMaster)  fcout << "Progress: " << floor(100 * (ip + 1) / srcNum) << " % " << flush;
		
		// DFT for Qn
		for (int it = 0; it < srcTNum; it++)
		{
			in_real[it] = srcQn[it][ip];
		}
		fftw_execute(plan); // run dft
		for (int ik = 0; ik < srcFrequencyNum; ik++)
		{
			QnCplx[ik][ip] = complex<double>(dft_cplx[ik][0], dft_cplx[ik][1]) * scaleFactor;
		}

		// DFT for Li0
		for (int it = 0; it < srcTNum; it++)
		{
			in_real[it] = srcLi[it][ip][0];
		}
		fftw_execute(plan); // run dft
		for (int ik = 0; ik < srcFrequencyNum; ik++)
		{
			Li0Cplx[ik][ip] = complex<double>(dft_cplx[ik][0], dft_cplx[ik][1]) * scaleFactor;
		}

		// DFT for Li1
		for (int it = 0; it < srcTNum; it++)
		{
			in_real[it] = srcLi[it][ip][1];
		}
		fftw_execute(plan); // run dft
		for (int ik = 0; ik < srcFrequencyNum; ik++)
		{
			Li1Cplx[ik][ip] = complex<double>(dft_cplx[ik][0], dft_cplx[ik][1]) * scaleFactor;
		}

		// DFT for Li2
		for (int it = 0; it < srcTNum; it++)
		{
			in_real[it] = srcLi[it][ip][2];
		}
		fftw_execute(plan); // run dft
		for (int ik = 0; ik < srcFrequencyNum; ik++)
		{
			Li2Cplx[ik][ip] = complex<double>(dft_cplx[ik][0], dft_cplx[ik][1]) * scaleFactor;
		}
	}
	fftw_destroy_plan(plan);
	if (isMaster) cout << endl;

	// this implies a uniform pressure sampling at observers
	// it is ok, for the convenience of signal processing, eg. fft.
	const double pi = 2. * acos(0.0);
	//
	const double deltaT = srcTime[1] - srcTime[0];
	const double samplingFrequency = 1.0 / deltaT;
	srcFrequency.resize(srcFrequencyNum, 0.0);
	double deltaF = samplingFrequency / srcTNum;
	for (int i = 0; i < srcFrequencyNum; i++) {
		srcFrequency[i] = double(i) * deltaF;
	}

	vec3 M(-M0, 0, 0);
	double MSqr = M.magSqr();
	CalTimeDelayAtStep(0);
	
	pPrimeCplx.resize(srcFrequencyNum, vector<complex<double>>(obsLocation.size(), complex<double>(0., 0.)));
	const complex<double> I(0., 1.);
	if (isMaster)  fcout << "Calculate pressure spectrum" << endl;
	thrust::complex<double>* d_QnCplx = (thrust::complex<double>*)malloc(sizeof(thrust::complex<double>)*srcFrequencyNum*srcNum);
	thrust::complex<double>* d_Li0Cplx = (thrust::complex<double>*)malloc(sizeof(thrust::complex<double>)*srcFrequencyNum*srcNum);
	thrust::complex<double>* d_Li1Cplx = (thrust::complex<double>*)malloc(sizeof(thrust::complex<double>)*srcFrequencyNum*srcNum);
	thrust::complex<double>* d_Li2Cplx = (thrust::complex<double>*)malloc(sizeof(thrust::complex<double>)*srcFrequencyNum*srcNum);
	for (int i = 0; i < srcFrequencyNum;++i)
	{
		for(int j = 0; j < srcNum;++j)
		{
			d_QnCplx[j+i*srcNum] = thrust::complex<double>(QnCplx[i][j].real(), QnCplx[i][j].imag());
			d_Li0Cplx[j+i*srcNum] = thrust::complex<double>(Li0Cplx[i][j].real(), Li0Cplx[i][j].imag());
			d_Li1Cplx[j+i*srcNum] = thrust::complex<double>(Li1Cplx[i][j].real(), Li1Cplx[i][j].imag());
			d_Li2Cplx[j+i*srcNum] = thrust::complex<double>(Li2Cplx[i][j].real(), Li2Cplx[i][j].imag());
		}
	}
	double* d_R = (double*)malloc(sizeof(double)*srcNum*obsLocation.size());
	double3* d_Rhat = (double3*)malloc(sizeof(double3)*srcNum*obsLocation.size());
	double* d_TimeDelay = (double*)malloc(sizeof(double)*srcNum*obsLocation.size());
	for (int i=0; i<srcNum;++i)
	{
		for(int j=0;j<obsLocation.size();++j)
		{
			d_R[j+i*obsLocation.size()] = R[i][j];
			d_Rhat[j+i*obsLocation.size()].x = Rhat[i][j][0];
			d_Rhat[j+i*obsLocation.size()].y = Rhat[i][j][1];
			d_Rhat[j+i*obsLocation.size()].z = Rhat[i][j][2];
			d_TimeDelay[j+i*obsLocation.size()] = timeDelay[i][j];
		}
	}
	double* d_srcArea = (double*)malloc(sizeof(double)*srcNum);
	for(int i = 0;i<srcNum;++i)
	{
		d_srcArea[i] = srcArea[i];
	}
	double* d_srcFreq = (double*)malloc(sizeof(double)*srcFrequencyNum);
	double* d_pPrimeCplxReal = (double*)malloc(sizeof(double) * srcFrequencyNum * obsLocation.size());
	double* d_pPrimeCplxImag = (double*)malloc(sizeof(double) * srcFrequencyNum * obsLocation.size());

	for(int i = 0;i<srcFrequencyNum;++i)
	{
		d_srcFreq[i] = srcFrequency[i];
	}
	for(int i = 0;i<srcFrequencyNum;++i)
	{
		for(int j = 0;j<obsLocation.size();++j)
		{
			d_pPrimeCplxReal[j+i*obsLocation.size()] = 0.0;
			d_pPrimeCplxImag[j+i*obsLocation.size()] = 0.0;
		}
	}
	double3 d_M = {-M0,0,0};

	CalFreqSignalCu(d_srcFreq,srcFrequencyNum,
	d_srcArea,srcNum,
	d_M, C0,
	d_QnCplx, d_Li0Cplx, d_Li1Cplx, d_Li2Cplx,
    d_R, d_Rhat, d_TimeDelay, obsLocation.size(),
    d_pPrimeCplxReal, d_pPrimeCplxImag,
	myRank,commSize
	);

	MPI_Barrier(MPI_COMM_WORLD);

	if (isMaster)   cout << "Finish computing! Copy the pressure from GPU to CPU";
	for (int i = 0;i<srcFrequencyNum;++i)
	{
		for(int j = 0;j<obsLocation.size();++j)
		{
			pPrimeCplx[i][j] = complex<double>(d_pPrimeCplxReal[j+i*obsLocation.size()],d_pPrimeCplxImag[j+i*obsLocation.size()]);
		}
	}

	if (isMaster)   cout << endl;
}

void Farrasat1A::RecoverSignals()
{
	size_t srcTNum = srcTime.size();
	size_t srcFrequencyNum = (size_t) floor(srcTNum / 2) + 1;

	// allocate memory space for pressure at observers
	if (isMaster)  fcout << "Calculate pressure signal" << endl;

	// 	declare real number arrays for 1d-fft real input / complex output
	double* out_real;
	fftw_complex* ift_cplx;
	// allocate memory
	out_real = (double*)fftw_malloc(sizeof(double) * srcTNum);
	ift_cplx = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * srcFrequencyNum);

	for (int i = 0; i < srcFrequencyNum; i++)
	{
		ift_cplx[i][0] = 0.;
		ift_cplx[i][1] = 0.;
	}
	fftw_plan ift_plan = fftw_plan_dft_c2r_1d((int) srcTNum, ift_cplx, out_real, FFTW_ESTIMATE);
	fftw_execute(ift_plan); // initialize the ift plan

	pPrime.resize(obsTimeNum, vector<double>(obsLocation.size(), 0.0));
	for (int io = 0; io < obsLocation.size(); io++)
	{
		if (isMaster)   cout << "\r";
		if (isMaster)  fcout << "Progress: " << floor(100 * (io + 1) / obsLocation.size()) << " % " << flush;

		for (int ik = 0; ik < srcFrequencyNum; ik++)
		{
			ift_cplx[ik][0] = pPrimeCplx[ik][io].real();
			ift_cplx[ik][1] = pPrimeCplx[ik][io].imag();
		}
		fftw_execute(ift_plan);
		for (int it = 0; it < obsTime.size(); it++)
		{
			pPrime[it][io] = out_real[it] / 2.0;
		}
	}
	fftw_destroy_plan(ift_plan);
	if (isMaster)   cout << endl;
}


void Farrasat1A::SaveSpectrumMag(string pFile) {
	
	if (isMaster)   fcout << pFile << endl;

	ofstream spectrumFile_mag(pFile.c_str());
	for (int ik = 0; ik < srcFrequency.size(); ik++) {
		double freq = srcFrequency[ik];
		spectrumFile_mag << freq << " ";
		for (int io = 0; io < obsLocation.size(); io++)
		{
			spectrumFile_mag << abs(pPrimeCplx[ik][io]) << " ";
		}
		spectrumFile_mag << endl;
	}
}

void Farrasat1A::SaveSpectrumMag2D(string pFile) {

	if (isMaster)   fcout << pFile << endl;

	ofstream spectrumFile_mag(pFile.c_str());
	for (int ik = 0; ik < srcFrequency2D.size(); ik++) {
		double freq = srcFrequency2D[ik];
		spectrumFile_mag << freq << " ";
		for (int io = 0; io < obsLocation.size(); io++)
		{
			spectrumFile_mag << abs(pPrimeCplx2D[ik][io]) << " ";
		}
		spectrumFile_mag << endl;
	}
	spectrumFile_mag.close();
}



void Farrasat1A::SaveSpectrumCplx(string pRealFile, string pImagFile) {

	if (isMaster)   fcout << pRealFile << endl;

	ofstream spectrumRealFile(pRealFile.c_str());
	for (int ik = 0; ik < srcFrequency.size(); ik++) {
		double freq = srcFrequency[ik];
		spectrumRealFile << freq << " ";
		for (int io = 0; io < obsLocation.size(); io++)
		{
			spectrumRealFile << real(pPrimeCplx[ik][io]) << " ";
		}
		spectrumRealFile << endl;
	}

	if (isMaster)   fcout << pImagFile << endl;
	ofstream spectrumImagFile(pImagFile.c_str());
	for (int ik = 0; ik < srcFrequency.size(); ik++) {
		double freq = srcFrequency[ik];
		spectrumImagFile << freq << " ";
		for (int io = 0; io < obsLocation.size(); io++)
		{
			spectrumImagFile << imag(pPrimeCplx[ik][io]) << " ";
		}
		spectrumImagFile << endl;
	}
}


void Farrasat1A::SaveSpectrumCplx2D(string pRealFile, string pImagFile) {

	if (isMaster)   fcout << pRealFile << endl;

	ofstream spectrumRealFile(pRealFile.c_str());
	for (int ik = 0; ik < srcFrequency2D.size(); ik++) {
		double freq = srcFrequency2D[ik];
		spectrumRealFile << freq << " ";
		for (int io = 0; io < obsLocation.size(); io++)
		{
			spectrumRealFile << real(pPrimeCplx2D[ik][io]) << " ";
		}
		spectrumRealFile << endl;
	}

	if (isMaster)   fcout << pImagFile << endl;
	ofstream spectrumImagFile(pImagFile.c_str());
	for (int ik = 0; ik < srcFrequency2D.size(); ik++) {
		double freq = srcFrequency2D[ik];
		spectrumImagFile << freq << " ";
		for (int io = 0; io < obsLocation.size(); io++)
		{
			spectrumImagFile << imag(pPrimeCplx2D[ik][io]) << " ";
		}
		spectrumImagFile << endl;
	}
}




void Farrasat1A::SaveTimeSignals(string pFile) {
	if (isMaster)   fcout << pFile << endl;
	// TDODO: deine output path and file name
	ofstream outfile(pFile.c_str(), ios::out);
	for (size_t i = 0; i < obsTime.size(); i++) 
	{
		outfile << obsTime[i] << " ";
		for (size_t j = 0; j < obsLocation.size(); j++) {
			outfile << pPrime[i][j] << " ";
		}
		outfile << "\n";
	}
	outfile.close();
}



void Farrasat1A::SaveTimeSignals2D(string pFile) {
	if (isMaster)   fcout << pFile << endl;
	// TDODO: deine output path and file name
	ofstream outfile2D(pFile.c_str(), ios::out);
	for (size_t i = 0; i < srcTime2D.size(); i++)
	{
		outfile2D << srcTime2D[i]+20.0 << " ";
		for (size_t j = 0; j < obsLocation.size(); j++) {
			outfile2D << pPrime2D[i][j] << " ";
		}
		outfile2D << "\n";
	}
	outfile2D.close();
}
