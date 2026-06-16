/*****************************************************************/
//	PROGRAM EIT-FWH version 2.0
// 
//	Developed by Hanbo JIANG (hanbojiang@eias.ac.cn)
//	Eastern Institute for Advanced Study
//	 
//	Copyright 2022 by Professor H. Jiang. All Rights Reserved 
// 
//	This program, including documentation, source code, executable code and related 
//	items, is instigated by Professor H. Jiang as principal investigator of various
//	research grants. Distribution to third parties is strictly prohibited without 
//  prior written permission from Professor H. Jiang. No part of this softwareand 
//	documentation may be used modified or reproduced stored in a retrieval system 
//	or transmitted in any form or by any means electronic mechanical optical 
//  photocopying recording or otherwise without the prior written permission of
//  Professor H. Jiang.
//  
// TODO:
//	Consider use binary file for i/o operations for accelaration
//
/*****************************************************************/

#include <iostream>
#include <fstream>
#include <string>
#include <iomanip> 
#include "mpi.h"
#include "Config.h"
#include "Farrasat1A.h"
#include <cstdio>

#define mcout cout<< "EIT-FWH" << ":: "

#define MPI_CALL_TIMED(acc, call) \
do { \
    double __mpi_t0 = MPI_Wtime(); \
    call; \
    acc += MPI_Wtime() - __mpi_t0; \
} while(0)


using namespace std;


void autoDivideIndex(int vecDim, int& locIdStart, int& locNumId, int myRank, int commSize)
{
	locIdStart = vecDim * myRank / commSize;
	int locIdEnd = vecDim * (myRank + 1) / commSize - 1;
	locNumId = locIdEnd - locIdStart + 1;
}

int main(int argc, char* argv[]) {
	//

	MPI_Init(&argc, &argv);

	int myRank, commSize;
	MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
	MPI_Comm_size(MPI_COMM_WORLD, &commSize);
	bool isMaster = !myRank;

	// 创建本地通信器
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, myRank, MPI_INFO_NULL, &localComm);

    int localRank, localSize;
    MPI_Comm_rank(localComm, &localRank);
    MPI_Comm_size(localComm, &localSize);

	double mpiPre_s      = 0.0;
	double mpiCompMain_s = 0.0;
	double mpiPost_s     = 0.0;

	double ioRead_s  = 0.0;
	double ioWrite_s = 0.0;

	
	//time calculate
	double Pre_Treat_start = MPI_Wtime();

	
	//select 2D/3D or Time/Frequency
	string dimension, type;
	if (isMaster)
	{
		string configFile = "config.inp";
		Config CF(configFile);
		dimension = CF.Read("dimension", dimension);
		type = CF.Read("type", type);
		cout << "*************************************************" << endl;
		mcout << "Computing dimension " << dimension << endl;
		mcout << "Calculation types " << type << endl;
	}

	/*
	//select 2D/3D or Time/Frequency
	string dimension, type;
	if (isMaster) {
		
		cout << "Choose computation dimensions (2D/3D): ";
		cin >> dimension;

		cout << "Choose computation type (Time/Frequency): ";
		cin >> type;
	}
	*/

	// Convert sizes to int for MPI_Bcast
	int dimensionLength = static_cast<int>(dimension.size());
	int typeLength = static_cast<int>(type.size());

	// Broadcast sizes
	MPI_CALL_TIMED(mpiPre_s, MPI_Bcast(&dimensionLength, 1, MPI_INT, 0, MPI_COMM_WORLD));
	MPI_CALL_TIMED(mpiPre_s, MPI_Bcast(&typeLength, 1, MPI_INT, 0, MPI_COMM_WORLD));

	// Ensure the strings are of the correct size
	if (!isMaster) {
		dimension.resize(dimensionLength);
		type.resize(typeLength);
	}

	// Broadcast strings
	MPI_CALL_TIMED(mpiPre_s, MPI_Bcast(&dimension[0], dimensionLength, MPI_CHAR, 0, MPI_COMM_WORLD));
	MPI_CALL_TIMED(mpiPre_s, MPI_Bcast(&type[0], typeLength, MPI_CHAR, 0, MPI_COMM_WORLD));


	if (myRank == 0)
	{
		cout << "*************************************************" << endl;
		cout << __TIMESTAMP__ << "Run at " << endl;
		cout << "*************************************************" << endl;
	}

	


	/***************** 1. read freestream condition ******************/
	//define freestream condition
	double MxInf = 0.0;
	double rhoInf = 0.0;
	double cInf = 0.0;
	double pInf = 0.0;
	
	if (myRank == 0)
	{
		mcout << "Read freestream conditions" << endl;
		string configFile = "config.inp";
		Config CF(configFile);
		MxInf = CF.Read("FreeStreamMachNumber", MxInf);
		rhoInf = CF.Read("FreestreamDensity", rhoInf);
		cInf = CF.Read("FreestreamSoundSpeed", cInf);
		pInf = CF.Read("FreestreamPressure", pInf);

		mcout << "Mach number x+ " << MxInf << endl;
		mcout << "Speed of sound " << cInf << endl;
		mcout << "Density " << rhoInf << endl;
		mcout << "Pressure " << pInf << endl;

	}
	MPI_CALL_TIMED(mpiPre_s, MPI_Bcast(&MxInf, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD));
	MPI_CALL_TIMED(mpiPre_s, MPI_Bcast(&rhoInf, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD));
	MPI_CALL_TIMED(mpiPre_s, MPI_Bcast(&cInf, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD));
	MPI_CALL_TIMED(mpiPre_s, MPI_Bcast(&pInf, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD));
	
	/*****************************************************************/
	Farrasat1A F1A(MxInf, cInf, rhoInf, pInf, true, myRank);
	/*****************************************************************/

	/***************** 2. define observers ***************************/
	vector<vec3> obsLocation;
	int observerNum = 0;
	if (myRank == 0)
	{
		double io_t0 = MPI_Wtime();

		string configFile = "config.inp";
		Config CF(configFile);
		// read observers
		string observerFileName;
		observerFileName = CF.Read("observerFileName", observerFileName);
		mcout << "Read observer coordinates from file " << observerFileName << endl;
		ifstream observerFile(observerFileName.c_str(), ios::in);

		observerFile >> observerNum;
		obsLocation.resize(observerNum, vec3(0., 0., 0.));
		for (int io = 0; io < observerNum; io++)
		{
			cout << "\r";
			mcout << "Progress: " << floor(100 * (io + 1) / observerNum) << " % " << flush;
			observerFile >> obsLocation[io][0] >> obsLocation[io][1] >> obsLocation[io][2];
		}
		cout << endl;
		observerFile.close();

		ioRead_s += MPI_Wtime() - io_t0;
	}
	
	/*****************************************************************/
	// COPY all observers to each processor							 */
	/*****************************************************************/

	/*****************************************************************/
	// Broadcast observerNum to let all processors know the same    
	// number of observers
	MPI_CALL_TIMED(mpiPre_s, MPI_Bcast(&observerNum, 1, MPI_INT, 0, MPI_COMM_WORLD));


	/*****************************************************************/
	if (myRank == 0)
	{
		mcout << "Deliver observer data to processors " << endl;
		int dataLength = observerNum * 3;
		vector<double> buffer(dataLength);
		for (int io = 0; io < observerNum; io++)
			for (int ik = 0; ik < 3; ik++)
				buffer[io * 3 + ik] = obsLocation[io][ik];
		for (int iRank = 1; iRank < commSize; iRank++)
		{
			cout << "\r";
			mcout << "Progress: " << floor(100 * (iRank + 1) / commSize) << " % " << flush;
			MPI_CALL_TIMED(mpiPre_s,
    			MPI_Send(&buffer[0], dataLength, MPI_DOUBLE, iRank, 1, MPI_COMM_WORLD));
		}
		cout << endl;
		
	}
	else
	{
		obsLocation.resize(observerNum, vec3(0., 0., 0.));
		int dataLength = observerNum * 3;
		vector<double> buffer(dataLength);
		MPI_CALL_TIMED(mpiPre_s,
    		MPI_Recv(&buffer[0], dataLength, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUSES_IGNORE));
		for (int io = 0; io < observerNum; io++)
			for (int ik = 0; ik < 3; ik++)
				obsLocation[io][ik] = buffer[io * 3 + ik];
	}
	/*****************************************************************/


	/***************** 3. define flow data ***************************/
	// a) Read data-file using the master processor
	// b) Quasi-equally divide these data w.r.t the number of cells
	// c) Send partitions to each processor
	// 
	// Note that some data arrays are useless after this step, you can 
	// release their memory if needed.
	/*****************************************************************/
	int flowCellNum = 0; // the number of cells on the integral surface
	int flowStepNum = 0; // the number of time steps
	vector<double> flowTime;
	
	/*
	vector<vector<double>> flowDensity;
	vector<vector<double>> flowPressure;
	vector<vector<vec3>>   flowVelocity;
	*/
	
	//MPI realize 
	vector<vector<double>> localFlowDensity;
	vector<vector<double>> localFlowPressure;
	vector<vector<vec3>>   localFlowVelocity;
	
	if (dimension == "3D")
	{
		string configFile = "config.inp";
		Config CF(configFile);
		string flowDataFileName;
		flowDataFileName = CF.Read("flowDataFileName", flowDataFileName);
		ifstream flowDataFile(flowDataFileName.c_str());
		
		double io_t0 = MPI_Wtime();
		
		flowDataFile >> flowStepNum >> flowCellNum;

		flowTime.resize(flowStepNum, 0.);

		if (isMaster) mcout << "Read flow data from file " << flowDataFileName << endl;
		
		// Read the flow data
		int locNumId = 0;
		int locIdStart = 0;
		autoDivideIndex(flowCellNum, locIdStart, locNumId, myRank, commSize);
		//mcout << "From processor " << myRank << " " << " LocIdStart is " << locIdStart << " Lod Id Num is " << locNumId << endl;
		
		double rhoTmp;
		double pTmp;
		double uTmp;
		double vTmp;
		double wTmp;
		localFlowDensity.resize(flowStepNum, vector<double>(locNumId, 0.));
		localFlowPressure.resize(flowStepNum, vector<double>(locNumId, 0.));
		localFlowVelocity.resize(flowStepNum, vector<vec3>(locNumId, vec3(0., 0., 0.)));
		for (int it = 0; it < flowStepNum; it++)
		{
			if (isMaster)
			{
				cout << "\r";
				mcout << "Progress: " << floor(100 * (it + 1) / flowStepNum) << " % " << flush;
			}
			
			flowDataFile >> flowTime[it];
			for (int ip = 0; ip< flowCellNum; ip++)
			{
				flowDataFile >> rhoTmp;
				flowDataFile >> pTmp;
				flowDataFile >> uTmp;
				flowDataFile >> vTmp;
				flowDataFile >> wTmp;
				if(ip >= locIdStart && ip < locIdStart + locNumId)
				{
					localFlowDensity[it][ip-locIdStart] = rhoTmp;
					localFlowPressure[it][ip-locIdStart] = pTmp;
					localFlowVelocity[it][ip-locIdStart][0] = uTmp;
					localFlowVelocity[it][ip-locIdStart][1] = vTmp;
					localFlowVelocity[it][ip-locIdStart][2] = wTmp;
				}
			}
		}

		ioRead_s += MPI_Wtime() - io_t0;

		double srcTimeMin = *min_element(flowTime.begin(), flowTime.end());
		double srcTimeMax = *max_element(flowTime.begin(), flowTime.end());
		

		if (isMaster) {
    		cout << '\n';
    		mcout << "Source-time step 3D " 
          		<< flowTime.size() << '\n';
    		mcout << "Source-time from 3D "
          		<< srcTimeMin << " to " << srcTimeMax << '\n';
		}
		
	}

	/***************** 3.1 define 2D flow data ***************************/
	// a) Read data-file using the master processor
	/*****************************************************************/
	int flowCellNum2D = 0; // the number of cells on the integral surface
	int flowStepNum2D = 0; // the number of time steps
	vector<double> flowTime2D;
	
	/*
	vector<vector<double>> flowDensity2D;
	vector<vector<double>> flowPressure2D;
	vector<vector<vec3>>   flowVelocity2D;
	*/

	//MPI realize
	vector<vector<double>> localFlowDensity2D;
	vector<vector<double>> localFlowPressure2D;
	vector<vector<vec3>>   localFlowVelocity2D;

	if (dimension == "2D")
	{
		string configFile = "config.inp";
		Config CF(configFile);
		string flowDataFileName2D;
		flowDataFileName2D = CF.Read("flowDataFileName2D", flowDataFileName2D);
		
		double io_t0 = MPI_Wtime();

		ifstream flowDataFile2D(flowDataFileName2D.c_str());
		flowDataFile2D >> flowStepNum2D >> flowCellNum2D;
			
		// allocate memory space before reading data
		flowTime2D.resize(flowStepNum2D, 0.);
		
		if (isMaster) mcout << "Read 2D flow data from file " << flowDataFileName2D << endl;
		
		// Read the flow data
		int locNumId = 0;
		int locIdStart = 0;
		autoDivideIndex(flowCellNum2D, locIdStart, locNumId, myRank, commSize);

		double rhoTmp;
		double pTmp;
		double uTmp;
		double vTmp;
		double wTmp;
		localFlowDensity2D.resize(flowStepNum2D, vector<double>(locNumId, 0.));
		localFlowPressure2D.resize(flowStepNum2D, vector<double>(locNumId, 0.));
		localFlowVelocity2D.resize(flowStepNum2D, vector<vec3>(locNumId, vec3(0., 0., 0.)));
		for (int it = 0; it < flowStepNum2D; it++)
		{	
			if (isMaster)
			{
				cout << "\r";
				mcout << "Progress: " << floor(100 * (it + 1) / flowStepNum2D) << " % " << flush;
			}

			flowDataFile2D >> flowTime2D[it];
			for (int ip = 0; ip< flowCellNum2D; ip++)
			{
				flowDataFile2D >> rhoTmp;
				flowDataFile2D >> pTmp;
				flowDataFile2D >> uTmp;
				flowDataFile2D >> vTmp;
				flowDataFile2D >> wTmp;
				if(ip >= locIdStart && ip < locIdStart + locNumId)
				{
					localFlowDensity2D[it][ip-locIdStart] = rhoTmp;
					localFlowPressure2D[it][ip-locIdStart] = pTmp;
					localFlowVelocity2D[it][ip-locIdStart][0] = uTmp;
					localFlowVelocity2D[it][ip-locIdStart][1] = vTmp;
					localFlowVelocity2D[it][ip-locIdStart][2] = wTmp;
				}
			}
		}

		ioRead_s += MPI_Wtime() - io_t0;

		/*****************************************************************/
		double srcTimeMin2D = *min_element(flowTime2D.begin(), flowTime2D.end());
		double srcTimeMax2D = *max_element(flowTime2D.begin(), flowTime2D.end());

		if (isMaster) {
    		cout << '\n';
    		mcout << "Source-time step 2D " 
          		<< flowTime2D.size() << '\n';
    		mcout << "Source-time from 2D "
          		<< srcTimeMin2D << " to " << srcTimeMax2D << '\n';
		}

	}
	/********** 4. Read the integral surface geometry data **********/
	// Similar to step 3
	// a) Read surface data-file using the master processor
	// b) Quasi-equally divide these data w.r.t the number of cells
	// c) Send partitions to each processor
	// 
	// Note that some data arrays are useless after this step, you can 
	// release their memory if needed.
	/*****************************************************************/
	vector<double> cellArea;
	// Note:: cellCentre and cellNormal are pre-assumed as time-dependent variables
	// even if the surface is static, the dimension is flowStepNum instead of nSteps!
	vector<vector<vec3>>   cellCentre;
	vector<vector<vec3>>   cellNormal;
	bool isMovingSurface = false;
	size_t nSteps, nPanels;

	//------------------------------The integral surface is stationary-------------------------------
	int flowStepNum_static = 1;


	if (dimension == "3D")
	{
		if (myRank == 0)
		{

			string configFile = "config.inp";
			Config CF(configFile);
			string intSurfaceFileName;
			intSurfaceFileName = CF.Read("intSurfaceFileName", intSurfaceFileName);
			mcout << "Read integral surface file " << intSurfaceFileName << endl;
			ifstream fwhSurfacefile(intSurfaceFileName.c_str(), ios::in);
			
			double io_t0 = MPI_Wtime();
			
			fwhSurfacefile >> nSteps >> nPanels;
			if (nPanels != flowCellNum)
			{
				mcout << "Found inconsistent surface-cell numbers in surface file and flow data file!!!" << endl;
				MPI_Finalize();
				return -1;
			}
			if (nSteps != flowStepNum)
				mcout << "The integral surface is stationary. " << endl;
			else
				mcout << "The integral surface is moving. " << endl;

			
			cellArea.resize(nPanels, 0.);
			// Note:: cellCentre and cellNormal are pre-assumed as time-dependent variables
			// even if the surface is static, the dimension is flowStepNum instead of nSteps!
			cellCentre.resize(flowStepNum_static, vector<vec3>(nPanels, vec3(0., 0., 0.)));
			cellNormal.resize(flowStepNum_static, vector<vec3>(nPanels, vec3(0., 0., 0.)));

			for (size_t ip = 0; ip < nPanels; ip++) {
				fwhSurfacefile >> cellArea[ip];
			}
			// if found insuffient time-step for fwhSurface,
			// we simply assume it is static
			if (nSteps != flowStepNum_static) {
				isMovingSurface = false;
				for (size_t ip = 0; ip < nPanels; ip++)
				{
					fwhSurfacefile >> cellCentre[0][ip][0] >> cellCentre[0][ip][1] >> cellCentre[0][ip][2];
					fwhSurfacefile >> cellNormal[0][ip][0] >> cellNormal[0][ip][1] >> cellNormal[0][ip][2];
				}

				for (size_t it = 1; it < flowStepNum_static; it++)
				{
					for (size_t ip = 0; ip < nPanels; ip++)
					{
						cellCentre[it][ip] = cellCentre[0][ip];
						cellNormal[it][ip] = cellNormal[0][ip];
					}
				}
			}
			else
			{
				// the surface is time-dependent, meaning that it is moving
				// for example, the propeller case
				isMovingSurface = true;
				for (size_t it = 0; it < flowStepNum_static; it++)
				{
					cout << "\r";
					mcout << "Progress: " << floor(100 * (it + 1) / flowStepNum_static) << " % " << flush;
					for (size_t ip = 0; ip < nPanels; ip++)
					{
						fwhSurfacefile >> cellCentre[0][ip][0] >> cellCentre[0][ip][1] >> cellCentre[0][ip][2];
						fwhSurfacefile >> cellNormal[0][ip][0] >> cellNormal[0][ip][1] >> cellNormal[0][ip][2];
					}
				}
				cout << endl;
			}
			fwhSurfacefile.close();

			ioRead_s += MPI_Wtime() - io_t0;

		}

	}
	/*****************************************************************/
	// Split and cast integral surface to processors 3D
	/*****************************************************************/
	vector<double>			localCellArea;
	vector<vector<vec3>>   localCellCentre;
	vector<vector<vec3>>   localCellNormal;

	if (dimension == "3D")
	{

		if (myRank == 0)
		{
			mcout << "Deliver integral surface to processors " << endl;
			for (int iRank = 1; iRank < commSize; iRank++)
			{
				cout << "\r";
				mcout << "Progress: " << floor(100 * (iRank + 1) / commSize) << " % " << flush;
				int locNumId = 0;
				int locIdStart = 0;
				autoDivideIndex(flowCellNum, locIdStart, locNumId, iRank, commSize);
				MPI_CALL_TIMED(mpiPre_s, MPI_Send(&cellArea[locIdStart], locNumId, MPI_DOUBLE, iRank, 10, MPI_COMM_WORLD));

				vector<double> sendBuffer2(flowStepNum_static * locNumId * 3, 0.);
				for (int it = 0; it < flowStepNum_static; it++)
					for (int ip = 0; ip < locNumId; ip++)
						for (int ik = 0; ik < 3; ik++)
							sendBuffer2[(it * locNumId + ip) * 3 + ik] = cellCentre[it][ip + locIdStart][ik];
				MPI_CALL_TIMED(mpiPre_s, MPI_Send(&sendBuffer2[0], flowStepNum_static * locNumId * 3, MPI_DOUBLE, iRank, 11, MPI_COMM_WORLD));

				vector<double> sendBuffer3(flowStepNum_static * locNumId * 3, 0.);
				for (int it = 0; it < flowStepNum_static; it++)
					for (int ip = 0; ip < locNumId; ip++)
						for (int ik = 0; ik < 3; ik++)
							sendBuffer3[(it * locNumId + ip) * 3 + ik] = cellNormal[it][ip + locIdStart][ik];
				MPI_CALL_TIMED(mpiPre_s, MPI_Send(&sendBuffer3[0], flowStepNum_static * locNumId * 3, MPI_DOUBLE, iRank, 12, MPI_COMM_WORLD));
			}
			if (commSize > 1) cout << endl;

			int iRank = 0;
			int locNumId = 0;
			int locIdStart = 0;
			autoDivideIndex(flowCellNum, locIdStart, locNumId, iRank, commSize);
			localCellArea.resize(locNumId, 0.);
			localCellCentre.resize(flowStepNum_static, vector<vec3>(locNumId, vec3(0., 0., 0.)));
			localCellNormal.resize(flowStepNum_static, vector<vec3>(locNumId, vec3(0., 0., 0.)));
			for (int ip = 0; ip < locNumId; ip++)
			{
				localCellArea[ip] = cellArea[ip + locIdStart];
			}
			for (int it = 0; it < flowStepNum_static; it++)
				for (int ip = 0; ip < locNumId; ip++)
				{
					localCellCentre[it][ip] = cellCentre[it][ip + locIdStart];
					localCellNormal[it][ip] = cellNormal[it][ip + locIdStart];
				}
		}
		else
		{
			int locNumId = 0;
			int locIdStart = 0;
			autoDivideIndex(flowCellNum, locIdStart, locNumId, myRank, commSize);
			localCellArea.resize(locNumId, 0.);
			MPI_CALL_TIMED(mpiPre_s, MPI_Recv(&localCellArea[0], locNumId, MPI_DOUBLE, 0, 10, MPI_COMM_WORLD, MPI_STATUSES_IGNORE));
			//*************************/


			localCellCentre.resize(flowStepNum_static, vector<vec3>(locNumId, vec3(0., 0., 0.)));
			vector<double> recvBuffer2(flowStepNum_static * locNumId * 3, 0.);
			MPI_CALL_TIMED(mpiPre_s, MPI_Recv(&recvBuffer2[0], flowStepNum_static * locNumId * 3, MPI_DOUBLE, 0, 11, MPI_COMM_WORLD, MPI_STATUSES_IGNORE));
			for (int it = 0; it < flowStepNum_static; it++)
				for (int ip = 0; ip < locNumId; ip++)
					for (int ik = 0; ik < 3; ik++)
						localCellCentre[it][ip][ik] = recvBuffer2[(it * locNumId + ip) * 3 + ik];

			localCellNormal.resize(flowStepNum_static, vector<vec3>(locNumId, vec3(0., 0., 0.)));
			vector<double> recvBuffer3(flowStepNum_static * locNumId * 3, 0.);
			MPI_CALL_TIMED(mpiPre_s, MPI_Recv(&recvBuffer3[0], flowStepNum_static * locNumId * 3, MPI_DOUBLE, 0, 12, MPI_COMM_WORLD, MPI_STATUSES_IGNORE));
			for (int it = 0; it < flowStepNum_static; it++)
				for (int ip = 0; ip < locNumId; ip++)
					for (int ik = 0; ik < 3; ik++)
						localCellNormal[it][ip][ik] = recvBuffer3[(it * locNumId + ip) * 3 + ik];
		}

	}

	/********** 4.1 Read the 2D integral surface geometry data **********/
	//
	// Note:: cellCentre and cellNormal are pre-assumed as time-dependent variables
	// even if the surface is static, the dimension is flowStepNum instead of nSteps!
	/*****************************************************************/
	double cellArea2D = 0.;
	vector<vector<vec3>>  cellCentre2D;
	vector<vector<vec3>>  cellNormal2D;
	bool isMovingSurface2D = false;
	size_t nSteps2D, nPanels2D;

	if (dimension == "2D")
	{

		if (myRank == 0)
		{
			string configFile = "config.inp";
			Config CF(configFile);
			string SurfaceFileName2D;
			SurfaceFileName2D = CF.Read("SurfaceFileName2D", SurfaceFileName2D);
			mcout << "Read 2D integral surface file " << SurfaceFileName2D << endl;
			
			double io_t0 = MPI_Wtime();

			ifstream fwhSurfacefile2D(SurfaceFileName2D.c_str(), ios::in);
			fwhSurfacefile2D >> nSteps2D >> nPanels2D;
			fwhSurfacefile2D >> cellArea2D;

			if (nPanels2D != flowCellNum2D)
			{
				mcout << "Found inconsistent surface-cell numbers in surface file and flow data file!!!" << endl;
				MPI_Finalize();
				return -1;
			}

			if (nSteps2D != flowStepNum2D)//
				mcout << "The integral surface is stationary. " << endl;
			else
				mcout << "The integral surface is moving. " << endl;

			// Note:: cellCentre and cellNormal are pre-assumed as time-dependent variables
			// even if the surface is static, the dimension is flowStepNum instead of nSteps!
			cellCentre2D.resize(flowStepNum2D, vector<vec3>(nPanels2D, vec3(0., 0., 0.)));
			cellNormal2D.resize(flowStepNum2D, vector<vec3>(nPanels2D, vec3(0., 0., 0.)));


			// if found insuffient time-step for fwhSurface,
			// we simply assume it is static
			if (nSteps2D != flowStepNum2D) {
				isMovingSurface = false;
				for (size_t ip = 0; ip < nPanels2D; ip++)
				{
					fwhSurfacefile2D >> cellCentre2D[0][ip][0] >> cellCentre2D[0][ip][1] >> cellCentre2D[0][ip][2];
					fwhSurfacefile2D >> cellNormal2D[0][ip][0] >> cellNormal2D[0][ip][1] >> cellNormal2D[0][ip][2];
				}

				for (size_t it = 1; it < flowStepNum2D; it++)
				{
					for (size_t ip = 0; ip < nPanels2D; ip++)
					{
						cellCentre2D[it][ip] = cellCentre2D[0][ip];
						cellNormal2D[it][ip] = cellNormal2D[0][ip];
					}
				}
			}
			else
			{
				// the surface is time-dependent, meaning that it is moving
				// for example, the propeller case
				isMovingSurface = true;
				for (size_t it = 0; it < flowStepNum2D; it++)
				{
					cout << "\r";
					mcout << "Progress: " << floor(100 * (it + 1) / flowStepNum2D) << " % " << flush;
					for (size_t ip = 0; ip < nPanels2D; ip++)
					{
						fwhSurfacefile2D >> cellCentre2D[it][ip][0] >> cellCentre2D[it][ip][1] >> cellCentre2D[it][ip][2];
						fwhSurfacefile2D >> cellNormal2D[it][ip][0] >> cellNormal2D[it][ip][1] >> cellNormal2D[it][ip][2];
					}
				}
				mcout << endl;
			}
			fwhSurfacefile2D.close();

			ioRead_s += MPI_Wtime() - io_t0;
		}

		MPI_CALL_TIMED(mpiPre_s, MPI_Bcast(&cellArea2D, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD));


	}
	//MPI realize
	/*****************************************************************/
	// Split and cast integral surface to processors
	/*****************************************************************/
	vector<vector<vec3>>   localCellCentre2D;
	vector<vector<vec3>>   localCellNormal2D;

	if (dimension == "2D")
	{
		if (myRank == 0)
		{
			mcout << "Deliver 2D integral surface to processors " << endl;
			for (int iRank = 1; iRank < commSize; iRank++)
			{
				cout << "\r";
				mcout << "Progress: " << floor(100 * (iRank + 1) / commSize) << " % " << flush;
				int locNumId = 0;
				int locIdStart = 0;
				autoDivideIndex(flowCellNum2D, locIdStart, locNumId, iRank, commSize);

				vector<double> sendBuffer2_2D(flowStepNum2D * locNumId * 3, 0.);
				for (int it = 0; it < flowStepNum2D; it++)
					for (int ip = 0; ip < locNumId; ip++)
						for (int ik = 0; ik < 3; ik++)
							sendBuffer2_2D[(it * locNumId + ip) * 3 + ik] = cellCentre2D[it][ip + locIdStart][ik];
				MPI_CALL_TIMED(mpiPre_s, MPI_Send(&sendBuffer2_2D[0], flowStepNum2D * locNumId * 3, MPI_DOUBLE, iRank, 14, MPI_COMM_WORLD));

				vector<double> sendBuffer3_2D(flowStepNum2D * locNumId * 3, 0.);
				for (int it = 0; it < flowStepNum2D; it++)
					for (int ip = 0; ip < locNumId; ip++)
						for (int ik = 0; ik < 3; ik++)
							sendBuffer3_2D[(it * locNumId + ip) * 3 + ik] = cellNormal2D[it][ip + locIdStart][ik];
				MPI_CALL_TIMED(mpiPre_s, MPI_Send(&sendBuffer3_2D[0], flowStepNum2D * locNumId * 3, MPI_DOUBLE, iRank, 15, MPI_COMM_WORLD));
			}
			if (commSize > 1) cout << endl;

			int iRank = 0;
			int locNumId = 0;
			int locIdStart = 0;
			autoDivideIndex(flowCellNum2D, locIdStart, locNumId, iRank, commSize);
			localCellCentre2D.resize(flowStepNum2D, vector<vec3>(locNumId, vec3(0., 0., 0.)));
			localCellNormal2D.resize(flowStepNum2D, vector<vec3>(locNumId, vec3(0., 0., 0.)));

			for (int it = 0; it < flowStepNum2D; it++)
				for (int ip = 0; ip < locNumId; ip++)
				{
					localCellCentre2D[it][ip] = cellCentre2D[it][ip + locIdStart];
					localCellNormal2D[it][ip] = cellNormal2D[it][ip + locIdStart];
				}
		}
		else
		{
			int locNumId = 0;
			int locIdStart = 0;
			autoDivideIndex(flowCellNum2D, locIdStart, locNumId, myRank, commSize);

			localCellCentre2D.resize(flowStepNum2D, vector<vec3>(locNumId, vec3(0., 0., 0.)));
			vector<double> recvBuffer2_2D(flowStepNum2D * locNumId * 3, 0.);
			MPI_CALL_TIMED(mpiPre_s, MPI_Recv(&recvBuffer2_2D[0], flowStepNum2D * locNumId * 3, MPI_DOUBLE, 0, 14, MPI_COMM_WORLD, MPI_STATUSES_IGNORE));
			for (int it = 0; it < flowStepNum2D; it++)
				for (int ip = 0; ip < locNumId; ip++)
					for (int ik = 0; ik < 3; ik++)
						localCellCentre2D[it][ip][ik] = recvBuffer2_2D[(it * locNumId + ip) * 3 + ik];

			localCellNormal2D.resize(flowStepNum2D, vector<vec3>(locNumId, vec3(0., 0., 0.)));
			vector<double> recvBuffer3_2D(flowStepNum2D * locNumId * 3, 0.);
			MPI_CALL_TIMED(mpiPre_s, MPI_Recv(&recvBuffer3_2D[0], flowStepNum2D * locNumId * 3, MPI_DOUBLE, 0, 15, MPI_COMM_WORLD, MPI_STATUSES_IGNORE));
			for (int it = 0; it < flowStepNum2D; it++)
				for (int ip = 0; ip < locNumId; ip++)
					for (int ik = 0; ik < 3; ik++)
						localCellNormal2D[it][ip][ik] = recvBuffer3_2D[(it * locNumId + ip) * 3 + ik];
		}


	}
	/*****************************************************************/
	F1A.ReadObserverLocation(obsLocation);
	/*****************************************************************/
	if (dimension == "3D")
	{
		/*****************************************************************/
		F1A.SetFlowData(flowTime, localFlowDensity, localFlowPressure, localFlowVelocity);
		/*****************************************************************/

		/*****************************************************************/
		F1A.ReadSurface(localCellArea, localCellCentre, localCellNormal, isMovingSurface);
		/*****************************************************************/
	}

	if (dimension == "2D")
	{
		/******************** set 2D flow data **************************/
		F1A.SetflowData2D(flowTime2D, localFlowDensity2D, localFlowPressure2D, localFlowVelocity2D);

		F1A.ReadSurface2D(cellArea2D, localCellCentre2D, localCellNormal2D, isMovingSurface);
	}

	/********** 5. set output time series **************************/
	if (dimension == "3D")
	{
		// This is for parallel computation, to define global time range
		F1A.CalTimeDelayMinMax();
		double timeDelayMin = F1A.GetTimeDelayMin();

		if (myRank != 0)
		{
			MPI_CALL_TIMED(mpiPre_s, MPI_Send(&timeDelayMin, 1, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD));
		}
		else
			for (int iRank = 1; iRank < commSize; iRank++) 
			{
				double bufferMin;
				MPI_CALL_TIMED(mpiPre_s, MPI_Recv(&bufferMin, 1, MPI_DOUBLE, iRank, 1, MPI_COMM_WORLD, MPI_STATUSES_IGNORE));
				timeDelayMin = min(timeDelayMin, bufferMin);
			}
		MPI_CALL_TIMED(mpiPre_s, MPI_Bcast(&timeDelayMin, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD));

		if (isMaster) mcout << "Minimum source-observer time delay " << timeDelayMin << endl;
		double tMin = F1A.GetSourceTimeMin() + timeDelayMin;
		double tMax = F1A.GetSourceTimeMax() + timeDelayMin;
		size_t tNum = F1A.GetNumberOfSourceTime();
		F1A.InitializeObserverTime(tMin, tMax, tNum);
	}
	/*****************************************************************/

	//time calculate
	double Pre_Treat_end = MPI_Wtime();
	if (isMaster)
	{
		cout << "*************************************************" << endl;
		cout << "Pre-Treatment Time cost: " << fixed << setprecision(2);
		cout << Pre_Treat_end - Pre_Treat_start << " s" << endl;
		cout << "*************************************************" << endl;
	}


	/*****************************************************************/
	if (isMaster)
	{
		cout << "*************************************************" << endl;
		cout << __TIMESTAMP__ << " Computations start" << endl;
		cout << "*************************************************" << endl;
	}
	MPI_CALL_TIMED(mpiCompMain_s, MPI_Barrier(MPI_COMM_WORLD));
	double MPIStartTime = MPI_Wtime();


	if (dimension == "3D")
	{
		// TODO: Make a choice between A and B.
		if (type == "Time")
		{
			// A  Use time-domain formulation to solve FW-H equations     */
			F1A.CalTimeSignal(localRank, localSize);

		}

		if (type == "Frequency")
		{
			// B  Use frequency-domain formulation to solve FW-H equations   */
			/*****************************************************************/
			F1A.CalFreqSpectra(localRank, localSize);
	
			/*   Recover signal data via inverse Fourier transform          */
			/*****************************************************************/
			F1A.RecoverSignals();
		}
	}

	if (dimension == "2D")
	{
		/*C     Use 2D frequency-domain  FW-H equations                  */
		/*****************************************************************/
		F1A.CalFreqSpectra2D(localRank, localSize);
		/* Recover signal data via inverse Fourier transform 2D          */
		/*****************************************************************/
		F1A.RecoverSignals2D();
	}


	/*****************************************************************/
	/*        Collect all pressure data from each processor          */
	/*****************************************************************/
	// TODO::
	//Find a way to avoid direct access to class members

	if (dimension == "3D")
	{
		if (myRank != 0)
		{
			size_t dataLength = F1A.GetNumberOfSourceTime() * F1A.GetObserverNum();
			vector<double> buffer(dataLength);
			for (int it = 0; it < F1A.GetNumberOfSourceTime(); it++)
			{
				for (int io = 0; io < F1A.GetObserverNum(); io++)
					buffer[it * F1A.GetObserverNum() + io] = F1A.pPrime[it][io];
			}
			MPI_CALL_TIMED(mpiCompMain_s, MPI_Send(&buffer[0], (int)dataLength, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD));
		}
		else
			for (int iRank = 1; iRank < commSize; iRank++)
			{
				size_t dataLength = F1A.GetNumberOfSourceTime() * F1A.GetObserverNum();
				vector<double> buffer(dataLength);
				MPI_CALL_TIMED(mpiCompMain_s, MPI_Recv(&buffer[0], (int)dataLength, MPI_DOUBLE, iRank, 1, MPI_COMM_WORLD, MPI_STATUSES_IGNORE));
				for (int it = 0; it < F1A.GetNumberOfSourceTime(); it++)
				{
					for (int io = 0; io < F1A.GetObserverNum(); io++)
						F1A.pPrime[it][io] += buffer[it * F1A.GetObserverNum() + io];
				}
			}
	}
	/*****************************************************************/
	/*       Collect all pressure data from each processor    2D     */
	/*****************************************************************/
	
	if (dimension == "2D")
	{
		if (myRank != 0)
		{
			size_t dataLength = F1A.GetNumberOfSourceTime2D() * F1A.GetObserverNum();
			vector<double> buffer(dataLength);
			for (int it = 0; it < F1A.GetNumberOfSourceTime2D(); it++)
			{
				for (int io = 0; io < F1A.GetObserverNum(); io++)
					buffer[it * F1A.GetObserverNum() + io] = F1A.pPrime2D[it][io];
			}
			MPI_CALL_TIMED(mpiCompMain_s, MPI_Send(&buffer[0], (int)dataLength, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD));
		}
		else
			for (int iRank = 1; iRank < commSize; iRank++)
			{
				size_t dataLength = F1A.GetNumberOfSourceTime2D() * F1A.GetObserverNum();
				vector<double> buffer(dataLength);
				MPI_CALL_TIMED(mpiCompMain_s, MPI_Recv(&buffer[0], (int)dataLength, MPI_DOUBLE, iRank, 2, MPI_COMM_WORLD, MPI_STATUSES_IGNORE));
				for (int it = 0; it < F1A.GetNumberOfSourceTime2D(); it++)
				{
					for (int io = 0; io < F1A.GetObserverNum(); io++)
						F1A.pPrime2D[it][io] += buffer[it * F1A.GetObserverNum() + io];
				}
			}

	}
	/****************************************************************/
	/*    collect FreqSpectra signal data from each processor   3D  */
	/****************************************************************/
	if (dimension == "3D" && type == "SaveFrequency")
	{
		if (myRank != 0)
		{
			size_t dataLength = F1A.GetNumberOfSourceFrequency() * F1A.GetObserverNum() * 2;
			vector<double> buffer(dataLength);
			for (int it = 0; it < F1A.GetNumberOfSourceFrequency(); it++)
			{
				for (int io = 0; io < F1A.GetObserverNum(); io++)
				{
					buffer[(it * F1A.GetObserverNum() + io) * 2 + 0] = real(F1A.pPrimeCplx[it][io]);
					buffer[(it * F1A.GetObserverNum() + io) * 2 + 1] = imag(F1A.pPrimeCplx[it][io]);
				}
			}
			MPI_Send(&buffer[0], (int)dataLength, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
		}
		else
			for (int iRank = 1; iRank < commSize; iRank++)
			{
				size_t dataLength = F1A.GetNumberOfSourceFrequency() * F1A.GetObserverNum() * 2;
				vector<double> buffer(dataLength);
				MPI_Recv(&buffer[0], (int)dataLength, MPI_DOUBLE, iRank, 1, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
				for (int it = 0; it < F1A.GetNumberOfSourceFrequency(); it++)
				{
					for (int io = 0; io < F1A.GetObserverNum(); io++)
						F1A.pPrimeCplx[it][io] +=
						complex<double>(buffer[(it * F1A.GetObserverNum() + io) * 2 + 0], buffer[(it * F1A.GetObserverNum() + io) * 2 + 1]);
				}
			}
	}
	/****************************************************************/
	/*    collect FreqSpectra signal data from each processor   2D  */
	/****************************************************************/
	if (dimension == "2D" && type == "SaveFrequency")
	{
		if (myRank != 0)
		{
			size_t dataLength = F1A.GetNumberOfSourceFrequency2D() * F1A.GetObserverNum() * 2;
			vector<double> buffer(dataLength);
			for (int it = 0; it < F1A.GetNumberOfSourceFrequency2D(); it++)
			{
				for (int io = 0; io < F1A.GetObserverNum(); io++)
				{
					buffer[(it * F1A.GetObserverNum() + io) * 2 + 0] = real(F1A.pPrimeCplx2D[it][io]);
					buffer[(it * F1A.GetObserverNum() + io) * 2 + 1] = imag(F1A.pPrimeCplx2D[it][io]);
				}
			}
			MPI_CALL_TIMED(mpiCompMain_s, MPI_Send(&buffer[0], (int)dataLength, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD));
		}
		else
			for (int iRank = 1; iRank < commSize; iRank++)
			{
				size_t dataLength = F1A.GetNumberOfSourceFrequency2D() * F1A.GetObserverNum() * 2;
				vector<double> buffer(dataLength);
				MPI_CALL_TIMED(mpiCompMain_s, MPI_Recv(&buffer[0], (int)dataLength, MPI_DOUBLE, iRank, 2, MPI_COMM_WORLD, MPI_STATUSES_IGNORE));
				for (int it = 0; it < F1A.GetNumberOfSourceFrequency2D(); it++)
				{
					for (int io = 0; io < F1A.GetObserverNum(); io++)
						F1A.pPrimeCplx2D[it][io] +=
						complex<double>(buffer[(it * F1A.GetObserverNum() + io) * 2 + 0], buffer[(it * F1A.GetObserverNum() + io) * 2 + 1]);
				}
			}

	}
	// time-stamp
	double MPIEndTime = MPI_Wtime();
	if (isMaster)
	{
		cout << "*************************************************" << endl;
		cout << __TIMESTAMP__ << " Computations end " << endl;
		cout << "*************************************************" << endl;
		cout << "Time cost: " << fixed << setprecision(2);
		cout << MPIEndTime - MPIStartTime << " s" << endl;
		cout << "*************************************************" << endl;
	}


	//time calculate
	double Post_Treat_start = MPI_Wtime();


	if (myRank == 0)
	{
		string configFile = "config.inp";
		Config CF(configFile);

		double io_t0 = MPI_Wtime();

		if (dimension == "3D")
		{
			string pSignalFile;
			pSignalFile = CF.Read("pSignalOutputFileName", pSignalFile);
			F1A.SaveTimeSignals(pSignalFile);

			/*
			string pSpectrumMagFile;
			pSpectrumMagFile = CF.Read("pSpectrumMagFileName", pSpectrumMagFile);
			F1A.SaveSpectrumMag(pSpectrumMagFile);

			string pSpectrumRealFileName, pSpectrumImagFileName;
			pSpectrumRealFileName = CF.Read("pSpectrumRealFileName", pSpectrumRealFileName);
			pSpectrumImagFileName = CF.Read("pSpectrumImagFileName", pSpectrumImagFileName);
			F1A.SaveSpectrumCplx(pSpectrumRealFileName, pSpectrumImagFileName);
			*/

		}

		if (dimension == "2D")
		{
			string pSignalFile2D;
			pSignalFile2D = CF.Read("pSignalOutputFileName2D", pSignalFile2D);
			F1A.SaveTimeSignals2D(pSignalFile2D);

			/*
			string pSpectrumMagFile2D;
			pSpectrumMagFile2D = CF.Read("pSpectrumMagFileName2D", pSpectrumMagFile2D);
			F1A.SaveSpectrumMag2D(pSpectrumMagFile2D);

			string pSpectrumRealFileName2D, pSpectrumImagFileName2D;
			pSpectrumRealFileName2D = CF.Read("pSpectrumRealFileName2D", pSpectrumRealFileName2D);
			pSpectrumImagFileName2D = CF.Read("pSpectrumImagFileName2D", pSpectrumImagFileName2D);
			F1A.SaveSpectrumCplx2D(pSpectrumRealFileName2D, pSpectrumImagFileName2D);
			*/

		}

		ioWrite_s += MPI_Wtime() - io_t0;
		
	}


	//time calculate
	double Post_Treat_end = MPI_Wtime();

	double pureFWHKernel_s   = F1A.GetPureFWHKernelTime();
	double mpiCompInternal_s = F1A.GetInternalMPITime();

	double mpiComp_s  = mpiCompMain_s + mpiCompInternal_s;
	double mpiTotal_s = mpiPre_s + mpiComp_s + mpiPost_s;
	double ioTotal_s  = ioRead_s + ioWrite_s;

	char timingFileName[256];
	std::sprintf(timingFileName, "timing_stats_rank%d.dat", myRank);

	std::ofstream timingFile(timingFileName, std::ios::out);
	timingFile << "# globalRank localRank "
			<< "preTreat_s postTreat_s "
			<< "ioRead_s ioWrite_s ioTotal_s "
			<< "mpiPre_s mpiCompMain_s mpiCompInternal_s mpiComp_s mpiPost_s mpiTotal_s "
			<< "pureFWHKernel_s computeWall_s\n";

	timingFile << myRank << " "
			<< localRank << " "
			<< std::fixed << std::setprecision(6)
			<< (Pre_Treat_end - Pre_Treat_start) << " "
			<< (Post_Treat_end - Post_Treat_start) << " "
			<< ioRead_s << " "
			<< ioWrite_s << " "
			<< ioTotal_s << " "
			<< mpiPre_s << " "
			<< mpiCompMain_s << " "
			<< mpiCompInternal_s << " "
			<< mpiComp_s << " "
			<< mpiPost_s << " "
			<< mpiTotal_s << " "
			<< pureFWHKernel_s << " "
			<< (MPIEndTime - MPIStartTime) << "\n";

	timingFile.close();


	if (myRank == 0)
	{
		cout << "*************************************************" << endl;
		cout << "Post treatment Time cost: " << fixed << setprecision(2);
		cout << Post_Treat_end - Post_Treat_start << " s" << endl;
		cout << "*************************************************" << endl;
		
		cout << "*************************************************" << endl;
		cout << __TIMESTAMP__ << " Exit " << endl;
		cout << "*************************************************" << endl;
	}
	MPI_Finalize();

	return 0;
}