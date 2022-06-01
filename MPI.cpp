#include<mpi.h>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <windows.h>
using namespace std;
typedef long long ll;

#define ROW 1024
#define INTERVAL 10000\

//use	mpiexec -n 8 MPI.exe	to execute

void init(float matrix[][ROW])
{
	for (int i = 0;i < ROW;i++)
	{
		for (int j = 0;j < i;j++)
			matrix[i][j] = 0;
		for (int j = i;j < ROW;j++)
			matrix[i][j] = rand() / double(RAND_MAX) * 1000 + 1;
	}
	for (int k = 0;k < 8000;k++)
	{
		int row1 = rand() % ROW;
		int row2 = rand() % ROW;
		float mult = rand() & 1 ? 1 : -1;
		float mult2 = rand() & 1 ? 1 : -1;
		mult = mult2 * (rand() / double(RAND_MAX)) + mult;
		for (int j = 0;j < ROW;j++)
			matrix[row1][j] += mult * matrix[row2][j];
	}
}

void check(float matrix[][ROW], float result[][ROW])
{
	float error = 0;
	for (int i = 0;i < ROW;i++)
		for (int j = 0;j < ROW;j++)
			error += fabs(matrix[i][j] - result[i][j]);
	cout << error;
}


void print(float matrix[][ROW])
{
	for (int i = 0;i < ROW;i++)
	{
		for (int j = 0;j < ROW;j++)
			cout << matrix[i][j] << ' ';
		cout << endl;
	}
}

void plain(float matrix[][ROW]) {
	for (int i = 0; i < ROW; i++) {
		for (int j = i + 1; j < ROW; j++) {
			matrix[i][j] = matrix[i][j] / matrix[i][i];
		}
		matrix[i][i] = 1;
		for (int k = i + 1; k < ROW; k++) {
			for (int j = i + 1; j < ROW; j++) {
				matrix[k][j] = matrix[k][j] - matrix[i][j] * matrix[k][i];
			}
			matrix[k][i] = 0;
		}
	}
}

void rowDiv()
{
	int comm_sz;
	int my_rank;
	double start, finish, time;//计时变量
	float(*matrix)[ROW] = NULL;//global matrix
	float(*result)[ROW] = NULL;//结果矩阵，用于校验
	float(*myMat)[ROW];//local matrix
	float myDiv[ROW];//本地消元行
	MPI_Init(NULL, NULL);
	MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	//计算每个进程的工作量和偏移
	int* sendCounts = new int[comm_sz];
	int* displs = new int[comm_sz + 1];
	int pos = 0;
	fill(sendCounts, sendCounts + ROW % comm_sz, (int)ceil((float)ROW / comm_sz) * ROW);
	fill(sendCounts + ROW % comm_sz, sendCounts + comm_sz, ROW / comm_sz * ROW);
	for (int i = 0;i < comm_sz;i++)
	{
		displs[i] = pos;
		pos += sendCounts[i];
	}
	displs[comm_sz] = pos;
	myMat = new float[sendCounts[my_rank]][ROW];
	if (my_rank == 0)
	{
		matrix = new float[ROW][ROW];
		result = new float[ROW][ROW];
		init(matrix);
		start = MPI_Wtime();
	}
	MPI_Scatterv(matrix, sendCounts, displs, MPI_FLOAT, myMat, sendCounts[my_rank], MPI_FLOAT, 0, MPI_COMM_WORLD);
	for (int k = 0;k < ROW;k++)
	{
		float* curDiv;
		int src = upper_bound(displs, displs + comm_sz + 1, k * ROW) - displs - 1;
		if (src == my_rank)
		{
			curDiv = myMat[k - displs[my_rank] / ROW];
			for (int j = k + 1;j < ROW;j++)
				curDiv[j] /= curDiv[k];
			curDiv[k] = 1.0;
		}
		else curDiv = myDiv;
		MPI_Bcast(curDiv, ROW, MPI_FLOAT, src, MPI_COMM_WORLD);
		for (int i = max(displs[my_rank] / ROW, k + 1) - displs[my_rank] / ROW;i < displs[my_rank + 1] / ROW - displs[my_rank] / ROW;i++)
		{
			for (int j = k + 1;j < ROW;j++)
				myMat[i][j] -= myMat[i][k] * curDiv[j];
			myMat[i][k] = 0;
		}
	}
	MPI_Gatherv(myMat, sendCounts[my_rank], MPI_FLOAT, result, sendCounts, displs, MPI_FLOAT, 0, MPI_COMM_WORLD);
	if (my_rank == 0)
	{
		finish = MPI_Wtime();
		cout << (finish - start)*1000 << endl;
		//串行比较部分
		ll head, tail, freq;
		double time = 0;
		QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
		QueryPerformanceCounter((LARGE_INTEGER*)&head);
		plain(matrix);
		QueryPerformanceCounter((LARGE_INTEGER*)&tail);
		time = (tail - head) * 1000.0 / freq;
		std::cout << time << '\n';
		check(matrix, result);
	}
	MPI_Finalize();
}

void rowDivBlockCycle(int blockSize)
{
	int comm_sz;
	int my_rank;
	double start, finish, time;//计时变量
	float(*matrix)[ROW] = NULL;//global matrix
	float(*result)[ROW] = NULL;//结果矩阵，用于校验
	float(*myMat)[ROW];//local matrix
	float myDiv[ROW];//本地消元行
	MPI_Init(NULL, NULL);
	MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	//向量数据类型
	int fullBlockAmt = ROW / (blockSize * comm_sz);
	myMat = new float[(fullBlockAmt+1) * blockSize][ROW];
	int fullKernelAmt = (ROW / blockSize) % comm_sz;//MID_VECTOR的rank
	int leftRow = ROW % blockSize;//MID_VECTOR尾部长度
	MPI_Datatype BIG_VECTOR, SMALL_VECTOR;
	40(fullBlockAmt + 1, blockSize, blockSize * comm_sz, MPI_FLOAT, &BIG_VECTOR);
	MPI_Type_vector(fullBlockAmt, blockSize, blockSize * comm_sz, MPI_FLOAT, &SMALL_VECTOR);
	MPI_Type_commit(&BIG_VECTOR);
	MPI_Type_commit(&SMALL_VECTOR);
	if (my_rank == 0)
	{
		matrix = new float[ROW][ROW];
		result = new float[ROW][ROW];
		init(matrix);
		start = MPI_Wtime();
		int dest;
		for(dest = 1;dest<fullKernelAmt;dest++)//分发BIG_VECTOR
			MPI_Send(matrix[dest*blockSize], 1, BIG_VECTOR, dest, 0, MPI_COMM_WORLD);
		for (;dest <= comm_sz;dest++)//分发SMALL_VECTOR
			MPI_Send(matrix[dest * blockSize], 1, SMALL_VECTOR, dest, 0, MPI_COMM_WORLD);
		if (leftRow)
			MPI_Send(matrix[ROW - leftRow], leftRow * ROW, MPI_FLOAT, fullKernelAmt, 1, MPI_COMM_WORLD);
		//for (row = 0;row + blockSize <= ROW;row += blockSize)
			//row / (blockSize * comm_sz) * blockSize + row % blockSize
		//MPI_Send(matrix[row], ROW * (ROW - row), MPI_FLOAT, (row / blockSize) % comm_sz, row, MPI_COMM_WORLD);
	}	
	else
	{
		MPI_
		/*int row;
		for (row = blockSize * comm_sz;row + blockSize <= ROW;row += blockSize * comm_sz)
			MPI_Recv(myMat[row / (blockSize * comm_sz) * blockSize + row % blockSize], ROW * blockSize, MPI_FLOAT, 0, row, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
		if (row < ROW)
			MPI_Recv(myMat[row / (blockSize * comm_sz) * blockSize + row % blockSize], ROW * (ROW - row), MPI_FLOAT, 0, row, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);*/
	}
	for (int k = 0;k < ROW;k++)
	{
		float* curDiv;
		int src = upper_bound(displs, displs + comm_sz + 1, k * ROW) - displs - 1;
		if (src == my_rank)
		{
			curDiv = myMat[k - displs[my_rank] / ROW];
			for (int j = k + 1;j < ROW;j++)
				curDiv[j] /= curDiv[k];
			curDiv[k] = 1.0;
		}
		else curDiv = myDiv;
		MPI_Bcast(curDiv, ROW, MPI_FLOAT, src, MPI_COMM_WORLD);
		for (int i = max(displs[my_rank] / ROW, k + 1) - displs[my_rank] / ROW;i < displs[my_rank + 1] / ROW - displs[my_rank] / ROW;i++)
		{
			for (int j = k + 1;j < ROW;j++)
				myMat[i][j] -= myMat[i][k] * curDiv[j];
			myMat[i][k] = 0;
		}
	}
	MPI_Gatherv(myMat, sendCounts[my_rank], MPI_FLOAT, result, sendCounts, displs, MPI_FLOAT, 0, MPI_COMM_WORLD);
	if (my_rank == 0)
	{
		finish = MPI_Wtime();
		cout << (finish - start) * 1000 << endl;
		//串行比较部分
		ll head, tail, freq;
		double time = 0;
		QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
		QueryPerformanceCounter((LARGE_INTEGER*)&head);
		plain(matrix);
		QueryPerformanceCounter((LARGE_INTEGER*)&tail);
		time = (tail - head) * 1000.0 / freq;
		std::cout << time << '\n';
		check(matrix, result);
	}
	MPI_Finalize();
}
int main()
{
	rowDiv();
	return 0;
}