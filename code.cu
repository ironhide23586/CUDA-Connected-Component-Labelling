
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <stdio.h>

#include <input_large.h>

#define indexFrom2D(i, j) ((i) * cols + (j))
#define greatest(a, b) ((a > b) ? a : b)
#define least(a, b) ((a < b) ? a : b)

#define MAP_SAMPLING_SIZE_SIDE  3 
#define MAP_SAMPLING_SIZE 9

#define LAST_THREADBLOCK (effectiveBlockId == lastThreadBlockIndex)

#define THREADS_PER_BLOCK 1024
#define SERIAL_SUM_LIMIT 10

#define MAX_NO_OF_THREADBLOCKS 65535

#define MAP_ROOT h_map[rootLinearIndex]
#define MAP_RIGHT h_map[rootRightIndex]
#define MAP_DOWN h_map[rootDownIndex]
#define MAP_CORNER h_map[rootCornerIndex]


__host__ __device__ unsigned int ceil_h_d(float f)
{
	unsigned int tmp = (unsigned int) f;
	if (f > tmp)
		tmp++;
	return tmp;
}



__host__ unsigned int nearestLesserSquare(unsigned int x)
{
	unsigned int tmp = sqrt((float) x);
	return tmp * tmp;
}

__host__ unsigned int getReqThreadUnitsCountCCL(unsigned int rows, unsigned int cols)
{
	return ceil_h_d((float) rows / 2) * ceil_h_d((float) cols / 2);
}



__device__ unsigned int getRealRow(unsigned int cols, unsigned int effectiveblockIndex)
{
	return (threadIdx.y - 1) + ceil_h_d((float) (effectiveblockIndex + 1) / (cols - 2));
}

__device__ unsigned int getRealCol(unsigned int cols, unsigned int effectiveblockIndex)
{
	return threadIdx.x + (effectiveblockIndex % (cols - 2));
}



__global__ void processMapSampleHillDale(unsigned int *d_map, unsigned int rows, unsigned int cols, char *d_hillDaleRemaining, unsigned int threadBlockBatchIndex, unsigned int *d_aux_map)
{
	unsigned int effectiveBlockIndex = blockIdx.x + MAX_NO_OF_THREADBLOCKS * threadBlockBatchIndex;
	unsigned int mapRow = threadIdx.y, mapCol = threadIdx.x;
	unsigned int mapLinearIndex = mapRow * 3 + mapCol;

	unsigned int realRow = getRealRow(cols, effectiveBlockIndex), realCol = getRealCol(cols, effectiveBlockIndex);
	unsigned int linearIndexFrom2D = indexFrom2D(realRow, realCol);

	if (linearIndexFrom2D == 0)
		d_hillDaleRemaining[0] = 0;

	__shared__ unsigned int mapContents[MAP_SAMPLING_SIZE_SIDE][MAP_SAMPLING_SIZE_SIDE];
	__shared__ int cellDecider;

	cellDecider = 0;

	mapContents[mapRow][mapCol] = d_map[linearIndexFrom2D];
	__syncthreads();


	if (mapContents[mapRow][mapCol] < mapContents[1][1])
		atomicAdd(&cellDecider, 1);
	else if (mapContents[mapRow][mapCol] > mapContents[1][1])
		atomicSub(&cellDecider, 1);
	__syncthreads();

	if (cellDecider == 8) //HILL
	{
		//////////////PARALLEL AVERAGE
		if (mapLinearIndex == 0)
		{
			mapContents[0][0] += mapContents[0][1];
			d_hillDaleRemaining[0] = 1;
		}
		else if (mapLinearIndex == 2)
			mapContents[0][2] += mapContents[1][2];
		else if (mapLinearIndex == 8)
			mapContents[2][2] += mapContents[2][1];
		else if (mapLinearIndex == 6)
			mapContents[2][0] += mapContents[1][0];
		__syncthreads();

		if (mapLinearIndex == 0)
			mapContents[0][0] += mapContents[0][2];
		else if (mapLinearIndex == 8)
			mapContents[2][2] += mapContents[2][0];
		__syncthreads();

		if (mapLinearIndex == 4)
		{
			(mapContents[0][0] += mapContents[2][2]) /= 8;
			d_aux_map[linearIndexFrom2D] = mapContents[0][0];
		}
	}

	else if (cellDecider == (-8)) //DALE
	{
		__shared__ unsigned int nos[8];
		__shared__ char isSorted[2], maxPartitions;

		if (mapLinearIndex == 4)
		{
			nos[4] = mapContents[2][2];
			isSorted[0] = 0;
			isSorted[1] = 0;
			maxPartitions = 4;
			d_hillDaleRemaining[0] = 1;
		}
		else
			nos[mapLinearIndex] = mapContents[mapRow][mapCol];
		__syncthreads();

		unsigned int tmp, v1 = 2 * mapLinearIndex, v2 = v1 + 1, v3 = v1 + 2;

		while (!(isSorted[0] && isSorted[1]))
		{
			if (mapLinearIndex < 2)
				isSorted[mapLinearIndex] = 1;
			__syncthreads();
			if (mapLinearIndex < maxPartitions) //EVEN
			{
				if (nos[v1] > nos[v2])
				{
					tmp = nos[v2];
					nos[v2] = nos[v1];
					nos[v1] = tmp;
					isSorted[0] = 0;
				}
				maxPartitions = 3;
			}
			__syncthreads();

			if (mapLinearIndex < maxPartitions) //ODD
			{
				if (nos[v2] > nos[v3])
				{
					tmp = nos[v2];
					nos[v2] = nos[v3];
					nos[v3] = tmp;
					isSorted[1] = 0;
				}
				maxPartitions = 4;
			}
			__syncthreads();
		}

		if (mapLinearIndex == 4)
		{
			(nos[3] += nos[4]) /= 2;
			d_aux_map[linearIndexFrom2D] = nos[3];
		}
	}
}

__global__ void calculateAverage(unsigned int *d_map, unsigned int mapSize, unsigned int *d_average, unsigned int lastThreadBlockLastIndex, unsigned int lastThreadBlockIndex, char odd, unsigned int threadBlockBatchIndex)
{
	unsigned tmp = threadIdx.x * 2, effectiveBlockId = blockIdx.x + MAX_NO_OF_THREADBLOCKS * threadBlockBatchIndex;
	unsigned int linearIndexFrom2D = (effectiveBlockId * THREADS_PER_BLOCK * 2) + tmp, incrLinearIndexFrom2D = linearIndexFrom2D + 1;

	__shared__ unsigned int vals[THREADS_PER_BLOCK];
	vals[threadIdx.x] = 0;

	if (linearIndexFrom2D < mapSize)
	{

		vals[threadIdx.x] = d_map[linearIndexFrom2D];
		if (mapSize > incrLinearIndexFrom2D)
			vals[threadIdx.x] += d_map[incrLinearIndexFrom2D];

		unsigned int adderIndex = tmp, incrAdderIndex = tmp + 1, i = 1;

		unsigned int compareVal = LAST_THREADBLOCK ? lastThreadBlockLastIndex : THREADS_PER_BLOCK;
		__syncthreads();

		while (incrAdderIndex < compareVal)
		{
			vals[adderIndex] += vals[incrAdderIndex];
			__syncthreads();

			i *= 2;
			adderIndex = i * tmp;
			incrAdderIndex = adderIndex + i;
		}
	
		if (threadIdx.x == 0)
		{
			d_average[effectiveBlockId] = vals[0];
			if (((LAST_THREADBLOCK) && odd) && (lastThreadBlockLastIndex > 0))
				d_average[effectiveBlockId] += vals[lastThreadBlockLastIndex];
		}
	}
}

__global__ void threshold(unsigned int *d_map, unsigned int thresholdVal, unsigned int mapSize, unsigned int threadBlockBatchIndex)
{
	unsigned int effectiveBlockId = blockIdx.x + MAX_NO_OF_THREADBLOCKS * threadBlockBatchIndex;
	unsigned int linearIndexFrom2D = (effectiveBlockId * THREADS_PER_BLOCK) + threadIdx.x;

	if (linearIndexFrom2D < mapSize)
	{
		if (d_map[linearIndexFrom2D] < thresholdVal)
			d_map[linearIndexFrom2D] = 0;
		else
			d_map[linearIndexFrom2D] = 1;
	}
}

__global__ void assignBlockNumbers(unsigned int *d_map, unsigned int rows, unsigned int cols, unsigned int blockMatrixRowLength, unsigned int threadBlockBatchIndex)
{
	unsigned int effectiveBlockId = blockIdx.x + MAX_NO_OF_THREADBLOCKS * threadBlockBatchIndex;
	unsigned int n = effectiveBlockId + 1, nFloor = n; //n -> Block ID to be assigned.

	if ((n % blockMatrixRowLength) == 0)
		nFloor--;
	unsigned int rootLinearIndex = ((n - 1) / blockMatrixRowLength) * cols + ((2 * n) - 2 - ((nFloor / blockMatrixRowLength) * ((cols % 2) > 0)));

	unsigned int r = rootLinearIndex / cols, c = rootLinearIndex - r * cols;

	if (threadIdx.x < 2)
		c += threadIdx.x;
	else
	{
		r++;
		c += threadIdx.x - 2;
	}
	unsigned int linearIndex = indexFrom2D(r, c);

	if (((r < rows) && (c < cols)) && d_map[linearIndex] == 1)
		d_map[linearIndex] = n;
}


__global__ void processMapSampleCCLZeroIndexHorizontal0(unsigned int *d_map, char *d_ifLesserFound, unsigned int rows, unsigned int cols, unsigned int blockMatrixRowLength, unsigned int threadBlockBatchIndex)
{
	unsigned int effectiveBlockId = blockIdx.x + MAX_NO_OF_THREADBLOCKS * threadBlockBatchIndex;
	unsigned int n = effectiveBlockId + 1, nFloor = n; //n -> Block ID to be assigned.

	if ((n % blockMatrixRowLength) == 0)
		nFloor--;
	unsigned int rootLinearIndex = ((n - 1) / blockMatrixRowLength) * cols + ((2 * n) - 2 - ((nFloor / blockMatrixRowLength) * ((cols % 2) > 0)));

	unsigned int r = rootLinearIndex / cols, c = rootLinearIndex - r * cols;

	if (threadIdx.x < 2)
		c += threadIdx.x;
	else
	{
		r++;
		c += threadIdx.x - 2;
	}
	unsigned int linearIndex = indexFrom2D(r, c);

	__shared__ unsigned int sample[4], leastNum, leastCandidates[2];
	if ((r < rows) && (c < cols))
		sample[threadIdx.x] = d_map[linearIndex];
	else
		sample[threadIdx.x] = 0;
	__syncthreads();

	if (threadIdx.x < 2)
	{
		unsigned int tmp1 = threadIdx.x * 2, tmp2 = tmp1 + 1;

		if ((sample[tmp1] > 0) && (sample[tmp2] > 0))
		{
			if (sample[tmp1] < sample[tmp2])
			{
				d_ifLesserFound[0] = 1;
				leastCandidates[threadIdx.x] = sample[tmp1];
			}
			else if (sample[tmp2] < sample[tmp1])
			{
				d_ifLesserFound[0] = 1;
				leastCandidates[threadIdx.x] = sample[tmp2];
			}
			else
				leastCandidates[threadIdx.x] = sample[tmp1];
		}
		else if (sample[tmp1] == 0)
			leastCandidates[threadIdx.x] = sample[tmp2];
		else
			leastCandidates[threadIdx.x] = sample[tmp1];
	}
	__syncthreads();
	
	if (threadIdx.x == 0)
	{
		if ((leastCandidates[0] > 0) && (leastCandidates[1] > 0))
		{
			if (leastCandidates[0] < leastCandidates[1])
			{
				d_ifLesserFound[0] = 1;
				leastNum = leastCandidates[0];
			}
			else if (leastCandidates[1] < leastCandidates[0])
			{
				d_ifLesserFound[0] = 1;
				leastNum = leastCandidates[1];
			}
			else
				leastNum = leastCandidates[0];
		}
		else if (leastCandidates[0] == 0)
			leastNum = leastCandidates[1];
		else
			leastNum = leastCandidates[0];
	}
	__syncthreads();

	if (((r < rows) && (c < cols)) && (d_map[linearIndex] > 0)) 
		d_map[linearIndex] = leastNum;
}

__global__ void processMapSampleCCLOneIndexHorizontal1(unsigned int *d_map, char *d_ifLesserFound, unsigned int rows, unsigned int cols, unsigned int blockMatrixRowLength, unsigned int threadBlockBatchIndex)
{
	unsigned int effectiveBlockId = blockIdx.x + MAX_NO_OF_THREADBLOCKS * threadBlockBatchIndex;
	unsigned int n = effectiveBlockId + 1, nFloor = n;

	if ((n % blockMatrixRowLength) == 0)
		nFloor--;

	unsigned int rootLinearIndex = (((n - 1) / blockMatrixRowLength) * (cols)) + ((2 * n) - 1 + ((nFloor / blockMatrixRowLength) * ((cols % 2) > 0)));

	unsigned int r = rootLinearIndex / cols, c = rootLinearIndex - r * cols;

	if (threadIdx.x < 2)
		c += threadIdx.x;
	else
	{
		r++;
		c += threadIdx.x - 2;
	}

	unsigned int linearIndex = indexFrom2D(r, c);

	__shared__ unsigned int sample[4], leastNum, leastCandidates[2];
	if ((r < rows) && (c < cols))
		sample[threadIdx.x] = d_map[linearIndex];
	else
		sample[threadIdx.x] = 0;
	__syncthreads();

	if (threadIdx.x < 2)
	{
		unsigned int tmp1 = threadIdx.x * 2, tmp2 = tmp1 + 1;

		if ((sample[tmp1] > 0) && (sample[tmp2] > 0))
		{
			if (sample[tmp1] < sample[tmp2])
			{
				d_ifLesserFound[1] = 1;
				leastCandidates[threadIdx.x] = sample[tmp1];
			}
			else if (sample[tmp2] < sample[tmp1])
			{
				d_ifLesserFound[1] = 1;
				leastCandidates[threadIdx.x] = sample[tmp2];
			}
			else
				leastCandidates[threadIdx.x] = sample[tmp1];
		}
		else if (sample[tmp1] == 0)
			leastCandidates[threadIdx.x] = sample[tmp2];
		else
			leastCandidates[threadIdx.x] = sample[tmp1];
	}
	__syncthreads();
	
	if (threadIdx.x == 0)
	{
		if ((leastCandidates[0] > 0) && (leastCandidates[1] > 0))
		{
			if (leastCandidates[0] < leastCandidates[1])
			{
				d_ifLesserFound[1] = 1;
				leastNum = leastCandidates[0];
			}
			else if (leastCandidates[1] < leastCandidates[0])
			{
				d_ifLesserFound[1] = 1;
				leastNum = leastCandidates[1];
			}
			else
				leastNum = leastCandidates[0];
		}
		else if (leastCandidates[0] == 0)
			leastNum = leastCandidates[1];
		else
			leastNum = leastCandidates[0];
	}
	__syncthreads();

	if (((r < rows) && ((c > 0) && (c < cols))) && (d_map[linearIndex] > 0)) 
		d_map[linearIndex] = leastNum;
}

__global__ void processMapSampleCCLZeroIndexVertical2(unsigned int *d_map, char *d_ifLesserFound, unsigned int rows, unsigned int cols, unsigned int blockMatrixRowLength, unsigned int threadBlockBatchIndex)
{
	unsigned int effectiveBlockId = blockIdx.x + MAX_NO_OF_THREADBLOCKS * threadBlockBatchIndex;
	unsigned int n = effectiveBlockId + 1, nFloor = n;

	if ((n % blockMatrixRowLength) == 0)
		nFloor--;

	unsigned int rootLinearIndex = (((n - 1) / blockMatrixRowLength) * (cols)) + (cols + (2 * n) - 2 - ((nFloor / blockMatrixRowLength) * ((cols % 2) > 0)));

	unsigned int r = rootLinearIndex / cols, c = rootLinearIndex - r * cols;

	if (threadIdx.x < 2)
		c += threadIdx.x;
	else
	{
		r++;
		c += threadIdx.x - 2;
	}

	unsigned int linearIndex = indexFrom2D(r, c);

	__shared__ unsigned int sample[4], leastNum, leastCandidates[2];
	if ((r < rows) && (c < cols))
		sample[threadIdx.x] = d_map[linearIndex];
	else
		sample[threadIdx.x] = 0;
	__syncthreads();

	if (threadIdx.x < 2)
	{
		unsigned int tmp1 = threadIdx.x * 2, tmp2 = tmp1 + 1;

		if ((sample[tmp1] > 0) && (sample[tmp2] > 0))
		{
			if (sample[tmp1] < sample[tmp2])
			{
				d_ifLesserFound[2] = 1;
				leastCandidates[threadIdx.x] = sample[tmp1];
			}
			else if (sample[tmp2] < sample[tmp1])
			{
				d_ifLesserFound[2] = 1;
				leastCandidates[threadIdx.x] = sample[tmp2];
			}
			else
				leastCandidates[threadIdx.x] = sample[tmp1];
		}
		else if (sample[tmp1] == 0)
			leastCandidates[threadIdx.x] = sample[tmp2];
		else
			leastCandidates[threadIdx.x] = sample[tmp1];
	}
	__syncthreads();
	
	if (threadIdx.x == 0)
	{
		if ((leastCandidates[0] > 0) && (leastCandidates[1] > 0))
		{
			if (leastCandidates[0] < leastCandidates[1])
			{
				d_ifLesserFound[2] = 1;
				leastNum = leastCandidates[0];
			}
			else if (leastCandidates[1] < leastCandidates[0])
			{
				d_ifLesserFound[2] = 1;
				leastNum = leastCandidates[1];
			}
			else
				leastNum = leastCandidates[0];
		}
		else if (leastCandidates[0] == 0)
			leastNum = leastCandidates[1];
		else
			leastNum = leastCandidates[0];
	}
	__syncthreads();

	if ((((r > 0) && (r < rows)) && (c < cols)) && (d_map[linearIndex] > 0)) 
		d_map[linearIndex] = leastNum;
}

__global__ void processMapSampleCCLOneIndexVertical3(unsigned int *d_map, char *d_ifLesserFound, unsigned int rows, unsigned int cols, unsigned int blockMatrixRowLength, unsigned int threadBlockBatchIndex)
{
	unsigned int effectiveBlockId = blockIdx.x + MAX_NO_OF_THREADBLOCKS * threadBlockBatchIndex;
	unsigned int n = effectiveBlockId + 1, nFloor = n;

	if ((n % blockMatrixRowLength) == 0)
		nFloor--;

	unsigned int rootLinearIndex = (((n - 1) / blockMatrixRowLength) * (cols)) + (cols + (2 * n) - 1 - ((nFloor / blockMatrixRowLength) * ((cols % 2) > 0)));

	unsigned int r = rootLinearIndex / cols, c = rootLinearIndex - r * cols;

	if (threadIdx.x < 2)
		c += threadIdx.x;
	else
	{
		r++;
		c += threadIdx.x - 2;
	}

	unsigned int linearIndex = indexFrom2D(r, c);

	__shared__ unsigned int sample[4], leastNum, leastCandidates[2];
	if ((r < rows) && (c < cols))
		sample[threadIdx.x] = d_map[linearIndex];
	else
		sample[threadIdx.x] = 0;
	__syncthreads();

	if (threadIdx.x < 2)
	{
		unsigned int tmp1 = threadIdx.x * 2, tmp2 = tmp1 + 1;

		if ((sample[tmp1] > 0) && (sample[tmp2] > 0))
		{
			if (sample[tmp1] < sample[tmp2])
			{
				d_ifLesserFound[3] = 1;
				leastCandidates[threadIdx.x] = sample[tmp1];
			}
			else if (sample[tmp2] < sample[tmp1])
			{
				d_ifLesserFound[3] = 1;
				leastCandidates[threadIdx.x] = sample[tmp2];
			}
			else
				leastCandidates[threadIdx.x] = sample[tmp1];
		}
		else if (sample[tmp1] == 0)
			leastCandidates[threadIdx.x] = sample[tmp2];
		else
			leastCandidates[threadIdx.x] = sample[tmp1];
	}
	__syncthreads();
	
	if (threadIdx.x == 0)
	{
		if ((leastCandidates[0] > 0) && (leastCandidates[1] > 0))
		{
			if (leastCandidates[0] < leastCandidates[1])
			{
				d_ifLesserFound[3] = 1;
				leastNum = leastCandidates[0];
			}
			else if (leastCandidates[1] < leastCandidates[0])
			{
				d_ifLesserFound[3] = 1;
				leastNum = leastCandidates[1];
			}
			else
				leastNum = leastCandidates[0];
		}
		else if (leastCandidates[0] == 0)
			leastNum = leastCandidates[1];
		else
			leastNum = leastCandidates[0];
	}
	__syncthreads();

	if ((((r > 0) && (r < rows)) && ((c > 0) && (c < cols))) && (d_map[linearIndex] > 0)) 
		d_map[linearIndex] = leastNum;
}


int main()
{
	dim3 dimMap(MAP_SAMPLING_SIZE_SIDE, MAP_SAMPLING_SIZE_SIDE);

	unsigned int *h_input = get_input();
	unsigned int NO_OF_MAPS = h_input[0];

	unsigned int rootLinearIndex, rootRightIndex, rootDownIndex, rootCornerIndex, leastNum1, leastNum2, leastNum, finalRowIndex, finalColIndex;

	unsigned int rows = h_input[1], cols = h_input[2];
	unsigned int currMapSize = rows * cols, r, c;

	unsigned int inputReadIndex = 3, currMapSizeBytes = currMapSize * sizeof(unsigned int), currMapSizeChar = currMapSize * sizeof(char); //Begin reading map
	unsigned int *d_map; //Variable to hold map read on GPU.
	
	unsigned int *h_map;
	unsigned int *h_buff, topIndex, k, l, j;
	unsigned int *h_average;
	char tmp;

	unsigned int mapCount = 0, mapCountStop = NO_OF_MAPS - 1, iterations;
	
	unsigned int *h_input_tmp = &h_input[inputReadIndex], reqThreadUnits1, reqThreadUnits2, reqThreadUnits3, reqThreadUnits4;

	char *d_hillDaleRemaining, h_hillDaleRemaining;

	unsigned int *d_average1, *d_average2, finalAverage, n, reqThreadUnits2FullBlockCount, maxMapDimensionLength, i;

	unsigned int threadBlockBatchCount, extraThreadBlockCount;
	
	cudaMalloc((void **) &d_hillDaleRemaining, sizeof(char));

	float threadBlockBatchCountFloat;
	unsigned int *d_aux_map;
	unsigned int lastTBLastIndex;
	unsigned int lastThreadBlockElementCount;
	char odd;
	unsigned int reqThreadUnits2Bytes;
	unsigned int reqThreadUnits3Bytes;
	char *d_ifLesserFound;
	unsigned int reqThreadUnits3FullBlockCount;
	char h_ifLesserFound[4];
	
	while (true)
	{
		cudaMalloc((void **) &d_map, currMapSizeBytes);
		cudaMemcpy(d_map, h_input_tmp, currMapSizeBytes, cudaMemcpyHostToDevice);

		//*****************************************MAP PROCESSING****************************************//

		//------------------------------------HILL & DALE-------------------------------//
		reqThreadUnits1 = (rows - 2) * (cols - 2);
		iterations = 0;
		h_hillDaleRemaining = 1;
		
		threadBlockBatchCountFloat = (float) reqThreadUnits1 / MAX_NO_OF_THREADBLOCKS;

		cudaMalloc((void **) &d_aux_map, currMapSizeBytes);
		cudaMemcpy(d_aux_map, d_map, currMapSizeBytes, cudaMemcpyDeviceToDevice);

		if (threadBlockBatchCountFloat <= 1)
		{
			while ((h_hillDaleRemaining) && (iterations < NUM_ITERATIONS))
			{
				processMapSampleHillDale<<<reqThreadUnits1, dimMap>>>(d_map, rows, cols, d_hillDaleRemaining, 0, d_aux_map);

				cudaMemcpy(d_map, d_aux_map, currMapSizeBytes, cudaMemcpyDeviceToDevice);

				iterations++;
				cudaMemcpy(&h_hillDaleRemaining, d_hillDaleRemaining, sizeof(h_hillDaleRemaining), cudaMemcpyDeviceToHost);
			}
		}
		else
		{
			threadBlockBatchCount = (unsigned int) threadBlockBatchCountFloat;
			extraThreadBlockCount = reqThreadUnits1 - threadBlockBatchCount * MAX_NO_OF_THREADBLOCKS;
			while ((h_hillDaleRemaining) && (iterations < NUM_ITERATIONS))
			{
				for (i = 0; i < threadBlockBatchCount; i++)
				{
					processMapSampleHillDale<<<MAX_NO_OF_THREADBLOCKS, dimMap>>>(d_map, rows, cols, d_hillDaleRemaining, i, d_aux_map);
				}
				processMapSampleHillDale<<<extraThreadBlockCount, dimMap>>>(d_map, rows, cols, d_hillDaleRemaining, threadBlockBatchCount, d_aux_map);
				
				cudaMemcpy(d_map, d_aux_map, currMapSizeBytes, cudaMemcpyDeviceToDevice);

				iterations++;
				cudaMemcpy(&h_hillDaleRemaining, d_hillDaleRemaining, sizeof(h_hillDaleRemaining), cudaMemcpyDeviceToHost);
			}
		}
		cudaMemcpy(d_map, d_aux_map, currMapSizeBytes, cudaMemcpyDeviceToDevice);
		cudaFree(d_aux_map);
		//------------------------------------HILL & DALE-------------------------------//



		//------------------------------------AVERAGE FINDING-------------------------------//
		reqThreadUnits2 = ceil_h_d((float) ceil_h_d((float) currMapSize / 2) / THREADS_PER_BLOCK);

		reqThreadUnits2FullBlockCount = (reqThreadUnits2 - 1) * THREADS_PER_BLOCK;
		cudaMalloc((void **) &d_average1, reqThreadUnits2 * sizeof(unsigned int));

		lastTBLastIndex = ceil_h_d((float) currMapSize / 2 - reqThreadUnits2FullBlockCount) - 1;
		
		lastThreadBlockElementCount = currMapSize - reqThreadUnits2FullBlockCount * 2;
		odd = ~(lastThreadBlockElementCount && (!(lastThreadBlockElementCount & (lastThreadBlockElementCount - 1)))); //To check if lastThreadBlockElementCount is a power of 2

		threadBlockBatchCountFloat = (float) reqThreadUnits2 / MAX_NO_OF_THREADBLOCKS;
		if ((threadBlockBatchCountFloat <= 1))
		{
			calculateAverage<<<reqThreadUnits2, THREADS_PER_BLOCK>>>(d_map, currMapSize, d_average1, lastTBLastIndex, (reqThreadUnits2 - 1), odd, 0);
		}
		else
		{
			threadBlockBatchCount = (unsigned int) threadBlockBatchCountFloat;
			extraThreadBlockCount = reqThreadUnits1 - threadBlockBatchCount * MAX_NO_OF_THREADBLOCKS;
			for (i = 0; i < threadBlockBatchCount; i++)
			{
				calculateAverage<<<MAX_NO_OF_THREADBLOCKS, THREADS_PER_BLOCK>>>(d_map, currMapSize, d_average1, lastTBLastIndex, (reqThreadUnits2 - 1), odd, i);

			}
			calculateAverage<<<extraThreadBlockCount, THREADS_PER_BLOCK>>>(d_map, currMapSize, d_average1, lastTBLastIndex, (reqThreadUnits2 - 1), odd, threadBlockBatchCount);
		}

		maxMapDimensionLength = greatest(rows, cols);

		if (reqThreadUnits2 < SERIAL_SUM_LIMIT)
		{
			reqThreadUnits2Bytes = reqThreadUnits2 * sizeof(unsigned int);
			h_average = (unsigned int *)malloc(reqThreadUnits2Bytes);
			cudaMemcpy(h_average, d_average1, reqThreadUnits2Bytes, cudaMemcpyDeviceToHost);
			finalAverage = 0;

			for (n = 0; n < reqThreadUnits2; n++)
			{
				finalAverage += h_average[n];
			}

			finalAverage /= currMapSize;
		}
		else
		{
			finalAverage = 0;
			reqThreadUnits3 = ceil_h_d((float) ceil_h_d((float) reqThreadUnits2 / 2) / THREADS_PER_BLOCK);
			reqThreadUnits3Bytes = reqThreadUnits3 * sizeof(unsigned int);
			reqThreadUnits3FullBlockCount = (reqThreadUnits3 - 1) * THREADS_PER_BLOCK;
			
			lastTBLastIndex = ceil_h_d((float) reqThreadUnits2 / 2 - reqThreadUnits3FullBlockCount) - 1;

			lastThreadBlockElementCount = reqThreadUnits2 - reqThreadUnits3FullBlockCount * 2;
			odd = ~(lastThreadBlockElementCount && (!(lastThreadBlockElementCount & (lastThreadBlockElementCount - 1))));

			cudaMalloc((void **) &d_average2, reqThreadUnits3Bytes);

			calculateAverage<<<reqThreadUnits3, THREADS_PER_BLOCK>>>(d_average1, reqThreadUnits2, d_average2, lastTBLastIndex, (reqThreadUnits3 - 1), odd, 0);

			h_average = (unsigned int *)malloc(reqThreadUnits3Bytes);
			cudaMemcpy(h_average, d_average2, reqThreadUnits3Bytes, cudaMemcpyDeviceToHost);

			for (n = 0; n < reqThreadUnits3; n++)
			{
				finalAverage += h_average[n];
			}

			finalAverage /= currMapSize;
		}
		//------------------------------------AVERAGE FINDING-------------------------------//



		//------------------------------------THRESHOLDING-------------------------------//
		reqThreadUnits3 = ceil_h_d((float) currMapSize / THREADS_PER_BLOCK);

		threadBlockBatchCountFloat = (float) reqThreadUnits3 / MAX_NO_OF_THREADBLOCKS;
		if (threadBlockBatchCountFloat <= 1)
		{
			threshold<<<reqThreadUnits3, THREADS_PER_BLOCK>>>(d_map, finalAverage, currMapSize, 0);
		}
		else
		{
			threadBlockBatchCount = (unsigned int) threadBlockBatchCountFloat;
			extraThreadBlockCount = reqThreadUnits3 - threadBlockBatchCount * MAX_NO_OF_THREADBLOCKS;
			for (i = 0; i < threadBlockBatchCount; i++)
			{
				threshold<<<MAX_NO_OF_THREADBLOCKS, THREADS_PER_BLOCK>>>(d_map, finalAverage, currMapSize, i);
			}
			threshold<<<extraThreadBlockCount, THREADS_PER_BLOCK>>>(d_map, finalAverage, currMapSize, threadBlockBatchCount);
		}
		//------------------------------------THRESHOLDING-------------------------------//

		

		//------------------------------------CONNECTED COMPONENT LABELLING-------------------------------//
		reqThreadUnits1 = getReqThreadUnitsCountCCL(rows, cols);
		reqThreadUnits2 = getReqThreadUnitsCountCCL(rows, (cols - 1));
		reqThreadUnits3 = getReqThreadUnitsCountCCL((rows - 1), cols);
		reqThreadUnits4 = getReqThreadUnitsCountCCL((rows - 1), (cols - 1));
		
		//########################################################################################################//
		threadBlockBatchCountFloat = (float) reqThreadUnits1 / MAX_NO_OF_THREADBLOCKS;
		if (threadBlockBatchCountFloat <= 1)
		{
			assignBlockNumbers<<<reqThreadUnits1, 4>>>(d_map, rows, cols, ceil_h_d((float) cols / 2), 0);
		}
		else
		{
			threadBlockBatchCount = (unsigned int) threadBlockBatchCountFloat;
			extraThreadBlockCount = reqThreadUnits1 - threadBlockBatchCount * MAX_NO_OF_THREADBLOCKS;
			for (i = 0; i < threadBlockBatchCount; i++)
			{
				assignBlockNumbers<<<MAX_NO_OF_THREADBLOCKS, 4>>>(d_map, rows, cols, ceil_h_d((float) cols / 2), i);
			}
			assignBlockNumbers<<<extraThreadBlockCount, 4>>>(d_map, rows, cols, ceil_h_d((float) cols / 2), threadBlockBatchCount);
		}
		//########################################################################################################//

		finalRowIndex = rows - 1;
		finalColIndex = cols - 1;
		h_map = (unsigned int *)malloc(currMapSizeBytes);
		cudaMemcpy(h_map, d_map, currMapSizeBytes, cudaMemcpyDeviceToHost);

		for (r = 0; r < finalRowIndex; r++)
		{
			for (c = 0; c < finalColIndex; c++)
			{
				rootLinearIndex = indexFrom2D(r, c);
				rootRightIndex = indexFrom2D(r, c + 1);
				rootDownIndex = indexFrom2D(r + 1, c);
				rootCornerIndex = indexFrom2D(r + 1, c + 1);

				if ((MAP_ROOT > 0) && (MAP_RIGHT > 0))
					leastNum1 = least(MAP_ROOT, MAP_RIGHT);
				else if (MAP_ROOT == 0)
					leastNum1 = MAP_RIGHT;
				else
					leastNum1 = MAP_ROOT;

				if ((MAP_DOWN > 0) && (MAP_CORNER > 0))
					leastNum2 = least(MAP_DOWN, MAP_CORNER);
				else if (MAP_DOWN == 0)
					leastNum2 = MAP_CORNER;
				else 
					leastNum2 = MAP_DOWN;

				if ((leastNum1 > 0) && (leastNum2 > 0))
					leastNum = least(leastNum1, leastNum2);
				else if (leastNum1 == 0)
					leastNum = leastNum2;
				else
					leastNum = leastNum1;

				if (leastNum > 0)
				{
					if (MAP_ROOT > 0)
						MAP_ROOT = leastNum;
					if (MAP_RIGHT > 0)
						MAP_RIGHT = leastNum;
					if (MAP_DOWN > 0)
						MAP_DOWN = leastNum;
					if (MAP_CORNER > 0)
						MAP_CORNER = leastNum;
				}
			}
		}

		cudaMemcpy(d_map, h_map, currMapSizeBytes, cudaMemcpyHostToDevice);
		free(h_map);

		h_ifLesserFound[0] = 1;

		cudaMalloc((void **) &d_ifLesserFound, 4 * sizeof(char));

		while (h_ifLesserFound[0] | h_ifLesserFound[1] | h_ifLesserFound[2] | h_ifLesserFound[3])
		{
			cudaMemset(d_ifLesserFound, 0, 4 * sizeof(char));
			

			threadBlockBatchCountFloat = (float) reqThreadUnits2 / MAX_NO_OF_THREADBLOCKS;
			if (threadBlockBatchCountFloat <= 1)
			{
				processMapSampleCCLOneIndexHorizontal1<<<reqThreadUnits2, 4>>>(d_map, d_ifLesserFound, rows, cols, ceil_h_d((float) (cols - 1) / 2), 0);
			}
			else
			{
				threadBlockBatchCount = (unsigned int) threadBlockBatchCountFloat;
				extraThreadBlockCount = reqThreadUnits2 - threadBlockBatchCount * MAX_NO_OF_THREADBLOCKS;
				for (i = 0; i < threadBlockBatchCount; i++)
				{
					processMapSampleCCLOneIndexHorizontal1<<<MAX_NO_OF_THREADBLOCKS, 4>>>(d_map, d_ifLesserFound, rows, cols, ceil_h_d((float) (cols - 1) / 2), i);
				}
				processMapSampleCCLOneIndexHorizontal1<<<extraThreadBlockCount, 4>>>(d_map, d_ifLesserFound, rows, cols, ceil_h_d((float) (cols - 1) / 2), threadBlockBatchCount);
			}


			threadBlockBatchCountFloat = (float) reqThreadUnits3 / MAX_NO_OF_THREADBLOCKS;
			if (threadBlockBatchCountFloat <= 1)
			{
				processMapSampleCCLZeroIndexVertical2<<<reqThreadUnits3, 4>>>(d_map, d_ifLesserFound, rows, cols, ceil_h_d((float) cols / 2), 0);
			}
			else
			{
				threadBlockBatchCount = (unsigned int) threadBlockBatchCountFloat;
				extraThreadBlockCount = reqThreadUnits3 - threadBlockBatchCount * MAX_NO_OF_THREADBLOCKS;
				for (i = 0; i < threadBlockBatchCount; i++)
				{
					processMapSampleCCLZeroIndexVertical2<<<MAX_NO_OF_THREADBLOCKS, 4>>>(d_map, d_ifLesserFound, rows, cols, ceil_h_d((float) cols / 2), i);
				}
				processMapSampleCCLZeroIndexVertical2<<<extraThreadBlockCount, 4>>>(d_map, d_ifLesserFound, rows, cols, ceil_h_d((float) cols / 2), threadBlockBatchCount);
			}


			threadBlockBatchCountFloat = (float) reqThreadUnits4 / MAX_NO_OF_THREADBLOCKS;
			if (threadBlockBatchCountFloat <= 1)
			{
				processMapSampleCCLOneIndexVertical3<<<reqThreadUnits4, 4>>>(d_map, d_ifLesserFound, rows, cols, ceil_h_d((float) (cols - 1) / 2), 0);
			}
			else
			{
				threadBlockBatchCount = (unsigned int) threadBlockBatchCountFloat;
				extraThreadBlockCount = reqThreadUnits4 - threadBlockBatchCount * MAX_NO_OF_THREADBLOCKS;
				for (i = 0; i < threadBlockBatchCount; i++)
				{
					processMapSampleCCLOneIndexVertical3<<<MAX_NO_OF_THREADBLOCKS, 4>>>(d_map, d_ifLesserFound, rows, cols, ceil_h_d((float) (cols - 1) / 2), i);
				}
				processMapSampleCCLOneIndexVertical3<<<extraThreadBlockCount, 4>>>(d_map, d_ifLesserFound, rows, cols, ceil_h_d((float) (cols - 1) / 2), threadBlockBatchCount);

			}


			threadBlockBatchCountFloat = (float) reqThreadUnits1 / MAX_NO_OF_THREADBLOCKS;
			if (threadBlockBatchCountFloat <= 1)
			{
				processMapSampleCCLZeroIndexHorizontal0<<<reqThreadUnits1, 4>>>(d_map, d_ifLesserFound, rows, cols, ceil_h_d((float) cols / 2), 0);
			}
			else
			{
				threadBlockBatchCount = (unsigned int) threadBlockBatchCountFloat;
				extraThreadBlockCount = reqThreadUnits1 - threadBlockBatchCount * MAX_NO_OF_THREADBLOCKS;
				for (i = 0; i < threadBlockBatchCount; i++)
				{
					processMapSampleCCLZeroIndexHorizontal0<<<MAX_NO_OF_THREADBLOCKS, 4>>>(d_map, d_ifLesserFound, rows, cols, ceil_h_d((float) cols / 2), i);
				}
				processMapSampleCCLZeroIndexHorizontal0<<<extraThreadBlockCount, 4>>>(d_map, d_ifLesserFound, rows, cols, ceil_h_d((float) cols / 2), threadBlockBatchCount);
			}

			cudaMemcpy(h_ifLesserFound, d_ifLesserFound, 4 * sizeof(char), cudaMemcpyDeviceToHost);
		}
		//------------------------------------CONNECTED COMPONENT LABELLING-------------------------------//


		//------------------------------------CONNECTED COMPONENT COUNTING-------------------------------//
		h_map = (unsigned int *)malloc(currMapSizeBytes);
		cudaMemcpy(h_map, d_map, currMapSizeBytes, cudaMemcpyDeviceToHost);
		cudaFree(d_map);
		h_buff = (unsigned int *)malloc(currMapSizeBytes);
		memset(h_buff, 0, currMapSizeBytes);
		topIndex = 0;

		for (k = 0; k < currMapSize; k++)
		{
			tmp = 0;
			if (h_map[k] > 0)
			{
				for (l = 0; l <= topIndex; l++)
				{
					if (h_buff[l] == h_map[k])
					{
						tmp = 1;
						l = topIndex + 1;
					}
				}
				if (!tmp)
					h_buff[topIndex++] = h_map[k];
			}
		}
		free(h_map);
		free(h_buff);
		//------------------------------------CONNECTED COMPONENT COUNTING-------------------------------//

		printf("Map #%u: %u\n", mapCount + 1, topIndex);

		//*****************************************MAP PROCESSING****************************************//

		if (mapCount == mapCountStop)
			break;

		inputReadIndex += currMapSize;

		rows = h_input[inputReadIndex++];
		cols = h_input[inputReadIndex++];
		h_input_tmp = &h_input[inputReadIndex];

		currMapSize = rows * cols;
		currMapSizeBytes = currMapSize * sizeof(unsigned int);
		currMapSizeChar = currMapSize * sizeof(char);
		
		mapCount++;
	}
}
