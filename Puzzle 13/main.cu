#include "..\base.h"

#include <stdio.h>

#define COLUMN_SIZE 1024 * 1024
#define ROW_SIZE 32
#define ARRAY_SIZE ROW_SIZE * COLUMN_SIZE
static f32 InitialArray[ARRAY_SIZE] = {0};
static f32 ResultArray[ROW_SIZE] = {0};

static void Init() {
	random_state RandomState = { 0xB40148552A2E3491ULL };
	for (u32 i = 0; i < ARRAY_SIZE; ++i) {
		InitialArray[i] = RandomFloat(&RandomState) * 2.0f - 1.0f;
	}
}

#if 0
__global__ void AxisSum(f32 *In, uint2 ArrayDimensions, f32 *Out) {
	u32 X = blockIdx.x * blockDim.x + threadIdx.x;
	u32 Y = blockIdx.y * blockDim.y + threadIdx.y;

	f32 Result = 0.0f;
	for (u32 OffsetY = 0; OffsetY < ArrayDimensions.y; OffsetY += 1) {
		u32 GlobalIndex = (Y + OffsetY) * ArrayDimensions.x + X;
		Result += In[GlobalIndex];
	}
	Out[threadIdx.x] = Result;
}
#else
__global__ void AxisSum(f32 *In, uint2 ArrayDimensions, f32 *Out) {

	__shared__ f32 SharedValues[1024];

	u32 X = blockIdx.x * blockDim.x + threadIdx.x;
	u32 Y = blockIdx.y * blockDim.y + threadIdx.y;

	f32 Result = 0.0f;
	for (u32 OffsetY = 0; OffsetY < ArrayDimensions.y; OffsetY += 32) {
		u32 GlobalIndex = (Y + OffsetY) * ArrayDimensions.x + X;
		Result += In[GlobalIndex];
	}

	SharedValues[threadIdx.y * 32 + threadIdx.x] = Result;
	__syncthreads();

	if (threadIdx.x < 32 && threadIdx.y == 0) {
		f32 FinalResult = 0.0f;
		for (u32 OffsetY = 0; OffsetY < 1024; OffsetY += 32) {
			FinalResult += SharedValues[OffsetY + threadIdx.x];
		}
		Out[threadIdx.x] = FinalResult;
	}
}
#endif

s32 main() {
	Init();

	f32 *GPUArray1 = 0, *GPUArray2 = 0;
	cudaMalloc(&GPUArray1, sizeof(InitialArray));
	cudaMalloc(&GPUArray2, sizeof(ResultArray));
	cudaMemcpy(GPUArray1, InitialArray, sizeof(InitialArray), cudaMemcpyHostToDevice);

	dim3 ThreadDimensions(32, 32);
	AxisSum<<<1, ThreadDimensions>>>(GPUArray1, make_uint2(ROW_SIZE, COLUMN_SIZE), GPUArray2);
	// AxisSum<<<1, 32>>>(GPUArray1, make_uint2(ROW_SIZE, COLUMN_SIZE), GPUArray2);
	cudaMemcpy(ResultArray, GPUArray2, sizeof(ResultArray), cudaMemcpyDeviceToHost);

#if 1
	f32 ExpectedResults[ROW_SIZE] = {0};
	for (u32 Y = 0; Y < COLUMN_SIZE; Y += 1) {
		for (u32 X = 0; X < ROW_SIZE; X += 1) {
			ExpectedResults[X] += InitialArray[Y * ROW_SIZE + X];
		}
	}
	for (u32 i = 0; i < ROW_SIZE; ++i) {
		f32 Expected = ExpectedResults[i];
		f32 Actual = ResultArray[i];
		if (Expected == Actual) {
			printf(ANSI_COLOR_GREEN "Expected: %.2f | Actual: %.2f\n", Expected, Actual);
		} else {
			printf(ANSI_COLOR_RED "Expected: %.2f | Actual: %.2f\n", Expected, Actual);
		}
	}
#endif
	printf(ANSI_COLOR_RESET);
}
