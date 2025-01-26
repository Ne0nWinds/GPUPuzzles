
#include "..\base.h"

#include <stdio.h>

static random_state RandomState = { 0xB40148552A2E3491 };
#define ARRAY_SIZE 16
static f32 InitialArray1[32] = {0};
static f32 InitialArray2[32] = {0};
static f32 ResultArray[32] = {0};

static void InitRandomIntegers() {
	for (u32 i = 0; i < ARRAY_SIZE; ++i) {
		InitialArray1[i] = i;
	}
	for (u32 i = 0; i < 4; ++i) {
		InitialArray2[i] = i;
	}
}

__global__ void Dot(f32 *In1, f32 *In2, f32 *Out, u32 Length) {
	u32 LocalIndex = threadIdx.x;

#if 0
	f32 Result = 0.0f;
	for (u32 j = 0; j < ARRAY_LEN(InitialArray2); ++j) {
		if (LocalIndex + j >= ARRAY_SIZE) break;
		f32 A = In1[LocalIndex + j];
		f32 B = In2[j];
		Result = fma(A, B, Result);
	}
#else

	f32 Result = 0.0f;
	f32 InputA = In1[LocalIndex];
	f32 InputB = In2[LocalIndex];
	__syncthreads();
	{
		f32 A = InputA;
		f32 B = __shfl_sync(0xFFFF'FFFF, InputB, 0);
		Result = A * B;
	}
	for (u32 i = 1; i < 4; ++i) {
		f32 A = __shfl_down_sync(0xFFFF'FFFF, InputA, i);
		f32 B = __shfl_sync(0xFFFF'FFFF, InputB, i);
		Result = fma(A, B, Result);
	}
#endif
	Out[LocalIndex] = Result;
}

s32 main() {
	InitRandomIntegers();
	puts("===");

	f32 *GPUArray1 = 0, *GPUArray2 = 0, *GPUArray3 = 0;
	cudaMalloc(&GPUArray1, sizeof(InitialArray1));
	cudaMalloc(&GPUArray2, sizeof(InitialArray2));
	cudaMalloc(&GPUArray3, sizeof(ResultArray));
	cudaMemset(GPUArray3, 0, sizeof(f32));
	cudaMemcpy(GPUArray1, InitialArray1, sizeof(InitialArray1), cudaMemcpyHostToDevice);
	cudaMemcpy(GPUArray2, InitialArray2, sizeof(InitialArray2), cudaMemcpyHostToDevice);

	dim3 threadDimension(ARRAY_SIZE, 1);
	dim3 blockDimension(1, 1);
	Dot<<<blockDimension, threadDimension>>>(GPUArray1, GPUArray2, GPUArray3, ARRAY_SIZE);
	cudaMemcpy(ResultArray, GPUArray3, sizeof(ResultArray), cudaMemcpyDeviceToHost);

	puts("===");

#if 1
	for (u32 i = 0; i < ARRAY_SIZE; i += 1) {
		f32 ExpectedResult = 0.0f;
		for (u32 j = 0; j < ARRAY_LEN(InitialArray2); ++j) {
			if (i + j > ARRAY_SIZE) break;
			ExpectedResult += InitialArray2[j] * InitialArray1[i + j];
		}
		f32 ActualResult = ResultArray[i];
		printf("Expected: %.2f -- Actual: %.2f\n", ExpectedResult, ActualResult);
	}
#endif
	printf(ANSI_COLOR_RESET);
}
