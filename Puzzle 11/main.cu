
#include "..\base.h"

#include <stdio.h>

#define SIZE_A (32 * 8)
#define SIZE_B (32 * 4)
static f32 ArrayA[SIZE_A * 2] = {0};
static f32 ArrayB[SIZE_B * 2] = {0};
static f32 ResultArray[SIZE_A] = {0};

static void Init() {
	random_state RandomState = { 0x2ED3368E4B108C0CULL };
	for (u32 i = 0; i < SIZE_A; ++i) {
		// ArrayA[i] = RandomFloat(&RandomState);
		ArrayA[i] = i;
	}
	for (u32 i = 0; i < SIZE_B; ++i) {
		ArrayB[i] = i / 16.0;
	}
}

#if 0
__global__ void Convolution(f32 *A, u32 LengthA, f32 *B, u32 LengthB, f32 *Out) {

	u32 i = blockIdx.x * blockDim.x + threadIdx.x;

	f32 Result = 0.0f;
	for (u32 j = 0; j < LengthB; ++j) {
		if (i + j >= LengthA) break;
		f32 ValueA = A[i + j];
		f32 ValueB = B[j];
		Result = fmaf(ValueA, ValueB, Result);
	}
	Out[i] = Result;
}
#elif 0
// Thread-level
__global__ void Convolution(f32 *A, u32 LengthA, f32 *B, u32 LengthB, f32 *Out) {
	u32 i = blockIdx.x * blockDim.x + threadIdx.x;

	f32 RegisterA = A[i];
	f32 RegisterB = B[i];
	f32 Result = RegisterA * __shfl_sync(0xFFFFFFFF, RegisterB, 0);

	for (u32 j = 1; j < LengthB; ++j) {
		f32 ShiftedRegisterA = __shfl_down_sync(0xFFFFFFFF, RegisterA, j);
		f32 ShiftedRegisterB = __shfl_sync(0xFFFFFFFF, RegisterB, j);
		if (i + j < LengthA) {
			Result = fmaf(ShiftRegisterA, ShiftRegisterB, Result);
		}
	}

	Out[i] = Result;
}
#elif 0
__global__ void Convolution(f32 *A, u32 LengthA, f32 *B, u32 LengthB, f32 *Out) {

	f32 Result = 0.0f;
	u32 i = blockIdx.x * blockDim.x + threadIdx.x;

	__shared__ f32 SharedB[32];

	for (u32 j = 0; j < LengthB; j += 32) {
		SharedB[LaneIndex] = B[j + threadIdx.x];
		__syncthreads();
		for (u32 s = 0; s < 32; ++s) {
			if (i + j + s >= LengthA) break;
			f32 RegisterA = A[i + j + s];
			Result = fmaf(RegisterA, SharedB[s], Result);
		}
		__syncthreads();
	}

	Out[i] = Result;
}
#elif 0
__global__ void Convolution(f32 *A, u32 LengthA, f32 *B, u32 LengthB, f32 *Out) {
	u32 i = blockIdx.x * blockDim.x + threadIdx.x;

	__shared__ f32 SharedB[32];

	f32 Result = 0.0f;
	for (u32 j = 0; j < LengthB; j += 32) {
		SharedB[LaneIndex] = B[j + threadIdx.x];
		__syncthreads();
		for (u32 s = 0; s < 32; ++s) {
			if (i + j + s >= LengthA) break;
			f32 ValueA = A[i + j + s];
			f32 ValueB = SharedB[s];
			Result = fmaf(ValueA, ValueB, Result);
		}
		__syncthreads();
	}

	Out[i] = Result;
}
#else
__global__ void Convolution(f32 *A, u32 LengthA, f32 *B, u32 LengthB, f32 *Out) {

	u32 i = blockIdx.x * blockDim.x + threadIdx.x;

	__shared__ f32 SharedA[64];
	__shared__ f32 SharedB[32];

	f32 Result = 0.0f;
	for (u32 j = 0; j < LengthB; j += 32) {
		SharedB[threadIdx.x] = B[j + threadIdx.x];
		SharedA[threadIdx.x] = A[i + j];
		SharedA[threadIdx.x + 32] = A[i + j + 32];
		__syncthreads();
		for (u32 s = 0; s < 32; ++s) {
			if (i + j + s >= LengthA) break;
			f32 ValueA = SharedA[threadIdx.x + s];
			f32 ValueB = SharedB[s];
			Result = fmaf(ValueA, ValueB, Result);
		}
		__syncthreads();
	}

	Out[i] = Result;
}
#endif

s32 main() {
	Init();

	f32 *GPUArray1 = 0, *GPUArray2 = 0, *GPUArray3 = 0;
	cudaMalloc(&GPUArray1, sizeof(ArrayA));
	cudaMalloc(&GPUArray2, sizeof(ArrayB));
	cudaMalloc(&GPUArray3, sizeof(ArrayA));
	cudaMemset(GPUArray3, 0, sizeof(ArrayA));
	cudaMemcpy(GPUArray1, ArrayA, sizeof(ArrayA), cudaMemcpyHostToDevice);
	cudaMemcpy(GPUArray2, ArrayB, sizeof(ArrayB), cudaMemcpyHostToDevice);

	u32 ThreadCount = 32;
	u32 BlockCount = SIZE_A / 32;

	Convolution<<<BlockCount, ThreadCount>>>(GPUArray1, SIZE_A, GPUArray2, SIZE_B, GPUArray3);
	cudaMemcpy(ResultArray, GPUArray3, sizeof(ResultArray), cudaMemcpyDeviceToHost);

#if 1
	for (u32 i = 0; i < SIZE_A; i += 1) {
		f32 ExpectedResult = 0.0f;
		for (u32 j = 0; j < SIZE_B; ++j) {
			if (i + j >= SIZE_A) break;
			ExpectedResult += ArrayB[j] * ArrayA[i + j];
		}
		if (i % 32 == 0) {
			printf(ANSI_COLOR_RESET "Warp %d\n", i / 32);
		}
		f32 ActualResult = ResultArray[i];
		if (ExpectedResult == ActualResult) {
			printf(ANSI_COLOR_GREEN "Expected: %.2f | Actual: %.2f\n", ExpectedResult, ActualResult);
		} else {
			printf(ANSI_COLOR_RED "Expected: %.2f | Actual: %.2f\n", ExpectedResult, ActualResult);
		}
	}
#endif
	printf(ANSI_COLOR_RESET);
}
