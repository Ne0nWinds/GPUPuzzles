#include <stdio.h>
#include <stdint.h>

typedef int32_t s32;

__global__ void add(s32 *a, s32 *b, s32 *c, s32 n) {
	s32 index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < n) {
		c[index] = a[index] + b[index];
	}
}

int main() {
	s32 n = 128;
	s32 size = n * sizeof(int);
	s32 *a = (s32 *)malloc(size);
	s32 *b = (s32 *)malloc(size);
	s32 *c = (s32 *)malloc(size);

	for (s32 i = 0; i < n; ++i) {
		a[i] = i;
		b[i] = i * 2;
	}

	s32 *d_a, *d_b, *d_c;
	cudaMalloc(&d_a, size);
	cudaMalloc(&d_b, size);
	cudaMalloc(&d_c, size);

	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

	add<<<(n+255) / 256, 256>>>(d_a, d_b, d_c, n);

	cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

	for (s32 i = 0; i < 128; ++i) {
		printf("%d + %d = %d\n", a[i], b[i], c[i]);
	}
}
