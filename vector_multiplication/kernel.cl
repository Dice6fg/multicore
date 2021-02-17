__kernel void mat_mul(__global int* A, __global int* B, __global int* C,
					  const int A_COLSIZE, const int A_ROWSIZE, const int B_COLSIZE)
{
	int i = get_global_id(0); // i: 0 ~ A_ROWSIZE-1
	int j = get_global_id(1); // j: 0 ~ B_COLSIZE-1
	int temp;
	int k;

	temp = 0;
	
	for (k = 0; k < A_COLSIZE; k++)
	{
		temp = temp + A[i * A_COLSIZE + k] * B[k * B_COLSIZE + j];
	}
	C[B_COLSIZE * i + j] = temp;
}