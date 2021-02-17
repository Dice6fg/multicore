#define ReLU(x) (((x)>0)?(x):0)

void conv(__global float* input, __global float* output, __global float* filter, __global float* biases, int D2, int D1, int N, int type, __local float* lFilter1, __local float* lFilter2)
{
		int i = get_global_id(1);	//COL		0 ~ (N-1)
		int j = get_global_id(0);	//LOW		0 ~ (N-1)
		int c = get_global_id(2);		

		int l_i = get_local_id(0);
		int l_j = get_local_id(1);
		int l_c = get_local_id(2);
		int l_size = get_local_size(1);

		int image_size = N*N;
		int	t1 = (c/3000)*D1*9, t2 = (c%3000)*image_size*D1;
		

		float sum = 0.0;

		for (int m = 0; m < D1; m++)
		{
			if(m%2 == 0)
			{
				if(l_size <= 2 && l_c < 9) 
					lFilter1[l_c] = filter[l_c + m*9 + t1];
				else if(l_size > 2 && (l_i*3 + l_j) < 9)
					lFilter1[l_i*3 + l_j] = filter[l_i*3+l_j + m*9 + t1];
			} else
			{
				if(l_size <= 2 && l_c < 9)
					lFilter2[l_c] = filter[l_c + m*9 + t1];
				else if(l_size > 2 && (l_i*3 + l_j) < 9)
					lFilter2[l_i*3 + l_j] = filter[l_i*3+l_j + m*9 + t1];
			}
			barrier(CLK_LOCAL_MEM_FENCE);

			for (int k = 0; k < 3; k++)
			{
				for (int l = 0; l < 3; l++)
				{
					int x = i + k - 1;
					int y = j + l - 1;
					if (x >= 0 && x < N && y >= 0 && y < N)
					{	
						if(m%2 == 0)
							sum += input[x * N + y + m*image_size + t2]* lFilter1[k * 3 + l];
						else
							sum += input[x * N + y + m*image_size + t2]* lFilter2[k * 3 + l];
					}
				}
			}
		}
		output[i * N + j + (c/3000)*image_size + (c%3000)*D2*image_size] = ReLU(sum + biases[c/3000]);	
}

void pool(__global float* input, __global float* output, __global float* filter, __global float* biases, int D2, int D1, int N, int type)
{
	int i = get_global_id(1);	//COL		0 ~ (N-1)
	int j = get_global_id(0);	//LOW		0 ~ (N-1)
	int c = get_global_id(2);	//channel	0 ~ (D-1)
	float max = 0.0;

	int image_size = N*N;
	int t1 = 4*image_size*c;
	int t2 = 2*N;

	float pixel = input[t1 + t2 * (i * 2) + j * 2];
	max = (max > pixel) ? max : pixel;
	pixel = input[t1 + t2 * (i * 2) + j * 2 + 1];
	max = (max > pixel) ? max : pixel;
	pixel = input[t1 + t2 * (i * 2 + 1) + j * 2];
	max = (max > pixel) ? max : pixel;
	pixel = input[t1 + t2 * (i * 2 + 1) + j * 2 + 1];
	max = (max > pixel) ? max : pixel;

	output[i * N + j + c * image_size] = max;
}

void fc(__global float* input_neuron, __global float* output_neuron, __global float* weights, __global float* biases, int M, int N, __local float* lWeight)
{
	int j = get_global_id(0);			//0 ~ (M - 1) (0,0) (1,0) (M-1, 0) 512번 동일한 input
	int k = get_global_id(1);			//0 ~ (num_images)

	int l_j = get_local_id(0);
	int l_k = get_local_id(1);			//0 ~ 250-1
	int i = 0;
	float sum = 0.0;

	if(l_k < 250)
	{
		lWeight[l_k] = weights[l_k + j*N];
		lWeight[l_k+250] = weights[(l_k+250) + j*N];
		if(l_k < 12)
		{
			lWeight[l_k + 500] = weights[l_k+500 + j*N];
		}
	}

	barrier(CLK_LOCAL_MEM_FENCE);
	
	for (i = 0; i < N; i++) 
	{
		sum += input_neuron[i + k*N] * lWeight[i];
	}
	output_neuron[j + k*M] = ReLU(sum + biases[j]);
}

__kernel void convolution(__global float* input, __global float* output, __global float* filter, __global float* biases, int D2, int D1, int N, int type) 
{
	__local float lFilter1[9];
	__local float lFilter2[9];
	__local float lWeight[512];

	if (type == 0)
		conv(input, output, filter, biases, D2, D1, N, type, lFilter1, lFilter2);
	else if(type == 1)
		pool(input, output, filter, biases, D2, D1, N, type);
	else if(type == 2)
		fc(input, output, filter, biases, D2, D1, lWeight);
}
