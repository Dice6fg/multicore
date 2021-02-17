#define CL_TARGET_OPENCL_VERSION 120

#include <CL/cl.h>
#include "cnn.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstdlib>
#include <time.h>
#include <math.h>

#define OUT	
#define CONV_MEM_OBJ_NUM	4
#define POOL_MEM_OBJ_NUM	2
#define FC_MEM_OBJ_NUM		4

#define ReLU(x) (((x)>0)?(x):0)

unsigned int WORK_GROUP_SIZE = 256;
//unsigned int GLOBAL_SIZE = ;

cl_context context;
cl_mem conv_memObjects[CONV_MEM_OBJ_NUM], pool_memObjects[POOL_MEM_OBJ_NUM], fc_memObjects[FC_MEM_OBJ_NUM];
cl_device_id device;
cl_command_queue commandQueue;
cl_program conv_program, pool_program, fc_program;
cl_kernel conv_kernel, pool_kernel, fc_kernel;
cl_int errNum;

cl_context CreateContext();
cl_command_queue CreateCommandQueue(cl_context context, cl_device_id* device);
cl_program CreateProgram(cl_context context, cl_device_id device, const char* fileName);
void Cleanup(cl_context context, cl_command_queue commandQueue,
	cl_program program, cl_kernel kernel, cl_mem* memObjects, int mem_obj_num);

void cnn_init()
 {
	context = CreateContext();
	for (int i = 0; i < CONV_MEM_OBJ_NUM; i++) conv_memObjects[i] = 0;
	for (int i = 0; i < POOL_MEM_OBJ_NUM; i++) pool_memObjects[i] = 0;
	for (int i = 0; i < FC_MEM_OBJ_NUM; i++) fc_memObjects[i] = 0;

	commandQueue = CreateCommandQueue(context, OUT & device);

	//Ŀ�� �ҽ��κ��� OpenCL ���α׷��� �����Ѵ�.
	//convolution program and kernel
	conv_program = CreateProgram(context, device, "conv_kernel.cl");
	conv_kernel = clCreateKernel(conv_program, "convolution", NULL);
	if (conv_kernel == NULL)
	{
		std::cerr << "Failed to create conv_kernel\n";
		return;
	}
}

float* alloc_layer(size_t n, int num_images)
{
	return (float*)malloc(n * num_images * sizeof(float));
}

static void softmax(float* output, int N) 
{
	int i;
	float max = output[0];
	for (i = 1; i < N; i++) 
	{
		max = (output[i] > max) ? output[i] : max;
	}
	float sum = 0;
	for (i = 0; i < N; i++) 
	{
		sum += exp(output[i] - max);
	}
	for (i = 0; i < N; i++) 
	{
		output[i] = exp(output[i] - max) / sum;
	}
}

int find_max(float* fc, int N) 
{
	int i;
	int maxid = 0;
	float maxval = 0;
	for (i = 0; i < N; i++) 
	{
		if (maxval < fc[i]) 
		{
			maxval = fc[i];
			maxid = i;
		}
	}
	return maxid;
}

void cnn(float* images, float** network, int* labels, float* confidences, int num_images)
{
	float* w1_1, * b1_1, * w1_2, * b1_2;
	float* w2_1, * b2_1, * w2_2, * b2_2;
	float* w3_1, * b3_1, * w3_2, * b3_2, * w3_3, * b3_3;
	float* w4_1, * b4_1, * w4_2, * b4_2, * w4_3, * b4_3;
	float* w5_1, * b5_1, * w5_2, * b5_2, * w5_3, * b5_3;
	float* w1, * b1, * w2, * b2, * w3, * b3;
	w1_1 = network[0]; b1_1 = network[1];
	w1_2 = network[2]; b1_2 = network[3];
	w2_1 = network[4]; b2_1 = network[5];
	w2_2 = network[6]; b2_2 = network[7];
	w3_1 = network[8]; b3_1 = network[9];
	w3_2 = network[10]; b3_2 = network[11];
	w3_3 = network[12]; b3_3 = network[13];
	w4_1 = network[14]; b4_1 = network[15];
	w4_2 = network[16]; b4_2 = network[17];
	w4_3 = network[18]; b4_3 = network[19];
	w5_1 = network[20]; b5_1 = network[21];
	w5_2 = network[22]; b5_2 = network[23];
	w5_3 = network[24]; b5_3 = network[25];
	w1 = network[26]; b1 = network[27];
	w2 = network[28]; b2 = network[29];
	w3 = network[30]; b3 = network[31];

	float* c1_1, * c1_2, * p1;
	float* c2_1, * c2_2, * p2;
	float* c3_1, * c3_2, * c3_3, * p3;
	float* c4_1, * c4_2, * c4_3, * p4;
	float* c5_1, * c5_2, * c5_3, * p5;
	float* fc1, * fc2, * fc3;
	float* filter;

	c1_1 = alloc_layer(64 * 32 * 32, num_images);
	c1_2 = alloc_layer(64 * 32 * 32, num_images);
	p1 = alloc_layer(64 * 16 * 16, num_images);
	c2_1 = alloc_layer(128 * 16 * 16, num_images);
	c2_2 = alloc_layer(128 * 16 * 16, num_images);
	p2 = alloc_layer(128 * 8 * 8, num_images);
	c3_1 = alloc_layer(256 * 8 * 8, num_images);
	c3_2 = alloc_layer(256 * 8 * 8, num_images);
	c3_3 = alloc_layer(256 * 8 * 8, num_images);
	p3 = alloc_layer(256 * 4 * 4, num_images);
	c4_1 = alloc_layer(512 * 4 * 4, num_images);
	c4_2 = alloc_layer(512 * 4 * 4, num_images);
	c4_3 = alloc_layer(512 * 4 * 4, num_images);
	p4 = alloc_layer(512 * 2 * 2, num_images);
	c5_1 = alloc_layer(512 * 2 * 2, num_images);
	c5_2 = alloc_layer(512 * 2 * 2, num_images);
	c5_3 = alloc_layer(512 * 2 * 2, num_images);
	p5 = alloc_layer(512 * 1 * 1, num_images);  
	fc1 = alloc_layer(512, num_images);
	fc2 = alloc_layer(512, num_images);
	fc3 = alloc_layer(10, num_images);

	conv_memObjects[0] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 32 * 32 * 64 * num_images, NULL, NULL);		// g_input
	conv_memObjects[1] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 32 * 32 * 64 * num_images, NULL, NULL);		// g_output
	conv_memObjects[2] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * 3 * 3 * 512 * 512, NULL, NULL);	// g_filter
	conv_memObjects[3] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * 512, NULL, NULL);				// g_biases

	//first image
	errNum = clEnqueueWriteBuffer(commandQueue, conv_memObjects[0], CL_TRUE, 0,
		sizeof(float) * 32*32*3 * num_images, images, 0, NULL, NULL);

	size_t globalWork[3];
	size_t localWork[3];
	cl_int N, D1, D2, type;

	//////////////////////////////////////////////////////////c1_1///////////////////////////// 
	N = 32; D1 = 3; D2 = 64; type = 0;
	globalWork[0] = N;
	globalWork[1] = N;
	globalWork[2] = D2 * num_images;
	localWork[0] = 16;
	localWork[1] = 16;
	localWork[2] = 1;

	//set kernel arguments
	errNum = clSetKernelArg(conv_kernel, 0, sizeof(cl_mem), &conv_memObjects[0]);		//g_input
	errNum |= clSetKernelArg(conv_kernel, 1, sizeof(cl_mem), &conv_memObjects[1]);		//g_output
	errNum |= clSetKernelArg(conv_kernel, 2, sizeof(cl_mem), &conv_memObjects[2]);		//g_filter
	errNum |= clSetKernelArg(conv_kernel, 3, sizeof(cl_mem), &conv_memObjects[3]);		//g_bias
	errNum |= clSetKernelArg(conv_kernel, 4, sizeof(cl_int), &D2);						//D2
	errNum |= clSetKernelArg(conv_kernel, 5, sizeof(cl_int), &D1);						//D1
	errNum |= clSetKernelArg(conv_kernel, 6, sizeof(cl_int), &N);						//N
	errNum |= clSetKernelArg(conv_kernel, 7, sizeof(cl_int), &type);					//type

	//filter
	errNum = clEnqueueWriteBuffer(commandQueue, conv_memObjects[2], CL_TRUE, 0,
	sizeof(float) * 3 * 3 * D1 * D2, w1_1, 0, NULL, NULL);
	//biases
	errNum |= clEnqueueWriteBuffer(commandQueue, conv_memObjects[3], CL_TRUE, 0,
		sizeof(float) * D2, b1_1, 0, NULL, NULL);
	
	//execute kernel
	errNum = clEnqueueNDRangeKernel(commandQueue, conv_kernel, 3, NULL,
		globalWork, localWork, 0, NULL, NULL);


	//test read buffer
	/*errNum = clEnqueueReadBuffer(commandQueue, conv_memObjects[1], CL_TRUE, 0,
		sizeof(float) * N * N * D2 * num_images, c1_1, 0, NULL, NULL);
	if (errNum != CL_SUCCESS)
	{
		std::cerr << "Error reading." << std::endl;
		Cleanup(context, commandQueue, conv_program, conv_kernel, conv_memObjects, CONV_MEM_OBJ_NUM);
		return;
	}
	for (int i = 0; i < 32; i++)
		printf("c1_1 %d: %f\n", i, c1_1[i]);*/

	//////////////////////////////////////////////////////////c1_2/////////////////////////////
	D1 = 64;

	//set kernel arguments
	errNum = clSetKernelArg(conv_kernel, 0, sizeof(cl_mem), &conv_memObjects[1]);		//g_input
	errNum |= clSetKernelArg(conv_kernel, 1, sizeof(cl_mem), &conv_memObjects[0]);		//g_output
	errNum |= clSetKernelArg(conv_kernel, 5, sizeof(cl_int), &D1);						//D1

	//write buffer
	//filter
	errNum = clEnqueueWriteBuffer(commandQueue, conv_memObjects[2], CL_TRUE, 0,
		sizeof(float) * 3 * 3 * D1 * D2, w1_2, 0, NULL, NULL);
	//biases
	errNum |= clEnqueueWriteBuffer(commandQueue, conv_memObjects[3], CL_TRUE, 0,
		sizeof(float) * D2, b1_2, 0, NULL, NULL);

	//execute kernel
	errNum = clEnqueueNDRangeKernel(commandQueue, conv_kernel, 3, NULL,
		globalWork, localWork, 0, NULL, NULL);


	/////////////////////////////////////////////////////p1//////////////////////////
	N = 16; type = 1;
	globalWork[0] = N;
	globalWork[1] = N;

	//set kernel arguments
	errNum = clSetKernelArg(conv_kernel, 0, sizeof(cl_mem), &conv_memObjects[0]);		//g_input
	errNum |= clSetKernelArg(conv_kernel, 1, sizeof(cl_mem), &conv_memObjects[1]);		//g_output
	errNum |= clSetKernelArg(conv_kernel, 6, sizeof(cl_int), &N);						//N
	errNum |= clSetKernelArg(conv_kernel, 7, sizeof(cl_int), &type);					//type

	//execute kernel
	errNum = clEnqueueNDRangeKernel(commandQueue, conv_kernel, 3, NULL,
		globalWork, localWork, 0, NULL, NULL);

	//test read buffer
	/*errNum = clEnqueueReadBuffer(commandQueue, conv_memObjects[1], CL_TRUE, 0,
		sizeof(float) * N * N * D2, p1, 0, NULL, NULL);

	for (int i = 0; i < 32; i++)
		printf("p1 %d: %f\n", i, p1[i]);*/

	///////////////////////////////////////////////////////c2_1/////////////////////////////////////////
	D2 = 128; type = 0;
	globalWork[2] = D2 * num_images;

	//set kernel arguments
	errNum = clSetKernelArg(conv_kernel, 0, sizeof(cl_mem), &conv_memObjects[1]);		//g_input
	errNum |= clSetKernelArg(conv_kernel, 1, sizeof(cl_mem), &conv_memObjects[0]);		//g_output
	errNum |= clSetKernelArg(conv_kernel, 4, sizeof(cl_int), &D2);						//D2
	errNum |= clSetKernelArg(conv_kernel, 7, sizeof(cl_int), &type);					//type

	//write buffer
	//filter
	errNum = clEnqueueWriteBuffer(commandQueue, conv_memObjects[2], CL_TRUE, 0,
		sizeof(float) * 3 * 3 * D1 * D2, w2_1, 0, NULL, NULL);
	//biases
	errNum |= clEnqueueWriteBuffer(commandQueue, conv_memObjects[3], CL_TRUE, 0,
		sizeof(float) * D2, b2_1, 0, NULL, NULL);


	//execute kernel
	errNum = clEnqueueNDRangeKernel(commandQueue, conv_kernel, 3, NULL,
		globalWork, localWork, 0, NULL, NULL);

	//test read buffer
	/*errNum = clEnqueueReadBuffer(commandQueue, conv_memObjects[0], CL_TRUE, 0,
		sizeof(float) * N * N * D2, c2_1, 0, NULL, NULL);

	for (int i = 0; i < 32; i++)
		printf("c2_1 %d: %f\n", i, c2_1[i]);*/


	//////////////////////////////////////////////////////c2_2//////////////////////////////////
	D1 = 128;
	
	//set kernel arguments
	errNum = clSetKernelArg(conv_kernel, 0, sizeof(cl_mem), &conv_memObjects[0]);		//g_input
	errNum |= clSetKernelArg(conv_kernel, 1, sizeof(cl_mem), &conv_memObjects[1]);		//g_output
	errNum |= clSetKernelArg(conv_kernel, 5, sizeof(cl_int), &D1);						//D1

	//write buffer
	//filter
	errNum = clEnqueueWriteBuffer(commandQueue, conv_memObjects[2], CL_TRUE, 0,
		sizeof(float) * 3 * 3 * D1 * D2, w2_2, 0, NULL, NULL);
	//biases
	errNum = clEnqueueWriteBuffer(commandQueue, conv_memObjects[3], CL_TRUE, 0,
		sizeof(float) * D2, b2_2, 0, NULL, NULL);

	//execute kernel
	errNum = clEnqueueNDRangeKernel(commandQueue, conv_kernel, 3, NULL,
		globalWork, localWork, 0, NULL, NULL);

	//test read buffer
	/*errNum = clEnqueueReadBuffer(commandQueue, conv_memObjects[1], CL_TRUE, 0,
		sizeof(float) * N * N * D2, c2_2, 0, NULL, NULL);

	for (int i = 0; i < 32; i++)
		printf("c2_2 %d: %f\n", i, c2_2[i]);*/
	////////////////////////////////////////////////////////p2///////////////////////////////
	N = 8; type = 1;
	globalWork[0] = N;
	globalWork[1] = N;
	localWork[0] = 8;
	localWork[1] = 8;
	localWork[2] = 4;

	//set kernel arguments
	errNum = clSetKernelArg(conv_kernel, 0, sizeof(cl_mem), &conv_memObjects[1]);		//g_input
	errNum |= clSetKernelArg(conv_kernel, 1, sizeof(cl_mem), &conv_memObjects[0]);		//g_output
	errNum |= clSetKernelArg(conv_kernel, 6, sizeof(cl_int), &N);						//N
	errNum |= clSetKernelArg(conv_kernel, 7, sizeof(cl_int), &type);					//type

	//execute kernel
	errNum = clEnqueueNDRangeKernel(commandQueue, conv_kernel, 3, NULL,
		globalWork, localWork, 0, NULL, NULL);

	//test read buffer
	/*errNum = clEnqueueReadBuffer(commandQueue, conv_memObjects[0], CL_TRUE, 0,
		sizeof(float) * N * N * D2, p2, 0, NULL, NULL);

	for (int i = 0; i < 32; i++)
		printf("p2 %d: %f\n", i, p2[i]);*/

	////////////////////////////////////////////////////////c3_1///////////////////////////////
	D2 =256; type = 0;
	globalWork[2] = D2 * num_images;

	//set kernel arguments
	errNum = clSetKernelArg(conv_kernel, 0, sizeof(cl_mem), &conv_memObjects[0]);		//g_input
	errNum |= clSetKernelArg(conv_kernel, 1, sizeof(cl_mem), &conv_memObjects[1]);		//g_output
	errNum |= clSetKernelArg(conv_kernel, 4, sizeof(cl_int), &D2);						//D2
	errNum |= clSetKernelArg(conv_kernel, 7, sizeof(cl_int), &type);					//type

	//write buffer
	//filter
	errNum = clEnqueueWriteBuffer(commandQueue, conv_memObjects[2], CL_TRUE, 0,
		sizeof(float) * 3 * 3 * D1 * D2, w3_1, 0, NULL, NULL);
	//biases
	errNum |= clEnqueueWriteBuffer(commandQueue, conv_memObjects[3], CL_TRUE, 0,
		sizeof(float) * D2, b3_1, 0, NULL, NULL);

	//execute kernel
	errNum = clEnqueueNDRangeKernel(commandQueue, conv_kernel, 3, NULL,
		globalWork, localWork, 0, NULL, NULL);

	//test read buffer
	/*errNum = clEnqueueReadBuffer(commandQueue, conv_memObjects[1], CL_TRUE, 0,
		sizeof(float) * N * N * D2 *num_images, c3_1, 0, NULL, NULL);

	for (int i = 0; i < 64; i++)
		printf("c3_1 %d: %f\n", i, c3_1[i]);*/

	//////////////////////////////////////////////////////////////////////////c3_2
	D1 = 256;

	//set kernel arguments
	errNum = clSetKernelArg(conv_kernel, 0, sizeof(cl_mem), &conv_memObjects[1]);		//g_input
	errNum |= clSetKernelArg(conv_kernel, 1, sizeof(cl_mem), &conv_memObjects[0]);		//g_output
	errNum |= clSetKernelArg(conv_kernel, 5, sizeof(cl_int), &D1);						//D1

	//write buffer
	//filter
	errNum = clEnqueueWriteBuffer(commandQueue, conv_memObjects[2], CL_TRUE, 0,
		sizeof(float) * 3 * 3 * D1 * D2, w3_2, 0, NULL, NULL);
	//biases
	errNum |= clEnqueueWriteBuffer(commandQueue, conv_memObjects[3], CL_TRUE, 0,
		sizeof(float) * D2, b3_2, 0, NULL, NULL);

	//execute kernel
	errNum = clEnqueueNDRangeKernel(commandQueue, conv_kernel, 3, NULL,
		globalWork, localWork, 0, NULL, NULL);


	//test read buffer
	/*errNum = clEnqueueReadBuffer(commandQueue, conv_memObjects[0], CL_TRUE, 0,
		sizeof(float) * N * N * D2, c3_2, 0, NULL, NULL);

	for (int i = 0; i < 32; i++)
		printf("c3_2 %d: %f\n", i, c3_2[i]);*/

	////////////////////////////////////////////////////////c3_3////////////////////

	//set kernel arguments
	errNum = clSetKernelArg(conv_kernel, 0, sizeof(cl_mem), &conv_memObjects[0]);		//g_input
	errNum |= clSetKernelArg(conv_kernel, 1, sizeof(cl_mem), &conv_memObjects[1]);		//g_output

	//write buffer
	//filter
	errNum = clEnqueueWriteBuffer(commandQueue, conv_memObjects[2], CL_TRUE, 0,
		sizeof(float) * 3 * 3 * D1 * D2, w3_3, 0, NULL, NULL);
	//biases
	errNum |= clEnqueueWriteBuffer(commandQueue, conv_memObjects[3], CL_TRUE, 0,
		sizeof(float) * D2, b3_3, 0, NULL, NULL);

	//execute kernel
	errNum = clEnqueueNDRangeKernel(commandQueue, conv_kernel, 3, NULL,
		globalWork, localWork, 0, NULL, NULL);

	//test read buffer
	/*errNum = clEnqueueReadBuffer(commandQueue, conv_memObjects[1], CL_TRUE, 0,
		sizeof(float) * N * N * D2 , c3_3, 0, NULL, NULL);

	for (int i = 0; i < 32; i++)
		printf("c3_3 %d: %f\n", i, c3_3[i]);*/

	//////////////////////////////////////////////////p3////////////////////////////////////
	N = 4; type = 1;
	globalWork[0] = N;
	globalWork[1] = N;
	localWork[0] = 4;
	localWork[1] = 4;
	localWork[2] = 16;

	//set kernel arguments
	errNum = clSetKernelArg(conv_kernel, 0, sizeof(cl_mem), &conv_memObjects[1]);		//g_input
	errNum |= clSetKernelArg(conv_kernel, 1, sizeof(cl_mem), &conv_memObjects[0]);		//g_output
	errNum |= clSetKernelArg(conv_kernel, 6, sizeof(cl_int), &N);						//N
	errNum |= clSetKernelArg(conv_kernel, 7, sizeof(cl_int), &type);					//type

	//execute kernel
	errNum = clEnqueueNDRangeKernel(commandQueue, conv_kernel, 3, NULL,
		globalWork, localWork, 0, NULL, NULL);

	//test read buffer
	/*errNum = clEnqueueReadBuffer(commandQueue, conv_memObjects[0], CL_TRUE, 0,
		sizeof(float) * N * N * D2, p3, 0, NULL, NULL);

	for (int i = 0; i < 32; i++)
		printf("p3 %d: %f\n", i, p3[i]);*/



	/////////////////////////////////////////////////////////c4_1//////////////////////////////
	D2 = 512; type = 0;
	globalWork[2] = D2 * num_images;
	localWork[2] = 15;

	//set kernel arguments
	errNum = clSetKernelArg(conv_kernel, 0, sizeof(cl_mem), &conv_memObjects[0]);		//g_input
	errNum |= clSetKernelArg(conv_kernel, 1, sizeof(cl_mem), &conv_memObjects[1]);		//g_output
	errNum |= clSetKernelArg(conv_kernel, 4, sizeof(cl_int), &D2);						//D2
	errNum |= clSetKernelArg(conv_kernel, 7, sizeof(cl_int), &type);					//type

	//write buffer
	//filter
	errNum = clEnqueueWriteBuffer(commandQueue, conv_memObjects[2], CL_TRUE, 0,
		sizeof(float) * 3 * 3 * D1 * D2, w4_1, 0, NULL, NULL);
	//biases
	errNum |= clEnqueueWriteBuffer(commandQueue, conv_memObjects[3], CL_TRUE, 0,
		sizeof(float) * D2, b4_1, 0, NULL, NULL);

	//execute kernel
	errNum = clEnqueueNDRangeKernel(commandQueue, conv_kernel, 3, NULL,
		globalWork, localWork, 0, NULL, NULL);

	//test read buffer
	/*errNum = clEnqueueReadBuffer(commandQueue, conv_memObjects[1], CL_TRUE, 0,
		sizeof(float) * N * N * D2, c4_1, 0, NULL, NULL);

	for (int i = 0; i < 32; i++)
		printf("c4_1 %d: %f\n", i, c4_1[i]);*/

	///////////////////////////////////////////////////////////////c4_2//////////////////////
	D1 = 512;

	//set kernel arguments
	errNum = clSetKernelArg(conv_kernel, 0, sizeof(cl_mem), &conv_memObjects[1]);		//g_input
	errNum |= clSetKernelArg(conv_kernel, 1, sizeof(cl_mem), &conv_memObjects[0]);		//g_output
	errNum |= clSetKernelArg(conv_kernel, 5, sizeof(cl_int), &D1);						//D1

	//write buffer
	//filter
	errNum = clEnqueueWriteBuffer(commandQueue, conv_memObjects[2], CL_TRUE, 0,
		sizeof(float) * 3 * 3 * D1 * D2, w4_2, 0, NULL, NULL);
	//biases
	errNum |= clEnqueueWriteBuffer(commandQueue, conv_memObjects[3], CL_TRUE, 0,
		sizeof(float) * D2, b4_2, 0, NULL, NULL);

	//execute kernel
	errNum = clEnqueueNDRangeKernel(commandQueue, conv_kernel, 3, NULL,
		globalWork, localWork, 0, NULL, NULL);

	//test read buffer
	/*errNum = clEnqueueReadBuffer(commandQueue, conv_memObjects[0], CL_TRUE, 0,
		sizeof(float) * N * N * D2, c4_2, 0, NULL, NULL);

	for (int i = 0; i < 32; i++)
		printf("c4_2 %d: %f\n", i, c4_2[i]);*/

	////////////////////////////////////////////////////c4_3//////////////////////////////////////////

	//set kernel arguments
	errNum = clSetKernelArg(conv_kernel, 0, sizeof(cl_mem), &conv_memObjects[0]);		//g_input
	errNum |= clSetKernelArg(conv_kernel, 1, sizeof(cl_mem), &conv_memObjects[1]);		//g_output

	//write buffer
	//filter
	errNum = clEnqueueWriteBuffer(commandQueue, conv_memObjects[2], CL_TRUE, 0,
		sizeof(float) * 3 * 3 * D1 * D2, w4_3, 0, NULL, NULL);
	//biases
	errNum |= clEnqueueWriteBuffer(commandQueue, conv_memObjects[3], CL_TRUE, 0,
		sizeof(float) * D2, b4_3, 0, NULL, NULL);

	//execute kernel
	errNum = clEnqueueNDRangeKernel(commandQueue, conv_kernel, 3, NULL,
		globalWork, localWork, 0, NULL, NULL);

	//test read buffer
	/*errNum = clEnqueueReadBuffer(commandQueue, conv_memObjects[1], CL_TRUE, 0,
		sizeof(float) * N * N * D2, c4_3, 0, NULL, NULL);

	for (int i = 0; i < 32; i++)
		printf("c4_3 %d: %f\n", i, c4_3[i]);*/

	//////////////////////////////////////////////////////////////////////p4
	N = 2; type = 1;
	globalWork[0] = N;
	globalWork[1] = N;
	localWork[0] = 2;
	localWork[1] = 2;
	localWork[2] = 64;

	//set kernel arguments
	errNum = clSetKernelArg(conv_kernel, 0, sizeof(cl_mem), &conv_memObjects[1]);		//g_input
	errNum |= clSetKernelArg(conv_kernel, 1, sizeof(cl_mem), &conv_memObjects[0]);		//g_output
	errNum |= clSetKernelArg(conv_kernel, 6, sizeof(cl_int), &N);						//N
	errNum |= clSetKernelArg(conv_kernel, 7, sizeof(cl_int), &type);					//type

	//execute kernel
	errNum = clEnqueueNDRangeKernel(commandQueue, conv_kernel, 3, NULL,
		globalWork, localWork, 0, NULL, NULL);

	//test read buffer
	/*errNum = clEnqueueReadBuffer(commandQueue, conv_memObjects[0], CL_TRUE, 0,
		sizeof(float) * N * N * D2, p4, 0, NULL, NULL);

	for (int i = 0; i < 32; i++)
		printf("p4 %d: %f\n", i, p4[i]);*/

	///////////////////////////////////////////////////////////////c5_1//////////////////////
	type = 0;
	localWork[2] = 60;

	//set kernel arguments
	errNum = clSetKernelArg(conv_kernel, 0, sizeof(cl_mem), &conv_memObjects[0]);		//g_input
	errNum |= clSetKernelArg(conv_kernel, 1, sizeof(cl_mem), &conv_memObjects[1]);		//g_output
	errNum |= clSetKernelArg(conv_kernel, 7, sizeof(cl_int), &type);					//type

	//write buffer
	//filter
	errNum = clEnqueueWriteBuffer(commandQueue, conv_memObjects[2], CL_TRUE, 0,
		sizeof(float) * 3 * 3 * D1 * D2, w5_1, 0, NULL, NULL);
	//biases
	errNum |= clEnqueueWriteBuffer(commandQueue, conv_memObjects[3], CL_TRUE, 0,
		sizeof(float) * D2, b5_1, 0, NULL, NULL);

	//execute kernel
	errNum = clEnqueueNDRangeKernel(commandQueue, conv_kernel, 3, NULL,
		globalWork, localWork, 0, NULL, NULL);

	//test read buffer
	/*errNum = clEnqueueReadBuffer(commandQueue, conv_memObjects[1], CL_TRUE, 0,
		sizeof(float) * N * N * D2, c5_1, 0, NULL, NULL);

	for (int i = 0; i < 32; i++)
		printf("c5_1 %d: %f\n", i, c5_1[i]);*/

	//////////////////////////////////////////////////////////////c5_2/////////////////////////

	//set kernel arguments
	errNum = clSetKernelArg(conv_kernel, 0, sizeof(cl_mem), &conv_memObjects[1]);		//g_input
	errNum |= clSetKernelArg(conv_kernel, 1, sizeof(cl_mem), &conv_memObjects[0]);		//g_output

	//write buffer
	//filter
	errNum = clEnqueueWriteBuffer(commandQueue, conv_memObjects[2], CL_TRUE, 0,
		sizeof(float) * 3 * 3 * D1 * D2, w5_2, 0, NULL, NULL);
	//biases
	errNum |= clEnqueueWriteBuffer(commandQueue, conv_memObjects[3], CL_TRUE, 0,
		sizeof(float) * D2, b5_2, 0, NULL, NULL);

	//execute kernel
	errNum = clEnqueueNDRangeKernel(commandQueue, conv_kernel, 3, NULL,
		globalWork, localWork, 0, NULL, NULL);

	//test read buffer
	/*errNum = clEnqueueReadBuffer(commandQueue, conv_memObjects[0], CL_TRUE, 0,
		sizeof(float) * N * N * D2, c5_2, 0, NULL, NULL);

	for (int i = 0; i < 32; i++)
		printf("c5_2 %d: %f\n", i, c5_2[i]);*/

	////////////////////////////////////////////////////////////////////c5_3//////////////// 

	//set kernel arguments
	errNum = clSetKernelArg(conv_kernel, 0, sizeof(cl_mem), &conv_memObjects[0]);		//g_input
	errNum |= clSetKernelArg(conv_kernel, 1, sizeof(cl_mem), &conv_memObjects[1]);		//g_output

	//write buffer
	//filter
	errNum = clEnqueueWriteBuffer(commandQueue, conv_memObjects[2], CL_TRUE, 0,
		sizeof(float) * 3 * 3 * D1 * D2, w5_3, 0, NULL, NULL);
	//biases
	errNum |= clEnqueueWriteBuffer(commandQueue, conv_memObjects[3], CL_TRUE, 0,
		sizeof(float) * D2, b5_3, 0, NULL, NULL);

	//execute kernel
	errNum = clEnqueueNDRangeKernel(commandQueue, conv_kernel, 3, NULL,
		globalWork, localWork, 0, NULL, NULL);

	//test read buffer
	/*errNum = clEnqueueReadBuffer(commandQueue, conv_memObjects[1], CL_TRUE, 0,
		sizeof(float) * N * N * D2, c5_3, 0, NULL, NULL);

	for (int i = 0; i < 32; i++)
		printf("c5_3 %d: %f\n", i, c5_3[i]);*/

	///////////////////////////////////////////////////////////////p5///////////////////
	N = 1; type = 1;
	globalWork[0] = N;
	globalWork[1] = N;
	localWork[0] = 1;
	localWork[1] = 1;
	localWork[2] = 64;

	//set kernel arguments
	errNum = clSetKernelArg(conv_kernel, 0, sizeof(cl_mem), &conv_memObjects[1]);		//g_input
	errNum |= clSetKernelArg(conv_kernel, 1, sizeof(cl_mem), &conv_memObjects[0]);		//g_output
	errNum |= clSetKernelArg(conv_kernel, 6, sizeof(cl_int), &N);						//N
	errNum |= clSetKernelArg(conv_kernel, 7, sizeof(cl_int), &type);					//type

	//execute kernel
	errNum = clEnqueueNDRangeKernel(commandQueue, conv_kernel, 3, NULL,
		globalWork, localWork, 0, NULL, NULL);
									
	//Result : read buffer
	/*errNum = clEnqueueReadBuffer(commandQueue, conv_memObjects[0], CL_TRUE, 0,
		sizeof(float) * N * N * D2 * num_images, p5, 0, NULL, NULL);*/
	//for (int i = 0; i < 32; i++)
	//	printf("p5 %d: %f\n", i, p5[i]);
	//for (int i = 0; i < 32; i++)
	//	printf("p5 %d: %f\n", i, p5[i + 512]);

	///////////////////////////////////////////////////
	////fc1

	
	///////////////////////////////////////////////////////////fc1/////////////
	// N= 512, M = 512

	int M = 512;
	N = 512;
	
	float* fc_weights[3] = { w1, w2, w3 };
	float* fc_biases[3] = { b1, b2, b3 };

	globalWork[0] = M;
	globalWork[1] = num_images;
	globalWork[2] = 1;
	localWork[0] = 1;
	localWork[1] = 250;
	localWork[2] = 1;

	type = 2;

	errNum = clSetKernelArg(conv_kernel, 0, sizeof(cl_mem), &conv_memObjects[0]);		//g_input(input_neuron)
	errNum |= clSetKernelArg(conv_kernel, 1, sizeof(cl_mem), &conv_memObjects[1]);		//g_output(output_neuron)
	errNum |= clSetKernelArg(conv_kernel, 2, sizeof(cl_mem), &conv_memObjects[2]);		//g_weights(weights)
	errNum |= clSetKernelArg(conv_kernel, 3, sizeof(cl_mem), &conv_memObjects[3]);		//g_biases(biases)
	errNum |= clSetKernelArg(conv_kernel, 4, sizeof(cl_int), &M);						//(D2)
	errNum |= clSetKernelArg(conv_kernel, 5, sizeof(cl_int), &N);						//(D1)
	errNum |= clSetKernelArg(conv_kernel, 7, sizeof(cl_int), &type);					//type

	//filter
	errNum = clEnqueueWriteBuffer(commandQueue, conv_memObjects[2], CL_TRUE, 0,
		sizeof(float) * M * N, fc_weights[0], 0, NULL, NULL);
	//biases
	errNum |= clEnqueueWriteBuffer(commandQueue, conv_memObjects[3], CL_TRUE, 0,
		sizeof(float) * M, fc_biases[0], 0, NULL, NULL);

	//execute kernel
	errNum = clEnqueueNDRangeKernel(commandQueue, conv_kernel, 3, NULL,
		globalWork, localWork, 0, NULL, NULL);

	//test read buffer
	/*errNum = clEnqueueReadBuffer(commandQueue, conv_memObjects[1], CL_TRUE, 0,
		sizeof(float) * 512 * num_images, fc1, 0, NULL, NULL);
	for (int i = 0; i < 32; i++)
		printf("fc1 %d: %f\n", i, fc1[i]);*/

	///////////////////////////////////////////////////////////////////fc2//////////////////////

	errNum = clSetKernelArg(conv_kernel, 0, sizeof(cl_mem), &conv_memObjects[1]);			//g_input
	errNum |= clSetKernelArg(conv_kernel, 1, sizeof(cl_mem), &conv_memObjects[0]);		//g_output

	//filter
	errNum = clEnqueueWriteBuffer(commandQueue, conv_memObjects[2], CL_TRUE, 0,
		sizeof(float) * M * N, fc_weights[1], 0, NULL, NULL);
	//biases
	errNum |= clEnqueueWriteBuffer(commandQueue, conv_memObjects[3], CL_TRUE, 0,
		sizeof(float) * M, fc_biases[1], 0, NULL, NULL);

	//execute kernel
	errNum = clEnqueueNDRangeKernel(commandQueue, conv_kernel, 3, NULL,
		globalWork, localWork, 0, NULL, NULL);

	//test read buffer
	/*errNum = clEnqueueReadBuffer(commandQueue, conv_memObjects[0], CL_TRUE, 0,
		sizeof(float) * 512 * num_images, fc2, 0, NULL, NULL);
	for (int i = 0; i < 32; i++)
		printf("fc2 %d: %f\n", i, fc2[i]);*/

	/////////////////////////////////////////////////////////////////////fc3////////////////////////
	M = 10;
	globalWork[0] = M;
	
	errNum = clSetKernelArg(conv_kernel, 0, sizeof(cl_mem), &conv_memObjects[0]);		//g_input
	errNum |= clSetKernelArg(conv_kernel, 1, sizeof(cl_mem), &conv_memObjects[1]);		//g_output
	errNum |= clSetKernelArg(conv_kernel, 4, sizeof(cl_int), &M);						//M

	//filter
	errNum = clEnqueueWriteBuffer(commandQueue, conv_memObjects[2], CL_TRUE, 0,
		sizeof(float) * M * N, fc_weights[2], 0, NULL, NULL);
	//biases
	errNum |= clEnqueueWriteBuffer(commandQueue, conv_memObjects[3], CL_TRUE, 0,
		sizeof(float) * M, fc_biases[2], 0, NULL, NULL);

	//execute kernel
	errNum = clEnqueueNDRangeKernel(commandQueue, conv_kernel, 3, NULL,
		globalWork, localWork, 0, NULL, NULL);

	//read buffer
	errNum = clEnqueueReadBuffer(commandQueue, conv_memObjects[1], CL_TRUE, 0,
		sizeof(float) * M * num_images, fc3, 0, NULL, NULL);

	/*for (int i = 0; i < 32; i++)
		printf("fc3 %d: %f\n", i, fc3[i]);*/

	//////////////////////////////////////////////////////
	float* temp = fc3;
	for (int i = 0; i < num_images; i++)
	{
		softmax(temp, 10);
		labels[i] = find_max(temp, 10);
		confidences[i] = temp[labels[i]];
		temp += 10;
	}

	//printf("fc_seq time: %f sec\n", (double)(end - start) / CLK_TCK);

	free(c1_1); free(c1_2); free(p1);
	free(c2_1); free(c2_2); free(p2);
	free(c3_1); free(c3_2); free(c3_3); free(p3);
	free(c4_1); free(c4_2); free(c4_3); free(p4);
	free(c5_1); free(c5_2); free(c5_3); free(p5);
	free(fc1); free(fc2); free(fc3);
}


cl_context CreateContext()
{
	cl_int errNum;
	cl_platform_id* platformIds;
	cl_uint numPlatforms;
	cl_context context;

	// �÷����� ������ ���Ѵ�.
	errNum = clGetPlatformIDs(0, NULL, &numPlatforms);
	// �÷����� ID ����� ���Ѵ�.
	platformIds = new cl_platform_id[numPlatforms];
	errNum = clGetPlatformIDs(numPlatforms, platformIds, NULL);
	if (errNum != CL_SUCCESS || numPlatforms <= 0)
	{
		std::cerr << "Failed to find any OpenCl platforms." << std::endl;
	}

	// ù��° �÷������κ��� OpenCL ������ �����Ѵ�.
	// GPU ��� ������ �����Ϸ��� �õ��ϰ�, ���� �����ϸ�
	// CPU ��� ������ �����Ϸ��� �õ��Ѵ�.
	cl_context_properties contexProperties[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platformIds[0], 0 };
	context = clCreateContextFromType(contexProperties, CL_DEVICE_TYPE_GPU, NULL, NULL, &errNum);

	if (errNum != CL_SUCCESS)
	{
		std::cout << "Could not create GPU context, trying CPU..." << std::endl;
		context = clCreateContextFromType(contexProperties, CL_DEVICE_TYPE_CPU, NULL, NULL, &errNum);
		if (errNum != CL_SUCCESS)
		{
			std::cerr << "Failed to create an OpenCL GPU or CPU context.";
			return NULL;
		}
	}

	delete[] platformIds;
	return context;
}

cl_command_queue CreateCommandQueue(cl_context context, cl_device_id* device)
{
	cl_int errNum;
	size_t size;
	cl_device_id* devices;

	//����̽� ID ����� ũ�⸦ ���Ѵ�.
	errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, OUT & size);

	if (errNum != CL_SUCCESS)
	{
		std::cerr << "Failed call to clGetContextInfo()" << std::endl;
		return NULL;
	}
	if (size <= 0)
	{
		std::cerr << "No devices available.\n";
		return NULL;
	}

	//����̽� ID ����� �����´�.
	devices = new cl_device_id[size / sizeof(cl_device_id)];
	errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, size, OUT devices, NULL);
	if (errNum != CL_SUCCESS)
	{
		std::cerr << "Failed to get devices IDs\n";
		return NULL;
	}


	// ù��° ����̽��� �����ؼ� Ŀ�ǵ�ť ����
	cl_command_queue commandQueue = clCreateCommandQueue(context, devices[0], CL_QUEUE_PROFILING_ENABLE, NULL);

	if (commandQueue == NULL)
	{
		std::cerr << "Failed to create commandQueue for device 0\n";
		return NULL;
	}

	*device = devices[0];
	delete[] devices;
	return commandQueue;
}

cl_program CreateProgram(cl_context context, cl_device_id device, const char* fileName)
{
	cl_int errNum;
	cl_program conv_program;
	std::ifstream kernelFile(fileName, std::ios::in);
	if (!kernelFile.is_open())
	{
		std::cerr << "Failed to open file for reading : " << fileName << std::endl;
		return NULL;
	}

	std::ostringstream oss;
	oss << kernelFile.rdbuf();

	std::string srcStdStr = oss.str();
	const char* srcStr = srcStdStr.c_str();
	conv_program = clCreateProgramWithSource(context, 1, (const char**)&srcStr, NULL, NULL);
	if (conv_program == NULL)
	{
		std::cerr << "Failed to create CL conv_program from source.\n";
		return  NULL;
	}

	errNum = clBuildProgram(conv_program, 0, NULL, NULL, NULL, NULL);
	if (errNum != CL_SUCCESS)
	{
		char buildLog[10000];
		clGetProgramBuildInfo(conv_program, device, CL_PROGRAM_BUILD_LOG, sizeof(buildLog), buildLog, NULL);
		std::cerr << "Error in kernel : " << std::endl;
		std::cerr << buildLog;
		clReleaseProgram(conv_program);
		return NULL;
	}
	return conv_program;
}

void Cleanup(cl_context context, cl_command_queue commandQueue,
	cl_program program, cl_kernel kernel, cl_mem* memObjects, int mem_obj_num)
{
	for (int i = 0; i < mem_obj_num; i++)
	{
		if (memObjects[i] != 0)
			clReleaseMemObject(memObjects[i]);
	}
	if (commandQueue != 0)
		clReleaseCommandQueue(commandQueue);

	if (kernel != 0)
		clReleaseKernel(kernel);

	if (program != 0)
		clReleaseProgram(conv_program);

	if (context != 0)
		clReleaseContext(context);
}