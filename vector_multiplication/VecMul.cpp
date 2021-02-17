#define _CRT_SECURE_NO_WARNINGS

#include <iostream>
#include <fstream>
#include <sstream>
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <windows.h>

#define _CRT_SECURE_NO_DEPRECATE

#define CHECK_ERROR(err) \
	if(err != CL_SUCCESS) { \
		printf("[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err); \
		exit(EXIT_FAILURE); \
	}

#define A_ROWSIZE 1000
#define A_COLSIZE 1000
#define A_MATSIZE A_ROWSIZE*A_COLSIZE
#define B_ROWSIZE A_COLSIZE
#define B_COLSIZE 1000
#define B_MATSIZE B_ROWSIZE*B_COLSIZE
#define C_MATSIZE A_ROWSIZE*B_COLSIZE

char* get_source_code(const char* file_name, size_t* len)
{
	char* source_code;
	char buf[2] = "\0";
	int cnt = 0;
	size_t length;
	FILE* file = fopen(file_name, "r");
	if (file == NULL)
	{
		printf("[%s:%d] Failed to open %s n", __FILE__, __LINE__, file_name);
		exit(EXIT_FAILURE);
	}
	fseek(file, 0, SEEK_END);
	length = (size_t)ftell(file);
	rewind(file);

	source_code = (char*)malloc(length + 1);
	fread(source_code, length, 1, file);
	for (int i = 0; i < length; i++)
	{
		buf[0] = source_code[i];
		if (buf[0] == '\n')
		{
			cnt++;
		}
	}
	source_code[length - cnt] = '\0';
	fclose(file);
	*len = length - cnt;
	return source_code;
}

void printMatrix(int* mat, int colsz, int rowsz)
{
	for (int i = 0; i < colsz; i++)
	{
		for (int j = 0; j < rowsz; j++)
			printf("%d ", mat[colsz*i + j]);
		printf("\n");
	}
	printf("\n");
}

void InitMatrix(int* mat, int matSize, int val)
{
	for (int i = 0; i < matSize; i++)
	{
			mat[i] = val;
	}
}

void InitMatrixRand(int* mat, int matSize)
{
	srand(time(NULL));
	for (int i = 0; i < matSize; i++)
	{
			mat[i] = rand()%10;
	}
}

int main(void)
{
	cl_uint num_platforms;

	//get # of platforms 
	clGetPlatformIDs(0, NULL, &num_platforms);

	cl_platform_id* platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * num_platforms);

	//get platform IDs
	clGetPlatformIDs(num_platforms, platforms, NULL);

	printf("%u platforms\n", num_platforms);

	size_t name_size;
	char* name;

	for (int i = 0; i < num_platforms; i++)
	{
		clGetPlatformInfo(platforms[0], CL_PLATFORM_NAME, 0, NULL, &name_size);
		name = (char*)malloc(sizeof(char) * name_size);
		clGetPlatformInfo(platforms[0], CL_PLATFORM_NAME, name_size, name, NULL);

		printf("%s\n\n", name);
		free(name);
	}

	//device
	cl_device_id* device;
	cl_uint num_devices;
	cl_context context;
	cl_int err;

	cl_command_queue queue;

	const char* source_code;
	size_t source_size;

	cl_program program;

	cl_kernel kernel_mat_mul;

	int A_ColNum = A_COLSIZE;
	int A_RowNum = A_ROWSIZE;
	int B_ColNum = B_COLSIZE;

	cl_mem buffer[3];

	//malloc
	int* A = (int*)malloc(sizeof(int) * A_MATSIZE);
	int* B = (int*)malloc(sizeof(int) * B_MATSIZE);
	int* C = (int*)malloc(sizeof(int) * C_MATSIZE);
	if (A == NULL || B == NULL || C == NULL)
		return 1;

	//for test;
	int* seqA = (int*)malloc(sizeof(int) * A_MATSIZE);
	int* seqB = (int*)malloc(sizeof(int) * B_MATSIZE);
	int* seqC = (int*)malloc(sizeof(int) * C_MATSIZE);
	if (seqA == NULL || seqB == NULL || seqC == NULL)
		return 1;
	
	//init
	InitMatrixRand(A, A_MATSIZE);
	InitMatrixRand(B, B_MATSIZE);
	InitMatrix(C, C_MATSIZE, 0);

	//seq init
	memcpy(seqA, A, sizeof(int) * A_MATSIZE);
	memcpy(seqB, B, sizeof(int) * B_MATSIZE);
	memcpy(seqC, C, sizeof(int) * C_MATSIZE);

	//seq test
	for (int i = 0; i < A_ROWSIZE; i++)
	{
		for (int j = 0; j < B_COLSIZE; j++)
		{
			for (int k = 0; k < A_COLSIZE; k++)
				seqC[i * B_COLSIZE + j] += seqA[i * A_COLSIZE + k] * seqB[k * B_COLSIZE + j];
		}
	}
	printf("sequantial pocessed matrix (first 10x10)\n\n");
	printMatrix(seqC, 10, 10);

	clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);

	device = (cl_device_id*)malloc(sizeof(cl_device_id) * num_devices);
	if (device == NULL)
		return 1;
	//get device
	clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, num_devices, device, NULL);

	//create context
	context = clCreateContext(NULL, 1, device, NULL, NULL, &err);
	CHECK_ERROR(err);

	//create command queue
	queue = clCreateCommandQueueWithProperties(context, *device, 0, &err);
	CHECK_ERROR(err);

	//read source
	source_code = get_source_code("kernel.cl", &source_size);

	//create program object
	program = clCreateProgramWithSource(context, 1, &source_code, &source_size, &err);
	CHECK_ERROR(err);

	//build program
	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	CHECK_ERROR(err);

	//create kernel object
	kernel_mat_mul = clCreateKernel(program, "mat_mul", &err);
	CHECK_ERROR(err);

	//create buffer obj and set deliver arg
	buffer[0] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * A_MATSIZE, A, &err);
	buffer[1] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * B_MATSIZE, B, &err);
	buffer[2] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * C_MATSIZE, NULL, &err);

	//set kernel arg deliver addr of buffer obj
	err = clSetKernelArg(kernel_mat_mul, 0, sizeof(cl_mem), &buffer[0]);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel_mat_mul, 1, sizeof(cl_mem), &buffer[1]);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel_mat_mul, 2, sizeof(cl_mem), &buffer[2]);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel_mat_mul, 3, sizeof(int), &A_ColNum); 
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel_mat_mul, 4, sizeof(int), &A_RowNum); 
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel_mat_mul, 5, sizeof(int), &B_ColNum); 
	CHECK_ERROR(err);

	//exe kernel
	size_t global_size[2] = { A_ROWSIZE, B_COLSIZE };
	size_t local_size[2] = { 1, 1 };

	//add command to execute kernel
	clEnqueueNDRangeKernel(queue, kernel_mat_mul, 2, NULL, global_size, local_size, 0, NULL, NULL);

	//wait executing
	clFinish(queue);

	//host reads buffer from device
	err = clEnqueueReadBuffer(queue, buffer[2], CL_TRUE, 0, sizeof(int) * C_MATSIZE, C, 0, NULL, NULL);
	CHECK_ERROR(err);

	printf("parallel pocessed matrix (first 10x10)\n\n");
	printMatrix(C, 10, 10);


	//check parallel processing did correctly
	if (memcmp(C, seqC, sizeof(C)) == 0)
		printf("processing success\n");
	else
		printf("processing false\n");

	free(device);
	free(platforms);
	free((void*)source_code);

	free(A);
	free(B);
	free(C);
	free(seqA);
	free(seqB);
	free(seqC);

	clReleaseProgram(program);
	clReleaseKernel(kernel_mat_mul);
	clReleaseMemObject(buffer[2]);
	clReleaseMemObject(buffer[1]);
	clReleaseMemObject(buffer[0]);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);

	return 0;
}