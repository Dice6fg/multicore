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

//time checking
clock_t start, finish;
double duration;

cl_int totalNum = 16777216;
cl_int localSize = 256;

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

double f(double x)
{
	return(3 * x * x + 2 * x + 1);
}

double integral(int N)
{
	double dx;
	double sum = 0;
	int i;
	for (i = 0; i < N; i++)
	{
		dx = i / (double)N;
		sum = sum + f(dx)/(double)N;
	}

	return sum;
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

	cl_kernel kernel_integral;


	cl_double* g_sum = (cl_double*)malloc(sizeof(cl_double) * totalNum/localSize);
	int isFirst = 1;

	cl_mem buffer;

	

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
	source_code = get_source_code("int_kernel.cl", &source_size);

	//create program object
	program = clCreateProgramWithSource(context, 1, &source_code, &source_size, &err);
	CHECK_ERROR(err);

	//build program
	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	CHECK_ERROR(err);

	//create kernel object
	kernel_integral = clCreateKernel(program, "integral", &err);
	CHECK_ERROR(err);

	//create buffer obj and set deliver arg
	buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_double) * totalNum / localSize, NULL, &err);
	

	//set kernel arg deliver addr of buffer obj
	err = clSetKernelArg(kernel_integral, 0, sizeof(cl_mem), &buffer);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel_integral, 1, sizeof(cl_mem), &buffer);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel_integral, 2, sizeof(cl_double)*256, NULL);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel_integral, 3, sizeof(cl_int), &totalNum);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel_integral, 4, sizeof(cl_int), &isFirst);
	CHECK_ERROR(err);
	isFirst = 0;

	//exe kernel
	size_t global_size = totalNum;
	size_t local_size = localSize;

	//time check
	start = clock();

	printf("global size: %d\n", global_size);
	err = clEnqueueNDRangeKernel(queue, kernel_integral, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
	CHECK_ERROR(err);
	//wait first reduction executing
	clFinish(queue); 

	global_size = global_size / local_size;
	//change input
	err = clSetKernelArg(kernel_integral, 0, sizeof(cl_mem), &buffer);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel_integral, 4, sizeof(cl_int), &isFirst);
	CHECK_ERROR(err);
	printf("global size: %d\n", global_size);
	err = clEnqueueNDRangeKernel(queue, kernel_integral, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
	CHECK_ERROR(err);
	//wait second reduction executing
	clFinish(queue);

	global_size = global_size / local_size;
	printf("global size: %d\n", global_size);
	err = clEnqueueNDRangeKernel(queue, kernel_integral, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
	CHECK_ERROR(err);
	//wait third reduction executing
	clFinish(queue);
	

	//host reads buffer from device
	err = clEnqueueReadBuffer(queue, buffer, CL_TRUE, 0, sizeof(cl_int) * totalNum/localSize, g_sum, 0, NULL, NULL);
	CHECK_ERROR(err);

	printf("parallel integral: %lf\n", g_sum[0]);

	//time check
	finish = clock();
	duration = (double)(finish - start) / CLOCKS_PER_SEC;
	printf("%fsec\n\n", duration);
	//////////////////////////

	//time check
	start = clock();

	double seqIntegral = integral(totalNum);

	
	printf("sequential integral: %lf\n", seqIntegral);

	//time check
	finish = clock();
	duration = (double)(finish - start) / CLOCKS_PER_SEC;
	printf("%fsec\n\n", duration);
	//////////////////////////

	//check parallel processing did correctly
	printf("%lf, %lf\n", g_sum[0], seqIntegral);
	if ((double)g_sum[0] == seqIntegral)
		printf("processing success\n");
	else
		printf("processing false\n");

	free(device);
	free(platforms);
	free((void*)source_code);

	free(g_sum);

	clReleaseProgram(program);
	clReleaseKernel(kernel_integral);
	clReleaseMemObject(buffer);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);

	return 0;
}