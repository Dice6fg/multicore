main: main.o cnn_opencl.o 
	g++ -o main main.o cnn_opencl.o -l OpenCL

main.o cnn_opencl.o: cnn.h
	g++ -c main.cpp cnn_opencl.cpp

.PHONY: clean
clean:
	del *.o