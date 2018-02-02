#define __CL_ENABLE_EXCEPTIONS
#define CL_USE_DEPRECATED_OPENCL_1_1_APIS

#include <CL/cl.h>
#include "cl.hpp"

#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <iterator>
#include <math.h>

typedef float elem_type;

void generate_input(size_t n, size_t m){
    std::ofstream out("input.txt");
    out << n << ' ' <<  m << "\n";
    for (int i = 0; i < n; ++i){
        for (int j = 0; j < n; ++j){
            out << "1 ";
        }
        out << "\n";
    }
    for (int i = 0; i < m; ++i){
        for (int j = 0; j < m; ++j){
            out << "1 ";
        }
        out << "\n";
    }
}

void read_one_matrix(std::ifstream &in, std::vector<elem_type> &M, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        for (size_t j = 0; j < size; ++j) {
            in >> M[i * size + j];
        }
    }
}

void write_output(std::vector<elem_type> &C, size_t size) {
    FILE *out = fopen("output.txt", "w");
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            fprintf(out, "%0.3f ", C[i * size + j]);
        }
        fprintf(out, "\n");
    }
    fclose(out);

//    for (int i = 0; i < size; ++i) {
//        for (int j = 0; j < size; ++j) {
//            std::cout << C[i * size + j] << " ";
//        }
//        std::cout << "\n";
//    }

}

int main() {
    std::vector<cl::Platform> platforms;
    std::vector<cl::Device> devices;
    std::vector<cl::Kernel> kernels;

    try {

        //generate_input(1024, 9);

        // create platform
        cl::Platform::get(&platforms);
        platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);

        // create context
        cl::Context context(devices);

        // create command queue
        cl::CommandQueue queue(context, devices[0]);

        // load opencl source
        std::ifstream cl_file("matrix_conv.cl");
        std::string cl_string(std::istreambuf_iterator<char>(cl_file), (std::istreambuf_iterator<char>()));
        cl::Program::Sources source(1, std::make_pair(cl_string.c_str(),
                                                      cl_string.length() + 1));

        // create program
        cl::Program program(context, source);

        // compile opencl source
        program.build(devices);

        // create a message to send to kernel
        std::ifstream in("input.txt");
        size_t n, m;
        in >> n >> m;
        size_t matrix_size = n * n;
        std::vector<elem_type> A(matrix_size, 0), B(m * m, 0), C(matrix_size, 0);
        read_one_matrix(in, A, n);
        read_one_matrix(in, B, m);

        // allocate device buffer to hold message
        cl::Buffer dev_a(context, CL_MEM_READ_ONLY, sizeof(elem_type) * matrix_size);
        cl::Buffer dev_b(context, CL_MEM_READ_ONLY, sizeof(elem_type) * matrix_size);
        cl::Buffer dev_c(context, CL_MEM_WRITE_ONLY, sizeof(elem_type) * matrix_size);

        // copy from cpu to gpu
        queue.enqueueWriteBuffer(dev_a, CL_TRUE, 0, sizeof(elem_type) * matrix_size, A.data());
        queue.enqueueWriteBuffer(dev_b, CL_TRUE, 0, sizeof(elem_type) * matrix_size, B.data());

        // load named kernel from opencl source
        cl::Kernel kernel(program, "convolution");
        size_t const block_size = 16;
        size_t const global_size = ((matrix_size + block_size - 1) / block_size) * block_size;
        cl::KernelFunctor convolution(kernel, queue, cl::NullRange, cl::NDRange(global_size, global_size),
                                      cl::NDRange(block_size, block_size));
        convolution(dev_a, dev_b, dev_c, (int) n, (int) m);

        queue.enqueueReadBuffer(dev_c, CL_TRUE, 0, sizeof(elem_type) * matrix_size, C.data());

        write_output(C, n);

    }
    catch (cl::Error e) {
        std::cout << std::endl << e.what() << " : " << e.err() << std::endl;
    }

    return 0;
}
