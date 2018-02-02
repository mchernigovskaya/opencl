#define __CL_ENABLE_EXCEPTIONS
#define CL_USE_DEPRECATED_OPENCL_1_1_APIS

#include <CL/cl.h>
#include "cl.hpp"

#include <vector>
#include <fstream>
#include <iostream>
#include <iterator>
#include <iomanip>
#include <assert.h>
#include <math.h>

typedef double elem_type;

size_t const BLOCK_SIZE = 256;

void generate_input(size_t n) {
    std::ofstream out("input.txt");
    out << n << "\n";
    for (int i = 0; i < n; ++i) {
            out << "1 ";
    }
}

std::vector<elem_type> read_input() {
    std::ifstream in("input.txt");
    size_t n;
    in >> n;
    std::vector<elem_type> A(n, 0);
    for (size_t i = 0; i < n; ++i) {
        in >> A[i];
    }
    return A;
}

void write_output(std::vector<elem_type> &output, size_t size) {
    FILE *out = fopen("output.txt", "w");
    for (int i = 0; i < size; ++i) {
        fprintf(out, "%0.3f ", output[i]);
    }
    fprintf(out, "\n");
    fclose(out);

//    for (int i = 0; i < size; ++i) {
//        std::cout << output[i] << " ";
//    }
//    std::cout << "\n";
}

std::vector<elem_type> inclusive_scan(cl::Context &context, cl::CommandQueue &queue, cl::Program &program, std::vector <elem_type> &input) {
    std::vector<elem_type> cur_result(input.size(), 0);

    // allocate device buffer to hold message
    cl::Buffer dev_input(context, CL_MEM_READ_ONLY, sizeof(elem_type) * input.size());
    cl::Buffer dev_output(context, CL_MEM_WRITE_ONLY, sizeof(elem_type) * cur_result.size());

    // copy from cpu to gpu
    queue.enqueueWriteBuffer(dev_input, CL_TRUE, 0, sizeof(elem_type) * input.size(), &input[0]);

    // load named kernel from opencl source
    cl::Kernel kernel(program, "scan_hillis_steele");
    cl::KernelFunctor scan_hs(kernel, queue, cl::NullRange, cl::NDRange(input.size()), cl::NDRange(BLOCK_SIZE));
    cl::Event event = scan_hs(dev_input, dev_output, cl::__local(sizeof(elem_type) * BLOCK_SIZE), cl::__local(sizeof(elem_type) * BLOCK_SIZE));

    event.wait();

    queue.enqueueReadBuffer(dev_output, CL_TRUE, 0, sizeof(elem_type) * cur_result.size(), &cur_result[0]);

    if (cur_result.size() <= BLOCK_SIZE) {
        return cur_result;
    }

    std::vector <elem_type> tails(BLOCK_SIZE * ((cur_result.size() / BLOCK_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE), 0);
    for (int i = 1; i * BLOCK_SIZE - 1 < cur_result.size() && i < tails.size(); ++i) {
        tails[i] = cur_result[i * BLOCK_SIZE - 1];
    }
    std::vector <elem_type> tails_pref = inclusive_scan(context, queue, program, tails);

    // allocate device buffer
    cl::Buffer dev_input_next_step(context, CL_MEM_READ_ONLY, sizeof(elem_type) * tails_pref.size());

    // copy from cpu to gpu
    queue.enqueueWriteBuffer(dev_input, CL_TRUE, 0, sizeof(elem_type) * cur_result.size(), &cur_result[0]);
    queue.enqueueWriteBuffer(dev_input_next_step, CL_TRUE, 0, sizeof(elem_type) * tails_pref.size(), &tails_pref[0]);

    // load named kernel from opencl source
    cl::Kernel kernel_next_step(program, "next_step");
    cl::KernelFunctor next_step(kernel_next_step, queue, cl::NullRange, cl::NDRange(input.size()), cl::NDRange(BLOCK_SIZE));
    cl::Event event_next_step = next_step(dev_input, dev_input_next_step, dev_output);

    event_next_step.wait();

    std::vector<elem_type> result(input.size(), 0);
    queue.enqueueReadBuffer(dev_output, CL_TRUE, 0, sizeof(elem_type) * result.size(), &result[0]);

    return result;
}

int main() {
    std::vector<cl::Platform> platforms;
    std::vector<cl::Device> devices;
    std::vector<cl::Kernel> kernels;

    try {

        //generate_input(10000);

        // create platform
        cl::Platform::get(&platforms);
        platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);

        // create context
        cl::Context context(devices);

        // create command queue
        cl::CommandQueue queue(context, devices[0]);

        // load opencl source
        std::ifstream cl_file("scan.cl");
        std::string cl_string(std::istreambuf_iterator<char>(cl_file), (std::istreambuf_iterator<char>()));
        cl::Program::Sources source(1, std::make_pair(cl_string.c_str(), cl_string.length() + 1));

        // create program
        cl::Program program(context, source);

        // compile opencl source
        program.build(devices, ("-D BLOCK_SIZE=" + std::to_string(BLOCK_SIZE)).c_str());

        std::vector<elem_type> input = read_input();

        const size_t n = input.size();
        input.resize(BLOCK_SIZE * ((n + BLOCK_SIZE - 1) / BLOCK_SIZE));

        std::vector <elem_type> output = inclusive_scan(context, queue, program, input);
        write_output(output, n);

    }
    catch (cl::Error e) {
        printf("\n%s: %d\n", e.what(), e.err());
    }

    return 0;
}