#include <iostream>
#include <cmath>
#include <vector>
#include <CL/sycl.hpp>
#include <chrono> 
#include <random>

using namespace std;
using namespace sycl;

void printVector(vector<double> x) {
    for (int i = 0; i < x.size(); ++i) {
        std::cout << x[i] << " ";
    }
    std::cout << std::endl << std::endl;
}


vector <double> softmax(vector<double>& input) {

    vector<double> softmaxOutput;
    vector<double> exponents;

    double denominator = 0;

    for (int i = 0; i < input.size(); ++i) {

        exponents.push_back(exp(input[i]));

        denominator += exp(input[i]);
    }

    for (int i = 0; i < input.size(); ++i) {
        softmaxOutput.push_back(exponents[i] / denominator);
    }
    return softmaxOutput;
}

vector<double> softmax_buffer(const vector<double>& input) {
    int NumOfElements = input.size();

    vector<double> output(NumOfElements);

    queue Q(cpu_selector_v);

    range<1> num_items(NumOfElements);

    buffer<double, 1> input_buffer(input.data(), num_items);
    buffer<double, 1> output_buffer(output.data(), num_items);


    Q.submit([&](handler& h) {

        auto input_accessor = input_buffer.get_access<access::mode::read_write>(h);
        auto output_accessor = output_buffer.get_access<access::mode::write>(h);

        h.parallel_for(num_items, [=](id<1> idx) {

            double denominator = 0;

            for (int i = 0; i < NumOfElements; ++i) {

                atomic_ref<double, sycl::memory_order::relaxed, memory_scope::device, access::address_space::global_space> InputAtomic(input_accessor[i]);
                atomic_ref<double, sycl::memory_order::relaxed, memory_scope::device, access::address_space::global_space> atomicdenominator(denominator);

                atomicdenominator += exp(InputAtomic.load());
            }


            for (int i = 0; i < NumOfElements; ++i) {

                atomic_ref<double, sycl::memory_order::relaxed, memory_scope::device, access::address_space::global_space> InputAtomic(input_accessor[i]);
                double current_input = InputAtomic.load();

                atomic_ref<double, sycl::memory_order::relaxed, memory_scope::device, access::address_space::global_space> OutPutAtomic(output_accessor[i]);

                OutPutAtomic = exp(current_input) / denominator;

            }

            });

        }).wait();


        return output;
}

vector<double> softmax_USM_Implicit(const vector<double>& input) {

    int NumOfElements = input.size();

    std::vector<double> output(NumOfElements);

    queue Q{ property::queue::in_order() };

    const double* input_data = input.data();
    double* output_data = output.data();

    double* sharedArray = malloc_shared<double>(input.size(), Q);


    Q.submit([&](handler& h) {

        h.parallel_for(NumOfElements, [=](id<1> idx) {
            double denominator = 0;

            for (int i = 0; i < NumOfElements; ++i) {
                denominator += exp(sharedArray[i]);
            }

            output_data[idx] = exp(sharedArray[idx]) / denominator;

            });

    }).wait();

        free(sharedArray, Q);

        return output;

}

vector<double> softmax_USM_Explicit(const vector<double>& input) {

    int NumOfElements = input.size();

    std::vector<double> output(NumOfElements);

    queue Q;

    const double* input_data = input.data();
    double* output_data = output.data();

    double* INPUT = malloc_device<double>(input.size(), Q);
    double* OUTPUT = malloc_device<double>(output.size(), Q);

    Q.memcpy(INPUT, input_data, sizeof(double) * NumOfElements).wait();


    Q.submit([&](handler& h) {

        h.parallel_for(NumOfElements, [=](id<1> idx) {
            double denominator = 0;

            for (int i = 0; i < NumOfElements; ++i) {
                denominator += exp(INPUT[i]);
            }

            OUTPUT[idx] = exp(INPUT[idx]) / denominator;

            });

        }).wait();

        std::vector<double> host_data(NumOfElements);

        Q.memcpy(host_data.data(), OUTPUT, sizeof(double) * NumOfElements).wait();


        free(INPUT, Q);
        free(OUTPUT, Q);

        return host_data;
}

vector<double> softmax_subgroups(const vector<double>& input) {

    const size_t NumOfElements = input.size();

    vector<double> output(NumOfElements);

    queue Q(cpu_selector_v);

    range<1> num_items(NumOfElements);

    buffer<double, 1> input_buffer(input.data(), num_items);
    buffer<double, 1> output_buffer(output.data(), num_items);

    const size_t Size = 9;

    Q.submit([&](handler& h) {


        auto input_accessor = input_buffer.get_access<access::mode::read_write>(h);
        auto output_accessor = output_buffer.get_access<access::mode::write>(h);

        sycl::accessor<float, 1, access::mode::read_write, access::target::local> tileA(range<1>(Size), h);
        sycl::accessor<float, 1, access::mode::write, access::target::local> tileB(range<1>(Size), h);

        h.parallel_for<class SoftMax>(nd_range<1>(range<1>(NumOfElements), range<1>(Size)), [=](nd_item<1> item) {



            auto local_id = item.get_local_id(0);
            auto idx = item.get_global_id(0);


            double denominator = 0;

            for (int i = 0; i < NumOfElements; i++) {
                const int local_idx = local_id + i;


                tileA[local_id] = sycl::exp(input_accessor[local_idx]);
                item.barrier(access::fence_space::local_space);

                denominator += sycl::exp(input_accessor[local_idx]);
            }

            item.barrier(access::fence_space::local_space);

            double local_output = sycl::exp(input_accessor[idx]) / denominator;

            item.barrier(access::fence_space::local_space);
            output_accessor[idx] = local_output;

            });
        }).wait();

        return output;
}


int main() {

    int vectorSize = 9000;
    vector<double> input(vectorSize);

    const int seed = 23;
    srand(seed);


    for (int i = 0; i < vectorSize; ++i) {
        input[i] = rand() % 100;
    }


    auto start_time = chrono::steady_clock::now();

    vector<double> output = softmax_subgroups(input);

    auto end_time = chrono::steady_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);



    double probabilitySum = 0;

    for (int i = 0; i < output.size(); ++i) {

        probabilitySum += output[i];
    }

    std::cout << "\nSum of the outputs: " << probabilitySum << std::endl << std::endl;
    std::cout << "Execution time for Soft Max RAW: " << duration.count() << " milliseconds" << std::endl;

    return 0;
}
