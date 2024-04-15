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

                atomic_ref<double, sycl::memory_order::relaxed, memory_scope::device, access::address_space::global_space> atomicElem(input_accessor[i]);
                atomic_ref<double, sycl::memory_order::relaxed, memory_scope::device, access::address_space::global_space> atomicdenominator(denominator);

                atomicdenominator += exp(atomicElem.load());
            }


            for (int i = 0; i < NumOfElements; ++i) {

                output_accessor[idx] = exp(input_accessor[idx]) / denominator;

            }

            });

        }).wait();


        return output;

}



vector<double> softmax_USM(const vector<double>& input) {
  
    int NumOfElements = input.size();

    std::vector<double> output(NumOfElements);

    queue Q{property::queue::in_order() };

    const double* input_data = input.data();
    double* output_data = output.data();

    double* sharedArray = malloc_shared<double>(output_data[0], Q);


    Q.submit([&](handler& h) {

        h.parallel_for(NumOfElements, [=](id<1> idx) {
            double denominator = 0;

            for (int i = 0; i < NumOfElements; ++i) {
                denominator += exp(input_data[i]);
            }

            output_data[idx] = exp(input_data[idx]) / denominator;
        
        });

    }).wait();

    free(sharedArray, Q);

    return output;
}



int main() {

    int vectorSize = 80000;
    vector<double> input (vectorSize);
  
    const int seed = 23;
    srand(seed);


    for (int i = 0; i < vectorSize; ++i) {
        input[i] = rand() % 100;
    }


    auto start_time = chrono::steady_clock::now();

    vector<double> output = softmax_buffer(input);

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