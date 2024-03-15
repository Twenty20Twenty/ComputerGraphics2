#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <fstream>
#include <math.h>
#include <chrono>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace std;

double* setMatrix(double* M, int len) {
    for (int i = 0; i < len; i++)
        for (int j = 0; j < len; j++)
            M[i * len + j] = 1 + ((int)(rand() % 120));
    return M;
}

void printMatrix(double* M, int len) {
    for (int i = 0; i < len; i++) {
        for (int j = 0; j < len; j++)
            printf("%lf\t", M[i * len + j]);
        printf("\n");
    }printf("-----------------\n");
}

__device__ void kof(double* M, int i, int j, double tmp, int len) {
    int k = threadIdx.x;
    M[j * len + k] -= ((M[i * len + k] / M[i * len + i]) * tmp);
}


__global__ void gpuDet(double* M, int len) {
    if (len > 1) {
        for (int i = 0; i < len - 1; i++)
            for (int j = i + 1; j < len; j++) {
                double tmp = M[j * len + i];
                kof(M, i, j, tmp, len);
            }
    }
}


double det(double* M, int len) {

    int MatrixSize = len * len;
    int byteSize = MatrixSize * sizeof(double);
    double* inMatrix_d;
    cudaMalloc((void**)&inMatrix_d, byteSize);
    cudaMemcpy(inMatrix_d, M, byteSize, cudaMemcpyHostToDevice);

    gpuDet <<< 1, len >>> (inMatrix_d, len);
    cudaDeviceSynchronize();

    cudaMemcpy(M, inMatrix_d, byteSize, cudaMemcpyDeviceToHost);
    cudaFree(inMatrix_d);

    double det = M[0 * len + 0];
    for (int i = 1; i < len; i++)
        det *= M[i * len + i];
    return det;
}

void laba2() {
    int lenArr[12] = { 2,4,8,16,32,64,128,256,350,512,738,1024 };
    int iterations = 12;
    double* Matrix;
    int len;
    ofstream f;
    f.open("resultsGPU.txt");
    if (f.is_open()) {
        f << "len: \tTime:\n";

        for (int k = 0; k < iterations; k++) {

            len = lenArr[k];

            Matrix = new double[len * len];
            Matrix = setMatrix(Matrix, len);

            //printMatrix(Matrix, len);
            auto start1 = chrono::high_resolution_clock::now();
            det(Matrix, len);
            auto end1 = chrono::high_resolution_clock::now();



            delete[] Matrix;

            double time_taken1 = chrono::duration_cast<chrono::nanoseconds>(end1 - start1).count();
            time_taken1 *= 1e-9;

            cout << len << " \t" << time_taken1 << endl;
			f << len << " \t" << time_taken1 << endl;
        }
    }
}

void main() {
    laba2();
}