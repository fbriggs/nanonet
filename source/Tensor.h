/*
Copyright 2017 Forrest Briggs

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

I am providing code in this repository to you under an open source license. Because this is my personal repository, the license you receive to my code is from me, and not from my employer (Facebook).
*/

#include <vector>
#include <iostream>

#include "Util.h"

namespace nanonet {

using namespace std;

struct Tensor {
  int rank;
  vector<int> dims;
  vector<float> data1;
  vector<vector<float>> data2;
  vector<vector<vector<float>>> data3;

  Tensor() : rank(0) {}

  Tensor(const int rank, const vector<int>& dims) : rank(rank), dims(dims) {
    switch(rank) {
    case 1: data1 = vector<float>(dims[0], 0.0f); break;
    case 2: data2 = vector<vector<float>>(dims[0], vector<float>(dims[1], 0.0f)); break;
    case 3: data3 = vector<vector<vector<float>>>(dims[0], vector<vector<float>>(dims[1], vector<float>(dims[2], 0.0f))); break;
    default: throw runtime_error("Tensor::Tensor unsupported tensor rank");
    }
  }

  Tensor(const vector<float>& v) : rank(1), data1(v) { dims.push_back(v.size()); }

  void zero() {
    switch(rank) {
    case 1: { for(int i = 0; i < dims[0]; ++i) { data1[i] = 0.0f; } } break;
    case 2: { for(int i = 0; i < dims[0]; ++i) { for (int j = 0; j < dims[1]; ++j) { data2[i][j] = 0.0f; } } } break;
    case 3: { for(int i = 0; i < dims[0]; ++i) { for (int j = 0; j < dims[1]; ++j) { for (int k = 0; k < dims[2]; ++k) { data3[i][j][k] = 0.0f; } } } } break;
    default: throw runtime_error("Tensor::zero unsupported tensor rank");
    }
  }

  // add another tensor with the same shape, multiplied by s, put resutsl in this tensor
  void scaleAdd(const Tensor& other, const float s) {
    assert(other.rank == rank && other.dims == dims);
    switch(rank) {
    case 1: { for (int i = 0; i < dims[0]; ++i) { data1[i] += s * other.data1[i]; } } break;
    case 2: { for (int i = 0; i < dims[0]; ++i) { for (int j = 0; j < dims[1]; ++j) { data2[i][j] += s * other.data2[i][j]; } } } break;
    case 3: { for (int i = 0; i < dims[0]; ++i) { for (int j = 0; j < dims[1]; ++j) { for (int k = 0; k < dims[2]; ++k) { data3[i][j][k] += s * other.data3[i][j][k]; } } } } break;
    default: throw runtime_error("Tensor::scaleAdd unsupported tensor rank");
    }
  }

  // clamp all elements
  void clamp(const float a, const float b) {
    switch(rank) {
    case 1: { for (int i = 0; i < dims[0]; ++i) { data1[i] = ::clamp<float>(data1[i], a, b); } } break;
    case 2: { for (int i = 0; i < dims[0]; ++i) { for (int j = 0; j < dims[1]; ++j) { data2[i][j] = ::clamp<float>(data2[i][j], a, b); } } } break;
    case 3: { for (int i = 0; i < dims[0]; ++i) { for (int j = 0; j < dims[1]; ++j) { for (int k = 0; k < dims[2]; ++k) { data3[i][j][k] = ::clamp<float>(data3[i][j][k], a, b); } } } } break;
    default: throw runtime_error("Tensor::clamp unsupported tensor rank");
    }
  }

  void print() const {
    if (rank == 1) {
      for (int i = 0; i < dims[0]; ++i) {
        cout << "\t" << i << ":" << data1[i] << endl;
      }
    }
    if (rank == 2) {
      for (int i = 0; i < dims[0]; ++i) {
        for (int j = 0; j < dims[1]; ++j) {
          cout << "\t" << i << "," << j << ":" << data2[i][j] << endl;
        }
      }
    }
    // TODO: finish
  }

  inline float& at(const int& i)                                          { assert(rank == 1); assert(i >= 0 && i < dims[0]); return data1[i]; }
  inline float& at(const int& i, const int& j)                            { assert(rank == 2); assert(i >= 0 && i < dims[0]); assert(j >= 0 && j < dims[1]); return data2[i][j]; }
  inline float& at(const int& i, const int& j, const int& k)              { assert(rank == 3); assert(i >= 0 && i < dims[0]); assert(j >= 0 && j < dims[1]); assert(k >= 0 && k < dims[2]); return data3[i][j][k]; }
  inline const float& at(const int& i) const                              { assert(rank == 1); assert(i >= 0 && i < dims[0]); return data1[i]; }
  inline const float& at(const int& i, const int& j) const                { assert(rank == 2); assert(i >= 0 && i < dims[0]); assert(j >= 0 && j < dims[1]); return data2[i][j]; }
  inline const float& at(const int& i, const int& j, const int& k) const  { assert(rank == 3); assert(i >= 0 && i < dims[0]); assert(j >= 0 && j < dims[1]); assert(k >= 0 && k < dims[2]); return data3[i][j][k]; }
};

};
