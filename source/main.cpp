/*
Copyright 2017 Forrest Briggs

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

I am providing code in this repository to you under an open source license. Because this is my personal repository, the license you receive to my code is from me, and not from my employer (Facebook).
*/

#include <iostream>

#include "opencv2/calib3d.hpp"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include "Util.h"
#include "NanoNet.h"

using namespace std;
using namespace cv;
using namespace nanonet;

void testImageRegression(const string& activationFunc) {
  Mat srcImage = imread("lena.png");
  cout << "srcImage: " << srcImage.size() << " " << srcImage.channels() << endl;

  cout << "building training dataset" << endl;
  vector<Tensor> trainInputs;
  vector<Tensor> trainTargets;
  for (int y = 0; y < srcImage.rows; ++y) {
    for (int x = 0; x < srcImage.cols; ++x) {
      //if (rand() % 100 != 0) { continue; } // subsample
      // compute feature vector
      const float u = float(x) / float(srcImage.rows - 1);
      const float v = float(y) / float(srcImage.cols - 1);
      const vector<float> feature = {u, v};
      // compute target vector
      Vec3b color = srcImage.at<Vec3b>(y, x);

      const vector<float> target = {
        color[0] / 255.0f,
        color[1] / 255.0f,
        color[2] / 255.0f};

      trainInputs.push_back(Tensor(feature));
      trainTargets.push_back(Tensor(target));
    }
  }

  NanoNet net;
  if (activationFunc == "sin") {
    net.layers = {
      new FullyConnectedLayer<Sine>(2, 16),
      new FullyConnectedLayer<Sine>(16, 16),
      new FullyConnectedLayer<Sine>(16, 16),
      new FullyConnectedLayer<Sine>(16, 16),
      new FullyConnectedLayer<Sine>(16, 3),
      new OutputLayer1D(3)
    };
  } else if (activationFunc == "relu") {
    net.layers = {
      new FullyConnectedLayer<ReLu>(2, 16),
      new FullyConnectedLayer<ReLu>(16, 16),
      new FullyConnectedLayer<ReLu>(16, 16),
      new FullyConnectedLayer<ReLu>(16, 16),
      new FullyConnectedLayer<ReLu>(16, 3),
      new OutputLayer1D(3)
    };
  } else {
    cout << "usage: ./TestNanoNet [sin | relu]" << endl;
    exit(1);
  }
  net.randomizeWeights();

  NanoNet::vertifyGradientNumerically(
    net,
    trainInputs[11],
    trainTargets[11],
    LOSS_MSE);

  for (int epoch = 0; epoch < 100; ++epoch) {
    cout << "epoch=" << epoch << endl;

    static const int kNumBatches = 100000;
    static const int kBatchSize = 16;
    static const float kLearningRate = 0.1f;
    net.train(trainInputs, trainTargets, LOSS_MSE, kNumBatches, kBatchSize, kLearningRate);

    Mat predictedImage(srcImage.size(), CV_32FC3);
    for (int y = 0; y < srcImage.rows; ++y) {
      for (int x = 0; x < srcImage.cols; ++x) {
        const float u = float(x) / float(srcImage.rows - 1);
        const float v = float(y) / float(srcImage.cols - 1);
        const vector<float> feature = {u, v};
        net.forwardPropagate(Tensor(feature));
        const Tensor& output = net.layers[net.layers.size() - 1]->a;
        predictedImage.at<Vec3f>(y, x) =
          Vec3f(output.at(0), output.at(1), output.at(2));
      }
    }
    imwrite(activationFunc + "_"+std::to_string(epoch)+".png", predictedImage * 255.0f);
  }
}

int main(int argc, char** argv) {
  if (argc != 2) {
    cout << "usage: ./TestNanoNet [sin | relu]" << endl;
    exit(1);
  }
  testImageRegression(argv[1]);
  return EXIT_SUCCESS;
}
