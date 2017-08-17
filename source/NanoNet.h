/*
Copyright 2017 Forrest Briggs

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

I am providing code in this repository to you under an open source license. Because this is my personal repository, the license you receive to my code is from me, and not from my employer (Facebook).
*/

/*
Derivation of backpropagation:
* http://ufldl.stanford.edu/tutorial/

Layer-wise modular formulation of backpropagation:
* https://www.cs.ox.ac.uk/people/nando.defreitas/machinelearning/lecture7.pdf

Random weight initialization:
* "Practical Recommendations for Gradient-Based Training of Deep Architectures" (http://arxiv.org/pdf/1206.5533v2.pdf)
* "Understanding the difficulty of training deep feedforward neural networks" (http://jmlr.csail.mit.edu/proceedings/papers/v9/glorot10a/glorot10a.pdf)

Weight projection onto a convex set:
In the works below, weights are projected to a ball of a fixed radius:
* "Maxout Networks" (Bengio, http://jmlr.csail.mit.edu/proceedings/papers/v28/goodfellow13.pdf)
* "Improving neural networks by preventing co-adaptation of feature detectors" (Hinton, http://arxiv.org/pdf/1207.0580.pdf)
We apply a similar projection, except instead of a sphere, it is a box.

Gradient clipping:
http://www.jmlr.org/proceedings/papers/v28/pascanu13.pdf
https://arxiv.org/pdf/1212.0901.pdf

TODO:
numerically verify bias gradients
conv layer
max pooling layer
cross-entropy
adam
*/

#pragma once

#include <exception>
#include <vector>

#include "Tensor.h"

namespace nanonet {

using namespace std;

struct Identity {
  static inline float f(const float& x) { return x; }
  static inline float df(const float& x) { return 1.0f; }
};

struct ReLu {
  static inline float f(const float& x) { return std::max(0.0f, x); }
  static inline float df(const float& x) { return x > 0.0f ? 1.0f : 0.0f; }
};

struct Sine {
  static inline float f(const float& x) { return sin(x); }
  static inline float df(const float& x) { return cos(x); }
};

struct Layer {
  virtual ~Layer() {}
  virtual void randomizeWeights() = 0;
  virtual void forwardPropagate(Layer* nextLayer) = 0;
  virtual void backPropagate(Layer* nextLayer) = 0;
  virtual Layer* copy() const = 0;
  Tensor w; // weights
  Tensor b; // biases
  Tensor g; // weight gradient
  Tensor gb; // bias gradient
  Tensor a; // activation
  Tensor z; // activation before non-linearity
  Tensor delta; // backprop sensitivity
};

struct OutputLayer1D : public Layer {
  int dim;

  OutputLayer1D(const int dim) : dim(dim) {
    a = Tensor(1, {dim});
    z = Tensor(1, {dim});
    delta = Tensor(1, {dim});
  }

  Layer* copy() const { return new OutputLayer1D(*this); }

  void randomizeWeights()                 { throw runtime_error("output layer cannot randomizeWeights()"); }
  void forwardPropagate(Layer* nextLayer) { throw runtime_error("output layer cannot forwardPropagate()"); }
  void backPropagate(Layer* nextLayer)    { throw runtime_error("output layer cannot backPropagate()"); }
};

template <typename ActivationFunc>
struct FullyConnectedLayer : public Layer {
  int inputDim, outputDim;

  FullyConnectedLayer(const int inputDim, const int outputDim)
    : inputDim(inputDim), outputDim(outputDim)
  {
    // w.at(i,j) = weight from input i to output j
    w = Tensor(2, {inputDim, outputDim});
    b = Tensor(1, {outputDim});
    g = Tensor(2, {inputDim, outputDim});
    gb = Tensor(1, {outputDim});
    a = Tensor(1, {inputDim});
    z = Tensor(1, {inputDim});
    delta = Tensor(1, {inputDim});
  }

  Layer* copy() const { return new FullyConnectedLayer<ActivationFunc>(*this); }

  void randomizeWeights() {
    const float r = sqrt(6.0f / (inputDim + outputDim));
    for (int j = 0; j < outputDim; ++j) {
      for (int i = 0; i < inputDim; ++i) {
        w.at(i,j) = (randUniform() - 0.5f) * 2.0f * r;
      }
      b.at(j) = 0.0f;
    }
  }

  void forwardPropagate(Layer* nextLayer) {
    for (int j = 0; j < outputDim; ++j) {
      nextLayer->z.at(j) = b.at(j);
      for (int i = 0; i < inputDim; ++i) {
        nextLayer->z.at(j) += w.at(i, j) * a.at(i);
      }
      nextLayer->a.at(j) = ActivationFunc::f(nextLayer->z.at(j));
    }
  }

  // fills in deltas and gradient for this layer
  void backPropagate(Layer* nextLayer) {
    for (int i = 0; i < inputDim; ++i) {
      delta.at(i) = 0.0f;
      for (int j = 0; j < outputDim; ++j) {
        delta.at(i) += nextLayer->delta.at(j) * w.at(i, j);
      }
      delta.at(i) *= ActivationFunc::df(z.at(i));
    }
    for (int j = 0; j < outputDim; ++j) {
      for (int i = 0; i < inputDim; ++i) {
        g.at(i, j) += a.at(i) * nextLayer->delta.at(j);
      }
      gb.at(j) += nextLayer->delta.at(j);
    }
  }
};

enum LossFunction {
  LOSS_MSE
};

struct NanoNet {
  vector<Layer*> layers;

  NanoNet() {}

  NanoNet(const NanoNet& copy) { // deep copy
    for (Layer* l : copy.layers) {
      layers.push_back(l->copy());
    }
  }

  void zeroGradient() {
    for (int i = 0; i < layers.size() - 1; ++i) { layers[i]->g.zero(); layers[i]->gb.zero(); }
  }
  void randomizeWeights() {
    for (int i = 0; i < layers.size() - 1; ++i) { layers[i]->randomizeWeights(); }
  }

  void forwardPropagate(const Tensor& input) {
    assert(layers[0]->a.dims == input.dims);
    for (Layer* l : layers) { l->z.zero(); l->a.zero(); }
    layers[0]->a = input;
    for (int l = 0; l < layers.size() - 1; ++l) {
      layers[l]->forwardPropagate(layers[l+1]);
    }
  }

  void backPropagate(LossFunction loss, const Tensor& target) {
    Layer& lastLayer = *layers[layers.size() - 1];
    assert(lastLayer.a.dims == target.dims);
    if (loss == LOSS_MSE) {
      for (int j = 0; j < target.dims[0]; ++j) {
        lastLayer.delta.at(j) = -2.0f * (target.at(j) - lastLayer.a.at(j));
      }
    } else {
      throw runtime_error("unimplemented loss");
    }
    for (int l = layers.size() - 2; l >= 0; --l) {
      layers[l]->backPropagate(layers[l+1]);
    }
  }

  void train(
      const vector<Tensor>& trainInputs,
      const vector<Tensor>& trainTargets,
      const LossFunction loss,
      const long numBatches,
      const int batchSize,
      const float learningRate) {

    const int n = trainInputs.size();
    cout << "# training examples=" << n << endl;
    for (long batch = 0; batch < numBatches; ++batch) {

      if (batch % 100000 == 0) {
        long randIndex = rand() % n; // TODO: is this correct for n > maxint?
        float objCurrExample = objective(trainInputs, trainTargets, loss);
        cout << batch << " " << "\t" << objCurrExample << endl;
      }

      zeroGradient(); // backPropagate doesn't zero the gradient, it just adds
      for (int b = 0; b < batchSize; ++b) {
        int randIndex = rand() % n;
        forwardPropagate(trainInputs[randIndex]);
        backPropagate(loss, trainTargets[randIndex]);
        //layers[0]->gb.print();
      }
      for (int l = 0; l < layers.size() - 1; ++l) {
        const float s =  -1.0f * learningRate / float(batchSize);
        layers[l]->gb.clamp(-1.0f, 1.0f); // this is a form of gradient clipping, but we apply it independently to the biases. this allows it to work with the same learning rate for weights/biases
        layers[l]->w.scaleAdd(layers[l]->g, s);
        layers[l]->b.scaleAdd(layers[l]->gb, s * 0.01f);
        layers[l]->w.clamp(-10.0f, 10.0f);
        layers[l]->b.clamp(-10.0f, 10.0f);
      }
    }
  }

  float objectiveSingleExample(const Tensor& input, const Tensor& target, LossFunction loss) {
    float obj = 0.0f;
    forwardPropagate(input);
    const Tensor& output = layers[layers.size()-1]->a;
    assert(target.rank == 1 && output.rank == 1);
    if (loss == LOSS_MSE) {
      for (int j = 0; j < target.dims[0]; ++j) {
        obj += (target.at(j) - output.at(j)) * (target.at(j) - output.at(j));
      }
    } else {
      throw runtime_error("unknown loss");
    }
    assert(!isnan(obj));
    assert(!isinf(obj));
    return obj;
  }

  float objective(
      const vector<Tensor>& trainInputs,
      const vector<Tensor>& trainTargets,
      const LossFunction loss) {
    float obj = 0.0f;
    for (int i = 0; i < trainInputs.size(); ++i) {
      obj += objectiveSingleExample(trainInputs[i], trainTargets[i], loss);
    }
    assert(!isnan(obj));
    assert(!isinf(obj));
    return obj / float(trainInputs.size());
  }

  static void vertifyGradientNumerically(
      NanoNet& net,
      const Tensor& input,
      const Tensor& target,
      const LossFunction loss) {

    cout << "verifying gradient" << endl;
    net.zeroGradient();
    net.forwardPropagate(input);
    net.backPropagate(loss, target);

    static const float kEpsilon = 1E-4;
    for (int l = 0; l < net.layers.size(); ++l) {
      cout << "l=" << l << ", rank(W)=" << net.layers[l]->w.rank << endl;
      if (net.layers[l]->w.rank == 2) {
        for (int i = 0; i < net.layers[l]->w.dims[0]; ++i) {
          for (int j = 0; j < net.layers[l]->w.dims[1]; ++j) {
            NanoNet netPlusEps(net);
            NanoNet netMinusEps(net);
            netPlusEps.layers[l]->w.at(i, j)  += kEpsilon;
            netMinusEps.layers[l]->w.at(i, j) -= kEpsilon;
            const float objPlusEps = netPlusEps.objectiveSingleExample(input, target, loss);
            const float objMinusEps = netMinusEps.objectiveSingleExample(input, target, loss);
            const float approxDeriv = (objPlusEps - objMinusEps) / (2.0f * kEpsilon);
            const float derivErr = fabs(net.layers[l]->g.at(i,j) - approxDeriv);
            cout << "l=" << l << " i=" << i << " j=" << j << " g=" << net.layers[l]->g.at(i,j) << " g_approx=" << approxDeriv << " L1_err = " << derivErr << endl;
          }
        }
      } else if (net.layers[l]->w.rank == 0) {
        cout << "nothing to verify in output layer" << endl;
      } else {
        cout << "TODO: numerically verify different tensor size";
      }
    }

  }
};

}; // end namespace
