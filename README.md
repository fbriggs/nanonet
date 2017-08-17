== License ==

Copyright 2017 Forrest Briggs

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

I am providing code in this repository to you under an open source license. Because this is my personal repository, the license you receive to my code is from me, and not from my employer (Facebook).

== About ==

This is a basic implementation of a feed-forward multi-layer neural net (no convolutions). It includes a few tricks (documented in the code) so that it can converge 10+ layers deep. It uses a modular formulation of back-propagation. As a test, it learns a function from R^2 --> R^3 representing an image (lena.png), i.e. a map from (x, y) to (r, g, b). It is surprising and interesting that the reconstructed image looks better with sin() activation rather than relu() activation. I am not the first to try sin() as an activation function, but it doesn't seem to be widely used or studied. Some ideas as to why it may work well include:

(1) Many regions of the parameter space are equivalent. Take a step with SGD that is too large is not catastrophic, but instead jumps to another region of the parameter space from which it can still converge.

(2) Vanishing and exploding gradients are a well-known issue in training deep neural nets. Relu activation has 0 gradient over half its domain, and no bound on the magnitude of the gradient in its linear domain. In contrast, the gradient of sin() is always bounded in [-1, 1], and only 0 with probability 0. Hence, it can be expected that nets with sin() activation are less prone to vanishing and exploding gradients.

(3) The paper "Random Features for Large-Scale Kernel Machines" by Ali Rahimi and Ben Recht (https://people.eecs.berkeley.edu/~brecht/papers/07.rah.rec.nips.pdf), proposes to upgrade linear support vector machines to non-linear classification by applying a pre-processing feature transform, before applying a linear learning algorithm. They show that with an appropriate transform, inner products between transformed features are approximately equal (in expectation) to kernel evaluations. In order to approximate the RBF kernel, the transform consists simply of taking the cos() of the original feature's dot product with several random weight vectors sampled from a normal distribution. With appropriately chosen biases, and assuming weights are initialized from a normal distribution, it can be seen that Rahimi and Recht's feature transform to approximate RBF kernels is equivalent to the computation performed by one layer of a multi-layer feed-forward neural net with sin() activation functions! This is an interesting theoretical connection which I haven't seen pointed out anywhere outside of my work.

== install dependencies ==
brew install cmake
brew install opencv

== compile ==
cmake -DCMAKE_BUILD_TYPE=Release
make -j8

== run test ==
./TestNanoNet sin
./TestNanoNet relu

