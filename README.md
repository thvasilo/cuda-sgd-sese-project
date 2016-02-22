# Implementation of mini-batch Stochastic Gradient Descent (SGD) in CUDA using Thrust and cuBLAS

This is a (bad) implementation of mini-batch SGD I did in CUDA using [Thrust](http://docs.nvidia.com/cuda/thrust/) and [cuBLAS](http://docs.nvidia.com/cuda/cublas/),
originally in the context of the KTH PDC summer school ["Introduction to  High Performance Computing"](http://agenda.albanova.se/conferenceDisplay.py?confId=5012) and extended
for the course ["Introduction to GPU and accelerator programming for scientific computing"](http://sese.nu/introduction-to-gpu-and-accelerator-programming-for-scientific-computing-2015/).

The SGD iterations code has three possible codepaths to choose from:

* **Use plain CUDA calls and Thrust.**
  This was the original implementation I did for the PDC summer school. It calls a CUDA kernel for the gradient
  calculations, and the rest is done through Thrust calls.
* **Use plain CUDA calls with cuBLAS through dynamic parallelism, and Thrust.**
  This is a small modification of the original code, where instead of performing dot products and AXPY through
  for loops, I used cuBLAS calls within the kernels instead. This led to abysmal performance, I am not sure why yet,
  but my guess is that the thread scheduling gets messed up somehow. I talk more about this in the report.
* **Use cuBLAS calls and Thrust.**
  This is the best performing codepath. The gradient calculation is done through a GEMV call using cuBLAS and the
  rest is done through either cuBLAS or Thrust calls. No CUDA device code is used in this codepath.

Take a look at the report to see how the runtimes compare. For a linear regression task with a dataset of ~1.6GB,
the cuBLAS implementation is ~5x faster than the plain CUDA one.

The reason I call this a bad implementation is because currently, it uses the size of the batch as the unit of parallelism.
That means that that if you run an experiment with batch size of 10, you will only get 10 threads running on the GPU
which is horrible. So if you want any kind of performance here you have to use large batch sizes (at least 1000).
I do plan to change the implementation to be able to run multiple batches in parallel and then it will
become an "OK" implementation.  
Still, since I couldn't find any other approachable/intuitive implementations of SGD in CUDA out there, I thought this might
prove useful to some people, despite this obvious design problem.

I'm also including the results (with plots) of various experiments I ran, as well as the report for the SeSE course, which uses
some parts (intro and related work) of the report we co-wrote with Ramón Heberto Martínez Mayorquin for the PDC
summer school. Ramón also wrote the original versions of the Python scripts used to automate our experiments.

Improvements I would like to make:

* Change the parallelism so it is no longer bound to the batch size, allowing us to fully occupy the GPU.
* The cuBLAS codepath currently uses twice more the GPU memory than it should. This is because in order to create a
  permuted copy of the original data (shuffle) before each iteration, I maintain both a copy of the original data and
  a permuted copy. So for a 1GB dataset, I use 2GB memory which is unnecessary. I should be doing the shuffle in-place
  instead.

Acknowledgements: I would like to thank [Robert Crovella](http://stackoverflow.com/users/1695960/robert-crovella)
for all the help he has provided on Stackoverflow, you will find a couple of his SO replies in this implementation.
