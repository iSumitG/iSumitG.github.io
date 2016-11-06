---
layout:     post
title:      Hands-on Deep Learning on Big Red II Supercomputer
date:       2016-03-23
summary:    A guide to run Caffe deep learning framework on Indiana University's Big Red II Supercomputer including installation, training and testing on MNIST and your own dataset.
---

This post is for Computer Vision (B657) students and anyone who wants to run [Caffe](http://caffe.berkeleyvision.org/) deep learning framework on Indiana University's [Big Red II](https://kb.iu.edu/d/bcqt) supercomputer. Taken from handout by Prof. [David Crandall](https://www.cs.indiana.edu/~djcran/) and presented by Associate Instructors [Sumit Gupta](http://gsumit.com/about/) and Zhenhua Chen.

This post will introduce you to using the open-source Caffe software package for deep learning with convolutional neural networks. Caffe is just one of many packages that have become available recently ([TensorFlow](https://www.tensorflow.org/), [Theano](http://deeplearning.net/software/theano/) etc.). As deep learning can be extremely computationally demanding, Caffe is highly optimized to take advantage of modern Graphics Processing Units (GPUs), and easily outperforms CPU-only implementations by a factor of 10 or more with a high-end GPU. Big Red II has many GPUs -- 676 high-end GPUs, each with 32 compute cores. Big Red II was, when launched in 2013, the fastest supercomputer owned by a university. It's a Cray XE6/XK7 with a total of 1364 CPUs, 21824 processor cores, 43648 GB of memory, nearly 10 petabytes of disk space, and a peak performance over 1 petaFLOP (1 quadrillion operations per second). 

Caffe is a complicated software package and getting it running is always a little tricky. This post contains the steps required to run Caffe on Big Red II, and also some simple examples of using Caffe for real-life classification problems.

### Part 1: Getting started

**[1]** You will need a Big Red II account. Go to [IT Accounts](https://access.iu.edu/Accounts) to create a new one. 

**[2]** Log in to supercomputer via <span style="font-family: monospace">ssh</span> to <span style="font-family: monospace"><b>bigred2.uits.iu.edu</b></span>.

**[3]** You have a home directory on Big Red II that has a maximum storage area of 100 GBs. It is mounted on a shared network drive, which is problematic for Caffe's internal database (LMDB), so we'll do our work in a [scratch space](https://kb.iu.edu/d/bcqt#storage) instead. However the scratch space is automatically deleted after 14 days, so make sure to move files you care about to your home directory.

Go to your scratch space and make a local copy of the Caffe distribution:

{% highlight powershell %}
cd /N/dc2/scratch/your-user-id #substitute in your user ID! 
cp -R -d /N/soft/cle4/caffe/caffe/caffe-master/ caffe
cd caffe
{% endhighlight %}

**[4]** Big Red II supports a large variety of different libraries and software packages, many of which would conflict with one another if all were installed at the same time. Big Red II thus uses the Linux module system to allow you to easily select which libraries and packages you need. To load the ones required for Caffe, type: 

{% highlight powershell %}
module switch PrgEnv-cray/5.2.82 PrgEnv-gnu/5.2.82
module add cudatoolkit/5.5.51-1.0402.9594.3.1
module add opencv
module add boost/1.54.0
module add cray-hdf5/1.8.13
module add intel/15.0.3.187
module add caffe
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/N/soft/cle4/caffe/caffe/caffe-master/build/lib
{% endhighlight %}

You may want to create a UNIX script that will do this automatically, so you don't have to type this in every time you want to use Big Red II.

### Part 2: A first example: MNIST

Let's start by working through one of the example that comes with Caffe. This is the original handwriting recognition dataset that inspired CNNs nearly 20 years ago ([more](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf)). 

**[1]** First, we need to make a few changes to the source code to make it compatible with Big Red II. Edit the file <span style="font-family: monospace">examples/mnist/convert_mnist_data.cpp</span>, and find this line:

{% highlight powershell %}
CHECK_EQ(mdb_env_set_mapsize(mdb_env, 1099511627776), MDB_SUCCESS) // 1TB
{% endhighlight %}

For some reason, the Cray variant of Linux doesn't like this, so change that huge number to be 4294967296 instead.

Then edit the file <span style="font-family: monospace">src/caffe/util/db.cpp</span> and make the same change to the line:

{% highlight powershell %}
const size_t LMDB_MAP_SIZE = 1099511627776; // 1 TB
{% endhighlight %}

**[2]** Re-make the examples by typing:

{% highlight powershell %}
make
{% endhighlight %}

**[3]** Now download and prepare the MNIST data:

{% highlight powershell %}
./data/mnist/get_mnist.sh
./examples/mnist/create_mnist.sh
{% endhighlight %}

**[4]** You might be wondering which of Big Red II's 1364 CPUs you are currently using, and how you can get access to the other 1363 of them. The answer is that you are not able to access any of the compute or GPU nodes directly. Instead, you request compute nodes through a job scheduler: you request a certain number and type of nodes, you wait until the system has enough nodes available to meet your specifications, and then you run the program though a special scheduling program called <span style="font-family: monospace">aprun</span>. In our case, let's request interactive (as opposed to batch) time on a single GPU node: 

{% highlight powershell %}
qsub -I -q gpu
{% endhighlight %}

You may have to wait a while -- a few seconds, or maybe a few minutes -- for a node to become available.

**[5]** Now, let's check that everything so far is set up properly and that Caffe can access the GPU:

{% highlight powershell %}
aprun build/tools/caffe device_query -gpu 0
{% endhighlight %}

If successful, you should see a list of information about the system's GPU.

**[6]** Finally, let's start deep learning! 

{% highlight powershell %}
aprun examples/mnist/train_lenet.sh
{% endhighlight %}

If all is set up correctly, you'll now see Caffe swinging into action, and displaying constant information about its learning progress. Every few iterations, it displays a message telling you the current *loss* (e.g. current training error -- the sum of squared distances between predictions and ground truth across the training data). You should see this number generally go down over time. Every 1000 or so iterations, Caffe will pause training to test its current model on the test dataset, and will display the current accuracy and the current loss. Over time, if training goes well, we'd expect the loss to decrease and the accuracy to increase.

**[7]** Training will stop automatically after 10000 iterations, or you can quit it manually with CTRL-C if you get bored. The final model is stored in <span style="font-family: monospace">lenet_iter_10000.caffemodel</span>, and this model can then be used for classifying new digits. 

**[8]** The power of Caffe is that it allows you to customize the network architecture and parameters. We just used the defaults above, but you can change these defaults by modifying the Caffe model and solver configuration files, which are called the *prototxt* files. The relevant files for this example are <span style="font-family: monospace">examples/mnist/lenet_solver.prototxt</span> and <span style="font-family: monospace">examples/mnist/lenet_train_test.prototxt</span>. Read about these files and their settings in the [Caffe MNIST tutorial](http://caffe.berkeleyvision.org/gathered/examples/mnist.html). 

### Part 3: Image classification on your own data from scratch

Choose a problem with some image data to try out. For example, you could use the landmark classification data from assignment 2 (again, this is related to B657 class only). To do this: 

**[1]** Generate two files, <span style="font-family: monospace">train.txt</span> and <span style="font-family: monospace">test.txt</span>, that contain the image file names and correct ground truth labels. The labels should be integers starting at 0, and the format of each line should be <span style="font-family: monospace">image_filename label</span>, e.g. 

{% highlight powershell %}
image1.jpg 3
image2.jpg 3
image3.jpg 4
{% endhighlight %}

The <span style="font-family: monospace">test.txt</span> file is actually validation data, not testing data.

**[2]** Design your network and set meta-parameters. The tricky part is how to set these meta-parameters. One method is to set the parameters and then train the network, observing two things: (1) if the network is converging, and (2) is it fast enough? Based on your observation you can then readjust your parameters. In addition, you can also set other things such as the output directory, name of the net, etc. To set parameters and architecture, you'll need to edit <span style="font-family: monospace">train_val.prototxt</span> and <span style="font-family: monospace">solver.prototxt</span>.

**[3]** To do the training, first get access to GPU, as before:

{% highlight powershell %}
qsub -I -q gpu
cd /N/dc2/scratch/your-username
aprun build/tools/caffe device_query -gpu 0
{% endhighlight %}

**[4]** Then perform training:

{% highlight powershell %}
aprun ./build/tools/caffe train -solver MyExample/solver.prototxt -gpu 0
{% endhighlight %}

### Part 4: Image classification with fine-tuning

Were you satisfied with the result of classification in Part 3? Well, we can borrow a pre-trained model (like [AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) -- which was trained on millions of images!) and use it on our comparatively small dataset to (hopefully!) get better results.

**[1]** Add python

{% highlight powershell %}
module load python
{% endhighlight %}

**[2]** Download 1000 images and train/val file lists.
{% highlight powershell %}
python examples/finetune_flickr_style/assemble_data.py --images=1000
{% endhighlight %}

**[3]** Compute the imagenet mean file (Why do we need this mean file?)

{% highlight powershell %}
./data/ilsvrc12/get_ilsvrc_aux.sh
{% endhighlight %}

**[4]** Adjust the learning rate and number of outputs of the fully-connected layer in the following files:

<span style="font-family: monospace">
models/finetune_flickr_style/solver.prototxt
models/finetune_flickr_style/train_val.prototxt
</span>

**[5]** Download pre-trained model.

{% highlight powershell %}
python scripts/download_model_binary.py models/bvlc_reference_caffenet
{% endhighlight %}

**[6]** Get access to GPU

{% highlight powershell %}
qsub -I -q gpu
cd /N/dc2/scratch/your-username
aprun build/tools/caffe device_query -gpu 0
{% endhighlight %}

**[7]** And finally, training:

{% highlight powershell %}
aprun ./build/tools/caffe train -solver models/finetune_flickr_style/solver.prototxt -weights models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel -gpu 0
{% endhighlight %}
