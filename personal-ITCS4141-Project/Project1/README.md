# Image filtering with convolution
Image filtering, which also referred to as image blurring, or convolution, is a frequently used image processing operation. The algorithm is to apply convolution operation on each image pixel through elemement-wise 3x3 matrix multiplication with a 3x3 filter matrix (also referred to as coefficients matrix). Applying different filter to an image using this algorithm yields images of different visual effects, such as blurring, smoothing, etc. The description of image filtering and an OpenCV example can be found from [OpenCV Tutorial for Filtering Images](https://docs.opencv.org/3.4.0/dc/dd3/tutorial_gausian_median_blur_bilateral_filter.html), and the [`filter_image` function in FILTER.C](FILTER.C) provide concise implementation for image filtering algorithms. 

You should implement the filtering algorithms according to the algorithms in [FILTER.C](FILTER.C). Please be noted that your implementation should work with RGB images (three layers) while FILTER.C is only for single-layer grayscale images.

## Source Code to Start
The skeleton code are provided in [Filtering.cpp](Filtering.cpp). You should also check [CMakeLists.txt](CMakeLists.txt) to see how they will be built to executables using cmake/make/gcc utilities. [README.md](README.md) provide instructions for how to build all executables at once. The intial source code of [Filtering.cpp](Filtering.cpp) file is a copy from [changing the contrast and brightness of an image](https://docs.opencv.org/3.4.0/d3/dc1/tutorial_basic_linear_transform.html). The methods for reading/writing/displaying an image and for getting/setting a pixel value of an image are pretty simple in OpenCV, which can be found from [this short description with code sample for operating images](https://docs.opencv.org/3.4.0/d5/d98/tutorial_mat_operations.html). The FilteringOpenCV.cpp file for OpenCV image processing are all from the OpenCV tutorial and they are for your reference only, which calls OpenCV implemented filtering. Your implementation should use loops to apply pixel-wise computation according to the convolution algorithms implemented in FILTER.c. 

## To Build
 1. clone this repo and cd to the clone work folder. 
 1. `mkdir build`, **only do once**. 
 1. `cd build; cmake ..` to create the Makefile, **only do once**
    1. On the lab machines, this should be enough. But if you use different machines and cmake fails to locate where OpenCV is installed, you can set it 
       by setting the OpenCV_DIR env to the right OpenCV installation location, e.g. `export OpenCV_DIR=/opt/opencv`, and then do the `cmake` call. 
 1. `make` to build the examples. **Do each time you change your source code, and want to build and execute the program** 
 1. execute an example, e.g. `./Filtering`

## Setting up OpenCV and SSH Access with X Forwarding
 1. [Setting up OpenCV on Centaurus](OpenCV_Centaurus.md)
 2. [SSH with X forwarding](SSH-XForwarding.md)

## Implementation
There are three versions for the implementation, one for each task. You need to submit your implementatio for each task and task 3 is the final submission and should include all the three implementation. The performance study should compare the performance of the three versions. 
 1. Task 1: Implement the sequential version of the filtering algorithm
 1. Task 2: Implement the parallelized GPU version of the filtering algorithm
 1. Task 3: Implement the parallelized CPU version of the filtering algorithm

## Submission
**Your submission should be a single zipped file named LastNameFirstName.zip that includes ONLY the following: the implemented source file, Filtering.cpp and a PDF file for your report. Please remove all other files, including the executables, Excel sheet, etc.** 
In the report, please describe:
1.	Short description on how you implement filtering.
1.	Performance report for comparing the three implementations (only for task 3). 

#### Grading: 

1. Functions implementations: 60 points. For source file that cannot be compiled, you only receive max 60% of function implementation points. For compliable, but with execution errors or incorrectness, you receive max 80% of function implementation points. 
1. Report: 40 points. 

## Assignment Policy
 1. Programming assignments are to be done individually. You may discuss assignments with others, but you must code your own solutions and submit them with a write-up in your own words. Indicate clearly the name(s) of student(s) you collaborated with, if any. 
 1. Although homework assignments will not be pledged, per se, the submitted solutions must be your work and not copied from other students' assignments or other sources. 
 1. You may not transmit or receive code from anyone in the class in any way--visually (by showing someone your code), electronically (by emailing, posting, or otherwise sending someone your code), verbally (by reading your code to someone), or in any other way.
 1. You may not collaborate with people who are not your classmates, TAs, or instructor in any way. For example, you may not post questions to programming forums. 
 1. Any violations of these rules will be reported to the honor council. Check the syllabus for the late policy and academic conduct. 

