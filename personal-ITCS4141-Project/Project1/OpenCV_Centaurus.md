## Setting up OpenCV 
You can create a conda environment and install OpenCV for your own use on [Centaurus](https://oneit.charlotte.edu/innovation/research-computing/educational-cluster/). 
Here are the basic steps needed to create the environment, using Python 3.12 and OpenCV 4.13.0 as example and you can install a different combination. 

      $ module load miniforge/25.9
      $ mamba create -n opencv-4.13.0 python=3.12 conda-forge::opencv=4.13.0


That's it! It should be installed in the user's $HOME/.conda/envs/opencv-4.13.0 directory. Now you can activate the new environment to use it:

      $ mamba activate opencv-4.13.0

Every subsequent SSH session, you will need to 1) load the miniforge module, and then 2) activate the opencv environment:

      $ module load miniforge/25.9
      $ mamba activate opencv-4.13.0

If you plan to compile C++ code, you may have to alter your environments, 
to make it easier for the compiler to find the libraries and headers. You will need to add to $LD_LIBRARY_PATH and $PKG_CONFIG_PATH:

      export LD_LIBRARY_PATH=$HOME/.conda/envs/opencv-4.13.0/lib:$LD_LIBRARY_PATH
      export PKG_CONFIG_PATH=$HOME/.conda/envs/opencv-4.13.0/lib/pkgconfig:$PKG_CONFIG_PATH

Below is an example Makefile for using OpenCV headers and library

      CFLAGS = $(shell pkg-config --cflags opencv4)
      LIBS = $(shell pkg-config --libs opencv4)
      # Define the target executable name
      TARGET = test
      # The default rule to build the target
      $(TARGET): test.cpp
        g++ $(CFLAGS) $< -o $@ $(LIBS)
      # Clean rule to remove the generated executable
      clean:
        rm -f $(TARGET)
      .PHONY: all clean
