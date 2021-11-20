Setup
=====


Requirements
------------

GENGA runs on NVIDIA and on AMD GPUs.

NVIDIA GPUs must have a compute capability of 3.0 or higher. 



Install CUDA
-------------

To be able to use the code on NVIDIA GPUs, one has to install the CUDA Toolkit first
as described here.

We strongly recommend using a recent CUDA version to get the full performance and correct results.
If an old CUDA version is used (< CUDA 9.0) then the :literal:`def_OldShuffle` parameter in the :ref:`define.h<Define>` file must be set to 1.
(See :ref:`OldShuffle`)

Linux
^^^^^

| Install the gcc compiler (for example, in Ubuntu install build-essential package) 
| Download the CUDA toolkit from: https://developer.nvidia.com/cuda-downloads .
| Install the CUDA toolkit.
| Reboot
| Run ``nvidia-smi`` to check CUDA and the available GPUs.


GCC version
^^^^^^^^^^^
It can happen that the used CUDA version needs an older GCC version than the current one on the system. In that case, either a newer CUDA version, or an older gcc version should be installed. Use the following compile option to tell CUDA to use an older GCC version (for example 7.0)::

	-ccbin=g++-7


.. Comment.
	.. _Windows:
	Windows
	^^^^^^^
	Install the compiler, for example Microsoft Visual Studio 2019 Community
	Download the CUDA toolkit from here: https://developer.nvidia.com/cuda-downloads .
	Install the CUDA toolkit.
	Reboot
	Run :literal:`nvidia-smi` to check CUDA and the available GPUs.
	Run Command Prompt shell
	Load the C compiler in the Command Prompt. Similar as::
	   call "C\Program Files (x86)\Microsoft Visual Studio\2019\Community\Common7\Tools\vsdevcmd.bat"
	where the exact path depends on the installation.


Install ROCM and HIP for AMD GPUs
---------------------------------

When an AMD GPU is used, then ROCM and HIP needs to be installed.

| Follow the instructions on: 
| https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html
| to install ROCM and HIP.
| Run ``/opt/rocm/bin/hipconfig --full`` to check the installation
| Run ``rocm-smi`` to check the GPU
| Choose the platform with either ``export HIP_PLATFORM=nvidia`` or ``export HIP_PLATFORM=amd``


Compile GENGA
-------------
If an old CUDA version is used (< CUDA 9.0) then the :literal:`def_OldShuffle` parameter in the :ref:`define.h<Define>` file must be set to 1.
(See :ref:`OldShuffle`)

GENGA must be compiled for a specific GPU compute capability. The compute capability corresponds to the GPU generation, 
a list of all NVIDIA GPUS with their compute capabilities can be found here: https://developer.nvidia.com/cuda-gpus .


The source code of GENGA and a Makefile is included in the source directory. To compile GENGA, go to the source directory
and type::

	make SM=xx

into a terminal, where :literal:`xx` corresponds to the compute capability of the GPU.

Use e.g. 'make SM=60' for compute capability of 6.0, or 'make SM=65' for compute capability of 6.5.


For example use::

	make SM=35 for Tesla K20
	make SM=52 for GeForce GTX 980
	make SM=60 for Tesla P100
	make SM=61 for GeForce GTX 1080 ti
	make SM=75 for GeForce RTX 2080 ti
	make SM=86 for GeForce RTX 3090


When compiling GENGA with the openGL real time visualization, go to the GengaGL directory.
(See :ref:`GengaGL`)

When GENGA is compiled for a newer compute capability then the GPU is able to run, then the following error message will appear by running GENGA:
`FGAlloc  error = 13 = invalid device symbol`.


Compile GENGA with HIP
^^^^^^^^^^^^^^^^^^^^^^
GENGA provides a tool to translate the source code from CUDA to HIP. The HIP version can run on AMD and on NVIDIA GPUs.
The translation tool is located in the HIP directory.

Run::

	python3 GengaHIP.py

to translate the code. GengaHIP will copy the translated source code to the HIP directory. 

Type::

	make

to compile GENGA with HIP. 


..
	Compile GENGA on Windows
	^^^^^^^^^^^^^^^^^^^^^^^^

	Gygwin setup search and install make


	If using Cygwin on Windows, then GENGA can be compiled the same way
	with::

		make SM=xx. 


	If using the Windows Command Prompt, type::

		nmake -f MakefileW SM=xx.

	Note, that the Windows C++ compiler ``cl``
	must be installed, and the compiler path must be loaded in the shell. If
	this is not the case, it can be loaded similar to this command::

		call "C:\Program Files (x86)\Microsoft Visual Studio 14.0\Common7\Tools\vsvars32.bat"
	, where the exact path and file name must eventually be changed.



