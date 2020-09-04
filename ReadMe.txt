Hi!
Run the Notebook 'RunMe.ipynb' for a guide on how to use the model. 
Some useful informations can also be found in section 3 - User Manual of my internship report. 

This version of the model uses some Python code that have been 'Cythonized'. In order to use this model, you'll need to install Cython (pip install Cython or conda install -c anaconda cython in Anaconda). 
Once Cython is installed, run the following command in the folder where the model is:
python setup.py build_ext --inplace

TO do so, you might need to download Visual Studio Build Tools and install:
    Visual C++ Build tools core features.
    VC++ 2017 v141 toolset (x86,x64)
    Visual C++ 2017 Redistributable Update
    Windows 10 SDK (10.0.16299.0) for Desktop C++

That's it! Cython should have created a 'build' folder, as well as some .c files and some ugly cpxx-xx_xxx.pyd files.
Simulations can then be made using the Notebook.

A clean model can always be cloned from my github:
https://github.com/axelroques/ and search for 2DModel-Roques2020

If you have difficulties getting the model to work, don't hesitate to contact me at roquesaxel98@gmail.com or axel.roques@espci.fr, I will be happy to help :)
- Axel