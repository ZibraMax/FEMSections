import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="FEMSections",
    version="1.0.0",
    author="Arturo Rodriguezh",
    author_email="davidarturo9911@hotmail.com",
    description="Paquete para elementos finitos en python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ZibraMax/FEMSections",
    packages=setuptools.find_packages(),
    install_requires=[
          'matplotlib',
          'numpy',
          'triangle',
          'IPython',
      ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)