import setuptools
import os

# Get requirements
lib_folder = os.path.dirname(os.path.realpath(__file__))
requirementPath = lib_folder + '/requirements.txt'
install_requires = []
if os.path.isfile(requirementPath):
    with open(requirementPath) as f:
        install_requires = f.read().splitlines()

with open("README.rst", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="IterativeNelderMead",
    version="0.0.1",
    author="Bryson Cale",
    author_email="bryson.cale1@gmail.com",
    description="Iterative Nelder Mead Optimizer",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    packages = setuptools.find_packages(),
    include_package_data=True,
    install_requires=install_requires,
    url="https://github.com/astrobc1/IterativeNelderMeadPython/",
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "Operating System :: Unix"
    ],
    python_requires='>=3.8'
)
