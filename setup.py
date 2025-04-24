from setuptools import setup, find_packages, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext

# Define the C++ extension module

ext_modules = [
    Pybind11Extension(
        "ViruLink.random_walk.biased_random_walk",
        ["ViruLink/random_walk/biased_random_walk.cpp"]),
    Pybind11Extension(
        "ViruLink.hypergeom.hypergeom",
        ["ViruLink/hypergeom/hypergeom.cpp"],
        extra_compile_args=["-fopenmp"],
        extra_link_args=["-fopenmp"]),
    # >>> NEW EXTENSION <<<
    Pybind11Extension(
        "ViruLink.relations.relationship_edges_cpp",
        ["ViruLink/relations/relationship_edges.cpp"],
        extra_compile_args=["-O3", "-march=native", "-fopenmp"],
        extra_link_args=["-fopenmp"]),
    Pybind11Extension(
        "ViruLink.sampler.triangle_sampler_cpp",
        ["ViruLink/sampler/triangle_sampler.cpp"],
        extra_compile_args=["-O3", "-march=native", "-fopenmp"],
        extra_link_args=["-fopenmp"]),
]


# Package setup
setup(
    name="ViruLink",
    version="0.0.0",
    description="A program to classify Caudoviricetes",
    author="Muhammad Sulman",
    author_email="sulmanmu40@gmail.com",
    packages=find_packages(),
    include_package_data=True,
    setup_requires=['torch'],
    install_requires=[
        "pandas>=2.0.0",
        "biopython>=1.84",
        "tqdm>=2.0.0",
        "matplotlib>=3.0.0",
        "numpy>=1.21.0",
        "torch-geometric>=2.3.0",
        "scikit-learn>=1.0.0",
        "torch-sparse>=0.6.15",
        "torch-cluster>=1.6.0",
        "torch-spline-conv>=1.2.1",
        "gdown",
    ],
    python_requires=">=3.10",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    entry_points={
        "console_scripts": [
            "ViruLink=ViruLink.main:main",
        ],
    },
)
