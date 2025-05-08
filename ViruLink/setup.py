from setuptools import setup, find_packages, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext

# Define the C++ extension module


ext_modules = [

    # 1. biased_random_walk  (keep OpenMP so it stays multi-core)
    Pybind11Extension(
        "ViruLink.random_walk.biased_random_walk",
        ["ViruLink/random_walk/biased_random_walk.cpp"],
        extra_compile_args=["-O3", "-march=native", "-fopenmp"],
        extra_link_args=["-fopenmp"],
    ),

    # 2. hypergeom
    Pybind11Extension(
        "ViruLink.hypergeom.hypergeom",
        ["ViruLink/hypergeom/hypergeom.cpp"],
        extra_compile_args=["-O3", "-march=native", "-fopenmp"],
        extra_link_args=["-fopenmp"],
    ),

    # 3. relationship_edges_cpp
    Pybind11Extension(
        "ViruLink.relations.relationship_edges_cpp",
        ["ViruLink/relations/relationship_edges.cpp"],
        extra_compile_args=["-O3", "-march=native", "-fopenmp"],
        extra_link_args=["-fopenmp"],
    ),

    # 4. triangle_sampler  ← now parallel too
    Pybind11Extension(
        "ViruLink.sampler.triangle_sampler",
        ["ViruLink/sampler/triangle_sampler.cpp"],
    ),
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
        "scikit-learn>=1.0.0",
        "gdown",
        "networkx",
        "glob2",
        "gensim>=4.3.3"
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
