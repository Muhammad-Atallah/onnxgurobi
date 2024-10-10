from setuptools import setup, find_packages

setup(
    name='onnx-to-gurobi',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',  # List your dependencies here
    ],
    author='Muhammad Atallah',
    description='A Python library that creates Gurobi models for neural networks in ONNX format',
    url='https://github.com/Muhammad-Atallah/onnx-to-gurobi',
)
