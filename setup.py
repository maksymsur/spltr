import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="spltr",
    version="0.3.2",
    author="Maksym Surzhynskyi",
    author_email="maksym.surzhynskyi@gmail.com",
    description="A simple PyTorch-based data loader and splitter",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/maksymsur/spltr",
    packages=setuptools.find_packages(),
    keywords = ['PyTorch', 'Data loader', 'Data splitter', 'DataLoader', 'random_split', 'train test', 'train test validation split', 'data preprocessing', 'pytorch dataset split'],
    python_requires='>=3.6',
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Intended Audience :: Science/Research",        
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
