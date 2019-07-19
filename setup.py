import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Spltr-Maksym",
    version="0.1",
    author="Maksym Surzhynskyi",
    author_email="maksym.surzhynskyi@gmail.com",
    description="A simple PyTorch-based data loader and splitter.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/maksymsur/Spltr",
    packages=setuptools.find_packages(),
    keywords = ['PyTorch', 'Data loader', 'Data splitter', 'DataLoader'],
    classifiers=[
        "Programming Language :: Python :: 3.6+",
        "Intended Audience :: Science / Research / ML Professionals",
        "Topic :: Deep Learning / Data Preparation",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
