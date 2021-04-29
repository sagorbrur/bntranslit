import codecs
import setuptools


setuptools.setup(
    name="bntranslit",
    version="2.0.0",
    author="Sagor Sarker",
    author_email="sagorhem3532@gmail.com",
    description="BNTRANSLIT is a deep learning based transliteration app for Bangla word",
    long_description=codecs.open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/sagorbrur/bntranslit",
    license="MIT",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        "numpy",
        "wasabi"
    ],
)
