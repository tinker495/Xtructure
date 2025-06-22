import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="xtructure",
    version="0.0.19",
    author="tinker495",
    author_email="wjdrbtjr495@gmail.com",
    description="JAX-optimized data structures",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tinker495/Xtructure",
    packages=setuptools.find_packages(),
    install_requires=[
        "jax[cuda]>=0.4.0",
        "chex>=0.1.0",
        "tabulate>=0.9.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
)
