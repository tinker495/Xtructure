import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Xtructure",
    version="0.0.1",
    author="Your Name",  # Please change this
    author_email="your.email@example.com",  # Please change this
    description="JAX-optimized data structures",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/Xtructure",  # Please change this
    packages=setuptools.find_packages(),
    install_requires=[
        "jax>=0.4.0",
        "jaxlib>=0.4.0", # jaxlib is often a direct dependency of jax but good to specify
        "chex>=0.1.0",
    ],
    extras_require={
        'dev': [
            'pytest>=7.0.0',
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # Choose an appropriate license
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Framework :: JAX",
    ],
    python_requires=">=3.8",
) 