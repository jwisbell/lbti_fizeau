from setuptools import setup, find_packages

setup(
    name="lbti_fizeau",
    version="0.1.0",
    description="LBTI‐Fizeau data reduction and analysis tools",
    author="Jacob Isbell",
    author_email="jwisbell@arizona.edu",
    url="https://github.com/jwisbell/lbti_fizeau",
    packages=find_packages(exclude=["tests", "docs"]),
    install_requires=[
        "astropy",
        "scipy",
        "pandas",
        "numpy",
        "matplotlib",
        "scikit-image",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    entry_points={},
)
