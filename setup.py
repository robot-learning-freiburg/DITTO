"""Install script for setuptools."""

# Try catkin install
try:
    from catkin_pkg.python_setup import generate_distutils_setup
    from distutils.core import setup

    d = generate_distutils_setup(packages=["DITTO"], package_dir={"": "."})

    setup(**d)

except:
    import setuptools
    from os import path

    # read the contents of your README file
    this_directory = path.abspath(path.dirname(__file__))
    with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
        long_description = f.read()

    setuptools.setup(
        name="DITTO",
        version="0.0.1",
        author="Nick Heppert",
        author_email="heppert@cs.uni-freiburg.de",
        install_requires=[
            # "matplotlib",
            # "numpy",
            # "opencv-python",
            # "Pillow",
            # "pymeshlab",
            # "scikit-image",
            # "scipy",
            # "torch",
            # "torchvision",
            # "tqdm",
        ],
        description="DITTO Package",
        long_description=long_description,
        long_description_content_type="text/markdown",
        # license="MIT",
        # packages=["src/DITTO"], # setuptools.find_packages(),
        packages=setuptools.find_packages(),
        # classifiers=[
        #     "Programming Language :: Python :: 3",
        #     "License :: OSI Approved :: MIT License",
        #     "Operating System :: OS Independent",
        # ],
        python_requires=">=3.7",
    )
