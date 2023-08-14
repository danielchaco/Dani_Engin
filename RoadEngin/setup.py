from setuptools import setup, find_packages

setup(
    name="RoadEngin",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[        
        # List your project dependencies here, e.g.
        # "numpy>=1.0",
        # "pandas>=1.0",
    ],
    entry_points={
        "console_scripts": [
            # If your project has command-line scripts, add their entry points here, e.g.
            # "my_script=my_package.my_module:main",
        ],
    },
    python_requires=">=3.6",
    # Add metadata about your project
    author="Daniel C, & Felipe R",
    author_email="daniel.chacon@upr.edu",
    description="Postprocessing tools for roadway assessment and disstress analysis.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    license="GPL-3.0 License",
    url="https://github.com/danielchaco/Dani_Engin/tree/master/RoadEngin",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: GPL-3.0 License",
        "Programming Language :: Python :: 3",
    ],
)