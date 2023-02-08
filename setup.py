import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="AqOrg",
    version="0.1.14",
    author="Grayson Boyer",
    author_email="gmboyer@asu.edu",
    description="Estimate thermodynamic properties of aqueous organic compounds",
    long_description=long_description,
    long_description_content_type="text/markdown",
    entry_points={},
    packages=['AqOrg'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=['chemparse',
                      'pubchempy',
                      'sigfig',
                      'pandas',
                      'rdkit',
                      'pgradd'],
    include_package_data=True,
    package_data={'': ['data/*.csv']},
    zip_safe=False
)

