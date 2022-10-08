from setuptools import setup, find_packages
from os.path import abspath, dirname, join

# Credit: https://dev.to/arnu515/create-a-pypi-pip-package-test-it-and-publish-it-using-github-actions-part-1-3cp8
# Fetches the content from README.md
# This will be used for the "long_description" field.
README_MD = open(join(dirname(abspath(__file__)), "README.md")).read()

setup(
    # The name of your project that we discussed earlier.
    # This name will decide what users will type when they install your package.
    # In my case it will be:
    # pip install pydash-arnu515
    # This field is REQUIRED
    name="seqchromloader",

    # The version of your project.
    # Usually, it would be in the form of:
    # major.minor.patch
    # eg: 1.0.0, 1.0.1, 3.0.2, 5.0-beta, etc.
    # You CANNOT upload two versions of your package with the same version number
    # This field is REQUIRED
    version="0.2.3",

    # The packages that constitute your project.
    # For my project, I have only one - "pydash".
    # Either you could write the name of the package, or
    # alternatively use setuptools.findpackages()
    #
    # If you only have one file, instead of a package,
    # you can instead use the py_modules field instead.
    # EITHER py_modules OR packages should be present.
    packages=find_packages(exclude='tests'),

    # dependencies
    install_requires=[
        'numpy',
        'pandas',
        'webdataset>=0.2.0',
        'pyfasta>=0.5.0',
        'pybedtools>=0.9.0',
        'pysam>=0.19.0',
        'pybigwig>=0.3.0',
        'torch>=1.10.0',
        'pytorch-lightning',
    ],

    # The description that will be shown on PyPI.
    # Keep it short and concise
    # This field is OPTIONAL
    description="Sequence and chromatin dataloader for deep learning",

    # The content that will be shown on your project page.
    # In this case, we're displaying whatever is there in our README.md file
    # This field is OPTIONAL
    long_description=README_MD,

    # Now, we'll tell PyPI what language our README file is in.
    # In my case it is in Markdown, so I'll write "text/markdown"
    # Some people use reStructuredText instead, so you should write "text/x-rst"
    # If your README is just a text file, you have to write "text/plain"
    # This field is OPTIONAL
    long_description_content_type="text/markdown",

    # The url field should contain a link to a git repository, the project's website
    # or the project's documentation. I'll leave a link to this project's Github repository.
    # This field is OPTIONAL
    url="https://github.com/yztxwd/seqchromloader",

    # The author name and email fields are self explanatory.
    # These fields are OPTIONAL
    author_name="Jianyu Yang",
    author_email="yztxwd@gmail.com",

    # Classifiers help categorize your project.
    # For a complete list of classifiers, visit:
    # https://pypi.org/classifiers
    # This is OPTIONAL
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Development Status :: 2 - Pre-Alpha",
        "Programming Language :: Python :: 3 :: Only"
    ],

    # Keywords are tags that identify your project and help searching for it
    # This field is OPTIONAL
    keywords="dataloder, pytorch, webdataset",

    # For additional fields, check:
    # https://github.com/pypa/sampleproject/blob/master/setup.py
)
