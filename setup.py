from setuptools import setup, find_packages

# Read requirements.txt, ignore comments
try:
    requires = list()
    f = open("requirements.txt", "rb")
    for line in f.read().decode("utf-8").split("\n"):
        line = line.strip()
        if "#" in line:
            line = line[: line.find("#")].strip()
        if line:
            requires.append(line)
except:
    print("'requirements.txt' not found!")
    requires = list()

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="tuneta",
    version="0.1.02",
    author="John Richardson",
    author_email="jmrichardson@gmail.com",
    # description="Optimize technical indicators for machine learning",
    description="tuneta",
    long_description="tuneta"
    # long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jmrichardson/tuneta",
    # keywords="technical analysis optimize tune indicators machine learning optuna",
    keywords="",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Development Status :: 4 - Beta",
        "Natural Language :: English",
        "Operating System :: OS Independent",
    ],
    install_requires=requires,
    python_requires='>=3.6',
    platform=["any"],
)