import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="FishLSS",
    version="1.0",
    author="Noah Sailer",
    author_email="nsailer@berkeley.edu",
    description="Fisher forecasting code for Large Scale Structure surveys using Lagrangian Perturbation Theory.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NoahSailer/FishLSS",
    packages=['FishLSS','FishLSS/input',\
              'FishLSS/bao_recon'],
    package_data={'FishLSS': ['*.txt','*.dat','*.md'],'FishLSS/input': ['*.txt','*.dat','*.md']},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=['numpy','scipy','pyfftw'],
)
