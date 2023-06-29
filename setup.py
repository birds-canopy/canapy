import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name="canapy",
    version="0.1.0",
    author="Nathan Trouvain",
    description="Semi automatic annotation tool for canary vocalizations.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
    install_requires=[
        "librosa>=0.8.1",
        "absl-py>=0.9.0",
        "matplotlib>=3.3.0",
        "bokeh>=2.2.3",
        "seaborn>=0.10.0",
        "panel>=0.10.2",
        "jupyter>=1.0.0",
        "joblib>=0.16.0",
        "numpy>=1.19.0",
        "scipy>=1.5.1",
        "scikit-learn>=0.23.1",
        "tqdm>=4.48.0",
        "pandas>=1.0.5",
        "SoundFile>=0.10.3.post1",
        "dill>=0.3.2",
        "reservoirpy>=0.3.5",
    ],
)
