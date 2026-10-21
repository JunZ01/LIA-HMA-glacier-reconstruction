from setuptools import setup, find_packages

setup(
    name="paleoice",
    version="0.1.0",
    description="LIA冰川冰厚与ELA重建工具包",
    author="Jun",
    author_email="",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "geopandas>=0.12.0",
        "rasterio>=1.3.0",
        "shapely>=2.0.0",
        "numpy>=1.21.0",
        "scipy>=1.9.0",
        "pandas>=1.5.0",
        "xarray>=2022.6.0",
        "tqdm>=4.64.0",
    ],
    extras_require={
        "oggm": ["oggm>=1.6.0"],
    },
    entry_points={
        "console_scripts": [
            "paleoice-prepare=paleoice.prepare:main",
            "paleoice-flowline=paleoice.flowline:main",
        ],
    },
)
