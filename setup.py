from setuptools import find_packages, setup  # pragma: no cover

setup(  # pragma: no cover
    name="cmc_obs",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    author="Peter Smith",
    email="smith.peter.902@gmail.com",
    license="MIT",
    description="Create realistic mock observations from CMC models",
    copyright="Copyright 2023 Peter Smith",
    python_requires=">=3.9",
    version="0.1.0",
    include_package_data=True,
)
