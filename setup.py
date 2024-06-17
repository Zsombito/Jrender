from setuptools import setup, find_packages


setup(
    name='jrenderer',
    version='1.0',
    packages=find_packages(),
    license="Apache License",
    author="Zsombor Klapper",
    install_requires=[
        "jax>=0.4.28",
        "brax>=0.10.4"
    ]
)