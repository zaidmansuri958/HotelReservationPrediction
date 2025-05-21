from setuptools import setup,find_packages

with open("requirements.txt") as f:
    requirements=f.read().splitlines()

setup(
    name="HotelReservation",
    version="0.1",
    author="Zaid",
    packages=find_packages(),
    install_requires=requirements
)