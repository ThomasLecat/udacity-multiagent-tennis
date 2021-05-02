from distutils.core import setup
from typing import List


def read_requirements(file: str) -> List[str]:
    """Returns content of given requirements file"""
    return [
        line
        for line in open(file)
        if not (line.startswith("#") or line.startswith("--"))
    ]


setup(
    name="Multiagent RL",
    version="0.1",
    description="Third project of the Udacity Deep Reinforcement Learning program.",
    author="Thomas LECAT",
    install_requires=read_requirements("requirements.txt"),
    packages=["multiagent"],
)
