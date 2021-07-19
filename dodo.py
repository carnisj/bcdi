"""
doit configuration for BCDI
"""


import coverage
import os
import unittest


def task_black():
    print(
        "Running black against the folder "
        f"{os.path.dirname(os.path.abspath(__file__))}\n"
    )
    return {
        "actions": [f"python -m black {os.path.dirname(os.path.abspath(__file__))}"],
    }
