"""doit configuration for BCDI."""

import os
import shutil
from pathlib import Path

import coverage

from bcdi import __version__

# Generic functions go here


def get_path():
    """Get the path of the dodo.py file."""
    return os.path.dirname(os.path.abspath(__file__))


# Tasks go here


def task_black():
    """Run black against the package."""
    path = get_path()
    return {
        "actions": [f"python -m black --line-length=88 {path}"],
        "verbosity": 2,
    }


def task_isort():
    """Run isort against the package."""
    path = get_path()
    return {
        "actions": [f"python -m isort {path}"],
        "verbosity": 2,
    }


def task_mypy():
    """Run mypy against the package."""
    path = get_path()
    return {
        "actions": [f"python -m mypy {path + '/bcdi'}"],
        "verbosity": 2,
    }


def task_clean_dist():
    """Remove the build directory and its content."""

    def delete_dir(dirname):
        """Delete the directory if it exists."""
        path = os.path.join(get_path(), dirname).replace("\\", "/")
        if os.path.isdir(path):
            shutil.rmtree(path)
            print(f"\n\tDeleted {path}\n")
        else:
            print("\n\tNo build directory to delete.\n")

    return {
        "actions": [(delete_dir, ["dist/"])],
        "verbosity": 2,
    }


def task_clean_doc():
    """Remove the compiled documentation."""

    def delete_dir(dirname):
        """Delete the directory if it exists."""
        path = os.path.join(get_path(), dirname).replace("\\", "/")
        if os.path.isdir(path):
            shutil.rmtree(path)
            print(f"\n\tDeleted {path}\n")
        else:
            print("\n\tNo built documentation to delete.\n")

    return {
        "actions": [(delete_dir, ["doc/doc_html/"])],
        "verbosity": 2,
    }


def task_coverage_xml():
    """
    Generate an XML version of the coverage report.

    It can be opened with Notepad++.
    """

    def create_coverage_xml(coverage_file, output_file):
        """Create an XML report for the coverage."""
        print("\n\tXML coverage report generated in 'test_output/'\n")
        cov = coverage.Coverage(data_file=coverage_file)
        cov.load()
        cov.xml_report(outfile=output_file)

    return {
        "actions": [
            (create_coverage_xml, [".coverage", "test_output/coverage-report.xml"])
        ],
        "file_dep": [".coverage"],
        "targets": ["test_output/coverage-report.xml"],
        "verbosity": 2,
    }


def task_clean_coverage():
    """Delete the coverage report."""

    def delete_coverage(filename):
        """Delete the file if it exists."""
        path = os.path.join(get_path(), filename).replace("\\", "/")
        if os.path.isfile(path):
            os.unlink(path)
            print(f"\n\tDeleted {path}\n")
        else:
            print("\n\tNo coverage file to delete.\n")

    return {
        "actions": [(delete_coverage, [".coverage"])],
        "file_dep": ["test_output/coverage-report.xml"],
        "verbosity": 2,
    }


def task_ruff():
    """Run ruff on the modules."""
    return {
        "actions": ["ruff check ."],
        "verbosity": 2,
    }


def task_docstrings():
    """Run pydocstyle on the modules."""
    return {
        "actions": ["pydocstyle bcdi --ignore=D102,D107,D212,D203"],
        "verbosity": 2,
    }


def task_tests():
    """Run unit tests with coverage."""
    return {
        "actions": ["coverage run --source=bcdi -m unittest discover"],
        "targets": [".coverage"],
        "verbosity": 1,
    }


def task_check_links_doc():
    """Check external links in the doc using sphinx."""
    sourcedir = Path(get_path()) / "doc"
    outputdir = sourcedir / "doc_html"
    return {
        "actions": [f"sphinx-build -b linkcheck {sourcedir} {outputdir}"],
        "verbosity": 1,
    }


def task_build_doc():
    """Build the documentation with sphinx."""
    sourcedir = Path(get_path()) / "doc"
    outputdir = sourcedir / "doc_html"
    return {
        "actions": [f"sphinx-build -b html {sourcedir} {outputdir}"],
        "targets": ["docs/"],
        "verbosity": 2,
    }


def task_build_distribution():
    """Build the distribution."""
    return {
        "actions": ["poetry build"],
        "verbosity": 2,
    }


def task_clean_build():
    """Remove the build directory"""

    def delete_dir(dirname):
        """Delete the directory if it exists."""
        path = os.path.join(get_path(), dirname).replace("\\", "/")
        if os.path.isdir(path):
            shutil.rmtree(path)
            print(f"\n\tDeleted {path}\n")
        else:
            print("\n\tNo build directory to delete.\n")

    return {
        "actions": [(delete_dir, ["build/"])],
        "verbosity": 2,
    }


def task_check_long_description_pypi():
    """Check whether the long description will render correctly on PyPI."""
    return {
        "actions": ["twine check dist/*"],
        "file_dep": [f"dist/bcdi-{__version__}.tar.gz"],
        "verbosity": 2,
    }
