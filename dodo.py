"""
doit configuration for BCDI
"""

import coverage
import os


def task_black():
    """
    Run black against the package.
    """
    return {
        "actions": [f"python -m black {os.path.dirname(os.path.abspath(__file__))}"],
        "verbosity": 2,
    }


def task_coverage_xml():
    """
    Generate an XML version of the coverage report. It can be opened with Notepad++.
    """

    def create_coverage_xml(coverage_file, output_file):
        print("XML coverage report generated in test_output/")
        cov = coverage.Coverage(data_file=coverage_file)
        cov.load()
        cov.xml_report(outfile=output_file)

    return {
        "actions": [
            (create_coverage_xml, [".coverage", "test_output/coverage-report.xml"])
        ],
        "file_dep": [".coverage"],
        "targets": ["test-output/coverage-report.xml"],
        "verbosity": 2,
    }


def task_tests():
    """
    Run unit tests with coverage.
    """
    return {
        "actions": [f"coverage run --source=bcdi -m unittest discover"],
        "targets": [".coverage", "test_output/unit-test-report.xml"],
        "verbosity": 2,
    }
