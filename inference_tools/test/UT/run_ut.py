import os

from msit.utils.log import logger
from msit.utils.toolkits import run_subprocess


class RunUT:
    def __init__(self):
        self.cur_dir = os.path.realpath(os.path.dirname(__file__))
        self.cov_dir = os.path.join(os.path.dirname(self.cur_dir), "../msit")
        self.report_dir = os.path.join(self.cur_dir, "report")
        self.cov_config_path = os.path.join(self.cur_dir, ".coveragerc")
        self.final_xml_path = os.path.join(self.report_dir, "final.xml")
        self.html_cov_report = os.path.join(self.report_dir, "htmlcov")
        self.xml_cov_report = os.path.join(self.report_dir, "coverage.xml")
        self.cmd = [
            "python3",
            "-m",
            "pytest",
            self.cur_dir,
            f"--junitxml={self.final_xml_path}",
            f"--cov-config={self.cov_config_path}",
            f"--cov={self.cov_dir}",
            "--cov-branch",
            f"--cov-report=html:{self.html_cov_report}",
            f"--cov-report=xml:{self.xml_cov_report}",
        ]

    def execute(self):
        run_subprocess(self.cmd, check_interval=0)
        logger.info("Unit tests executed successfully.")


if __name__ == "__main__":
    RunUT().execute()
