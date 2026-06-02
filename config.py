from pathlib import Path
import sys


class PathConfig:
    """
    Manage project paths from the repository root.

    Notebooks can import this module and use ``paths`` to avoid saving
    generated files back into the notebooks directory.
    """
    def __init__(self):
        # Assume config.py lives in the repository root.
        self.project_root = Path(__file__).parent

        self.figures_dir = self.project_root / "figures"
        self.notebooks_dir = self.project_root / "notebooks"
        self.tools_dir = self.project_root / "tools"
        self.data_dir = self.project_root / "data"
        self.results_dir = self.project_root / "results"

        # external data
        self.shared_nvme = self.project_root.parent / "shared-nvme"
        
        # Make sure standard output directories exist.
        self.figures_dir.mkdir(exist_ok=True)
        self.data_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)

        # Add the repository root so notebooks can import tools.* reliably.
        if str(self.project_root) not in sys.path:
            sys.path.insert(0, str(self.project_root))
        if str(self.tools_dir) not in sys.path:
            sys.path.insert(0, str(self.tools_dir))
        
    def get_trajectory_path(self, filename):

        return self.shared_nvme / filename

    def get_figure_path(self, filename):
        
        return self.figures_dir / filename

    def get_data_path(self, filename):
        
        return self.data_dir / filename

    def get_result_path(self, filename):

        return self.results_dir / filename

paths = PathConfig()
