from pathlib import Path
import sys


class PathConfig:
    """
    Manage the project path config
    
    """
    def __init__(self):
        # acquire the root of project (assume the config.py exists in the root)
        self.project_root = Path(__file__).parent

        # define
        self.figures_dir = self.project_root / "figures"
        self.notebooks_dir = self.project_root / "notebooks"
        self.tools_dir = self.project_root / "tools"
        self.data_dir = self.project_root / "data"

        # external data
        self.shared_nvme = self.project_root.parent / "shared-nvme"
        
        # make sure all the dirs exsit
        self.figures_dir.mkdir(exist_ok=True)
        self.data_dir.mkdir(exist_ok=True)

        # add to the python path
        if str(self.tools_dir) not in sys.path:
            sys.path.insert(0, str(self.tools_dir))
        
    def get_trajectory_path(self, filename):

        return self.shared_nvme / filename

    def get_figure_path(self, filename):
        
        return self.figures_dir / filename

    def get_data_path(self, filename):
        
        return self.data_dir / filename

paths = PathConfig()