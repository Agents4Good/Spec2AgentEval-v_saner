from pathlib import Path
import sys


class BaseAdapter:
    def __init__(self, spec_path: Path, output_dir: Path):
        self.spec_file = Path(spec_path)
        self.output_dir = Path(output_dir)
        self.logs_dir = self.output_dir / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    def check_spec(self):
        if not self.spec_file.exists():
            raise FileNotFoundError(f"spec.md not found: {self.spec_file}")

    def run(self):
        raise NotImplementedError