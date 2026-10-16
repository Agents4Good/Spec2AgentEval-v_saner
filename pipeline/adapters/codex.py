from pathlib import Path
from .base_adapter import BaseAdapter


class CodexAdapter(BaseAdapter):
    def __init__(self, spec_path: Path, output_dir: Path, model: str = "gpt-6-luna"):
        super().__init__(spec_path, output_dir)
        self.model = model
        self.usage = {}

    def run(self):
        self.check_spec()
        spec_content = self.spec_file.read_text(encoding="utf-8")

        content = f"Implement the following instructions in the current directory:\n\n{spec_content}"
        cmd = [
            "codex",
            "exec",
            "--json",
            "--yolo",
            "--model", self.model,
            content,
        ]

        self._collect(cmd, cli="codex", model=self.model, tag="CodexAdapter")

        return str(self.output_dir)
