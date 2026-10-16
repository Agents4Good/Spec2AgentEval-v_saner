from pathlib import Path
from .base_adapter import BaseAdapter


class ClaudeCodeAdapter(BaseAdapter):
    def __init__(self, spec_path: Path, output_dir: Path, model: str = "claude-sonnet-4-5", effort: str | None = None):
        super().__init__(spec_path, output_dir)
        self.model = model
        self.effort = effort
        self.usage = {}

    def run(self):
        self.check_spec()
        spec_content = self.spec_file.read_text(encoding="utf-8")

        content = f"Implement the following instructions in the current directory:\n\n{spec_content}"
        cmd = [
            "claude",
            "--dangerously-skip-permissions",
            "--model", self.model,
            "-p", content,
            "--output-format", "json",
        ]
        if self.effort:
            cmd.extend(["--effort", self.effort])

        self._collect(cmd, cli="claude", model=self.model, tag="ClaudeCodeAdapter")

        return str(self.output_dir)
