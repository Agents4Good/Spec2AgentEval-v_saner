from pathlib import Path

from .base_adapter import BaseAdapter


class OpenCodeAdapter(BaseAdapter):
    def __init__(
        self,
        spec_path: Path,
        output_dir: Path,
        model: str = "opencode/longcat-2.5-preview-free",
    ):
        super().__init__(spec_path, output_dir)
        self.model = model
        # Host working directory for the canonical command. The local
        # default is resolved once here, so later argv construction must
        # use it literally: stage 01 replaces it with /out for Docker,
        # which must never be resolved against the host filesystem.
        self.execution_dir = Path(output_dir).resolve()
        self.usage = {}

    def run(self):
        self.check_spec()
        spec_content = self.spec_file.read_text(encoding="utf-8")

        content = (
            "Implement the following instructions in the current directory:\n\n"
            f"{spec_content}"
        )

        cmd = [
            "opencode",
            "run",
            content,
            "--model",
            self.model,
            "--format",
            "json",
            "--auto",
            str(self.execution_dir),
        ]

        self._collect(cmd, cli="opencode", model=self.model, tag="OpenCodeAdapter")

        return str(self.output_dir)
