import json
import subprocess
from pathlib import Path
from .base_adapter import BaseAdapter
from utils.utils import gemini_parse_stats

class GeminiAdapter(BaseAdapter):
    def __init__(self, spec_path: Path, output_dir: Path, model: str = "gemini-3.1-pro"):
        super().__init__(spec_path, output_dir)
        self.model = model
        self.usage = {}

    def run(self):
        step_log = self.logs_dir / "01_generate.log"

        def log(msg):
            with step_log.open("a", encoding="utf-8") as f:
                f.write(msg + "\n")

        self.check_spec()
        spec_content = self.spec_file.read_text(encoding="utf-8")
        log(f"[GeminiAdapter] Spec carregado: {self.spec_file.resolve()}")

        content = f"Implement the following instructions in the current directory:\n\n{spec_content}"
        cmd = [
            "agy",
            "--print", content,
            "--model", self.model,
            "--dangerously-skip-permissions",
            "--output-format", "json",
            "--effort", "low"
        ]

        log(f"[GeminiAdapter] Comando a ser executado: {cmd}")

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self.output_dir,
                text=True,
                shell=False,
            )

            log("[GeminiAdapter] Processo iniciado, aguardando saída...")
            stdout, stderr = process.communicate()
            log("[GeminiAdapter] Processo finalizado.")
        except Exception as e:
            log(f"[GeminiAdapter] Erro: {e}")
            raise

        (self.logs_dir / "session.log").write_text(stdout or "", encoding="utf-8")
        (self.logs_dir / "cli_stderr.log").write_text(stderr or "", encoding="utf-8")
        log("[GeminiAdapter] Logs salvos em session.log e cli_stderr.log.")

        
        try:
            usage_summary = gemini_parse_stats(stdout, self.model)
            (self.logs_dir / "credits.log").write_text(str(usage_summary), encoding="utf-8")
            log("[GeminiAdapter] Log stats salvos em credits.log.")

            self.usage = usage_summary
        except json.JSONDecodeError:
            log("[GeminiAdapter] Erro ao processar JSON de saída.")
            
        return str(self.output_dir)