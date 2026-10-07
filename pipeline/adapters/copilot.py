import subprocess
from pathlib import Path
from .base_adapter import BaseAdapter
from utils.utils import copilot_parse_stats

class CopilotAdapter(BaseAdapter):
    def __init__(self, spec_path: Path, output_dir: Path, model: str = "auto"):
        super().__init__(spec_path, output_dir)
        self.model = model
        self.usage = {}


    def run(self):
        step_log = self.logs_dir / "01-generate.log"
        def log(msg):
            with step_log.open("a", encoding="utf-8") as f:
                f.write(msg + "\n")

        log(f"[CopilotAdapter] Iniciando runner com model={self.model} spec={self.spec_file}")
        
        self.check_spec()
        spec_content = self.spec_file.read_text(encoding="utf-8")
        log(f"[CopilotAdapter] Spec carregado: {self.spec_file.resolve()}")

        content = f"Implement the following instructions in the current directory:\n\n{spec_content}"
        cmd = ["copilot", "--allow-all-tools", "--model", self.model, "-p", content]
        
        log(f"[CopilotAdapter] Comando a ser executado: {cmd}")
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self.output_dir,
                text=True,
                shell=False
            )
            log("[CopilotAdapter] Processo iniciado, aguardando saída...")
            stdout, stderr = process.communicate()
            log("[CopilotAdapter] Processo finalizado.")
        except Exception as e:
            log(f"[CopilotAdapter] Erro ao executar subprocess: {e}")
            raise

        (self.logs_dir / "session.log").write_text(stdout or "", encoding="utf-8")
        (self.logs_dir / "credits.log").write_text(stderr or "", encoding="utf-8")
        log("[CopilotAdapter] Logs salvos em session.log e credits.log.")

        full_log = stdout + "\n" + stderr
        self.usage = copilot_parse_stats(full_log, self.model)

        return str(self.output_dir)