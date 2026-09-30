import json
import subprocess
from pathlib import Path
from .base_adapter import BaseAdapter
from utils.utils import claude_parse_stats


class ClaudeCodeAdapter(BaseAdapter):
    def __init__(self, spec_path: Path, output_dir: Path, model: str = "claude-sonnet-4-5"):
        super().__init__(spec_path, output_dir)
        self.model = model
        self.usage = {}

    def run(self):
        step_log = self.logs_dir / "01-generate.log"

        def log(msg):
            with step_log.open("a", encoding="utf-8") as f:
                f.write(msg + "\n")

        log(f"[ClaudeCodeAdapter] Iniciando runner com model={self.model} spec={self.spec_file}")

        self.check_spec()
        spec_content = self.spec_file.read_text(encoding="utf-8")
        log(f"[ClaudeCodeAdapter] Spec carregado: {self.spec_file.resolve()}")

        content = f"Implement the following instructions in the current directory:\n\n{spec_content}"
        cmd = [
            "claude",
            "--dangerously-skip-permissions",
            "--model", self.model,
            "-p", content,
            "--output-format", "json",
        ]

        log(f"[ClaudeCodeAdapter] Comando a ser executado: {cmd}")
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self.output_dir,
                text=True,
                shell=False,
            )
            log("[ClaudeCodeAdapter] Processo iniciado, aguardando saída...")
            stdout, stderr = process.communicate()
            log("[ClaudeCodeAdapter] Processo finalizado.")
        except Exception as e:
            log(f"[ClaudeCodeAdapter] Erro ao executar subprocess: {e}")
            raise
            
        try:
            data = json.loads(stdout)

            response_text = data.get("result", stdout) 
            (self.logs_dir / "session.log").write_text(response_text, encoding="utf-8")
            
            # 3. Salva o JSON completo com os dados financeiros no credits.log
            credits_text = json.dumps(data, indent=2, ensure_ascii=False)
            (self.logs_dir / "credits.log").write_text(credits_text, encoding="utf-8")
            
            # 4. Envia o dicionário JSON para o parser
            self.usage = claude_parse_stats(data, self.model)

        except json.JSONDecodeError:
            log("[ClaudeCodeAdapter] Aviso: Saída não foi JSON. Fazendo fallback para texto.")
            
            # Fallback caso a CLI falhe e cuspa texto puro
            (self.logs_dir / "session.log").write_text(stdout, encoding="utf-8")
            (self.logs_dir / "credits.log").write_text(stderr, encoding="utf-8")
            
            # Envia a string gigante pro parser fazer Regex (mesmo esquema dos anteriores)
            full_log = stdout + "\n" + stderr
            self.usage = claude_parse_stats(full_log, self.model)

        log(f"[ClaudeCodeAdapter] Stats capturados: {self.usage}")

        return str(self.output_dir)