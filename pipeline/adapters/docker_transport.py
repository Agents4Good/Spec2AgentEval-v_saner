"""Docker transport for stage-01 generation (phase 3, lane B).

Builds the ``docker run`` invocation for a canonical CLI argv, stages
spec+prompt under a private ``/in``, provisions exactly the selected CLI's
credential mode (auth-policy.md), and removes all per-job temporaries on
success and on setup failure.

Stdlib only; never imports BaseAdapter (acyclic: BaseAdapter may import
``DockerTransportError`` and catch it separately). No dotenv loading here;
configuration is read from the process environment at ``prepare`` time.

Diagnostics carry safe tokens only: no credential values, no raw
exception text, and no CLI stdout. Image inspect/build output is discarded;
successful setup logs contain metadata/status notes only.
"""
import os
import stat
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path

GEN_IMAGE_DEFAULT = "spec2agent-gen:infra"

SUPPORTED_CLIS = ("claude", "codex", "opencode")

ERROR_CODES = (
    "docker_unavailable",
    "docker_build_failed",
    "docker_setup_failed",
    "docker_auth_invalid",
    "docker_persistence_failed",
    "docker_cleanup_failed",
)

# Exact credential allowlist (auth-policy.md). Env mode forwards the NAME
# only via ``-e NAME``; file mode snapshots one selected file.
CLAUDE_ENV_NAMES = (
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_API_KEY",
    "CLAUDE_CODE_OAUTH_TOKEN",
)
CODEX_ENV_NAMES = (
    "CODEX_API_KEY",
    "CODEX_ACCESS_TOKEN",
)
OPENCODE_ENV_BY_MODEL_PREFIX = {
    "openai/": "OPENAI_API_KEY",
    "anthropic/": "ANTHROPIC_API_KEY",
    "google/": "GEMINI_API_KEY",
}

CONTAINER_HOME = "/home/agent"
SETUP_LOG_NAME = "docker_setup.log"


class DockerTransportError(Exception):
    """Typed setup failure carrying a safe code only (no payload).

    The transport itself raises only the agreed ``ERROR_CODES`` tokens;
    construction stays lenient so collectors can build and remap unknown
    codes through their own fallback (e.g. to ``docker_setup_failed``).
    ``str()`` exposes the code and nothing else: never exception text,
    credential values, or CLI output.
    """

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code

    def __str__(self) -> str:  # noqa: D105 (stable token, nothing else)
        return str(self.code)


def _default_temp_root() -> str:
    preferred = "/tmp/opencode"
    if os.path.isdir(preferred):
        return preferred
    return tempfile.gettempdir()


def _repo_docker_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "docker"


def _run_docker(args: list) -> tuple:
    """Run the docker CLI, capturing bytes; missing binary maps to unavailable."""
    try:
        proc = subprocess.run(
            ["docker", *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
        )
    except FileNotFoundError:
        raise DockerTransportError("docker_unavailable")
    except OSError:
        raise DockerTransportError("docker_unavailable")
    return proc.returncode, proc.stdout or b"", proc.stderr or b""


def _first_nonempty(names) -> str | None:
    for name in names:
        if os.environ.get(name):
            return name
    return None


def _model_from_argv(argv: list) -> str | None:
    try:
        idx = argv.index("--model")
    except ValueError:
        return None
    if idx + 1 < len(argv):
        return argv[idx + 1]
    return None


def _extract_prompt(cli: str, argv: list) -> str:
    """Locate the canonical prompt inside the adapter-built argv.

    Execution uses ``argv`` verbatim; this is only for faithful /in
    staging. Shapes mirror the canonical adapters; anything else fails
    loud with a setup error rather than staging a fabricated prompt.
    """
    try:
        if cli == "claude":
            return argv[argv.index("-p") + 1]
        if cli == "codex":
            return argv[-1]
        if cli == "opencode":
            return argv[argv.index("run") + 1]
    except (ValueError, IndexError):
        pass
    raise DockerTransportError("docker_setup_failed")


def _write_exclusive(path: Path, data: bytes) -> Path:
    """Create a file (or -N sibling) with O_EXCL; never truncate."""
    stem, suffix = path.stem, path.suffix
    for i in range(1000):
        cand = path if i == 0 else path.with_name(f"{stem}-{i}{suffix}")
        try:
            fd = os.open(str(cand), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError:
            continue
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
        except OSError:
            try:
                os.unlink(str(cand))
            except OSError:
                pass
            raise
        return cand
    raise DockerTransportError("docker_persistence_failed")


def _remove_owned_tree(path: Path) -> None:
    """Remove one private tree using directory-relative, no-follow operations.

    The CLI may remove owner access or leave symlinks. Restore access only
    on owned directories, never on symlink targets, then unlink all leaves.
    """
    def remove(parent_fd, name):
        try:
            info = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            return
        if not stat.S_ISDIR(info.st_mode):
            os.unlink(name, dir_fd=parent_fd)
            return
        if info.st_uid != os.getuid():
            raise PermissionError("unowned temporary directory")
        os.chmod(name, 0o700, dir_fd=parent_fd, follow_symlinks=False)
        fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                     dir_fd=parent_fd)
        try:
            for child in os.listdir(fd):
                remove(fd, child)
        finally:
            os.close(fd)
        os.rmdir(name, dir_fd=parent_fd)

    parent_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        remove(parent_fd, path.name)
    finally:
        os.close(parent_fd)


class DockerTransport:
    """Per-job Docker setup for one canonical CLI invocation."""

    def __init__(self, spec_file: Path):
        self.spec_file = Path(spec_file)
        self.log_paths: dict = {}

    def _cleanup(self, *dirs) -> None:
        failed = False
        for d in dirs:
            if d is None:
                continue
            try:
                _remove_owned_tree(d)
            except Exception:
                failed = True
        if failed:
            raise DockerTransportError("docker_cleanup_failed") from None

    def _ensure_image(self, image: str, notes: list) -> None:
        rc, _out, err = _run_docker(["image", "inspect", image])
        notes.append(f"inspect image exit={rc}")
        if rc == 0:
            return
        docker_dir = _repo_docker_dir()
        rc, _out, err = _run_docker([
            "build", "-f", str(docker_dir / "Dockerfile.gen"),
            "-t", image, str(docker_dir),
        ])
        notes.append(f"build image exit={rc}")
        if rc != 0:
            raise DockerTransportError("docker_build_failed")

    def _snapshot_file(self, src: Path, dest_dir: Path, name: str) -> Path:
        """Copy one credential file with 0600 into the private temp home."""
        dest_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(dest_dir, 0o700)
        dest = dest_dir / name
        dest.write_bytes(src.read_bytes())
        os.chmod(dest, 0o600)
        return dest

    def _provision_credentials(self, cli: str, argv: list, tmp_home: Path,
                               notes: list) -> dict:
        """Return {"env_names": [...], "container_config": ...|None}.

        Exactly one mode per CLI: named env forwarding (name only, never
        values in argv) or one selected file snapshot. Missing
        configuration runs credential-less; mismatched opt-in config is a
        typed auth error. No whole-home/config mounts, no duplicate binds.
        """
        home = Path.home()
        if cli == "claude":
            selected = _first_nonempty(CLAUDE_ENV_NAMES)
            src = Path(os.environ.get("CLAUDE_CONFIG_DIR",
                                      str(home / ".claude"))) / ".credentials.json"
            # The advertised config dir must always exist writable,
            # even when no credential file is provisioned.
            cfg_dir = tmp_home / ".claude"
            cfg_dir.mkdir(parents=True, exist_ok=True)
            os.chmod(cfg_dir, 0o700)
            if selected is not None:
                notes.append(f"credentials mode=env name={selected}")
                return {"env_names": [selected],
                        "container_config": f"{CONTAINER_HOME}/.claude"}
            if src.is_file():
                self._snapshot_file(src, cfg_dir, ".credentials.json")
                notes.append(f"credentials mode=file src={src}")
                return {"env_names": [],
                        "container_config": f"{CONTAINER_HOME}/.claude"}
            notes.append("credentials mode=none")
            return {"env_names": [],
                    "container_config": f"{CONTAINER_HOME}/.claude"}
        if cli == "codex":
            selected = _first_nonempty(CODEX_ENV_NAMES)
            src = Path(os.environ.get("CODEX_HOME",
                                      str(home / ".codex"))) / "auth.json"
            cfg_dir = tmp_home / ".codex"
            cfg_dir.mkdir(parents=True, exist_ok=True)
            os.chmod(cfg_dir, 0o700)
            if selected is not None:
                notes.append(f"credentials mode=env name={selected}")
                return {"env_names": [selected],
                        "container_config": f"{CONTAINER_HOME}/.codex"}
            if src.is_file():
                self._snapshot_file(src, cfg_dir, "auth.json")
                notes.append(f"credentials mode=file src={src}")
                return {"env_names": [],
                        "container_config": f"{CONTAINER_HOME}/.codex"}
            notes.append("credentials mode=none")
            return {"env_names": [],
                    "container_config": f"{CONTAINER_HOME}/.codex"}
        # opencode: no credentials by default; narrow matched opt-in only.
        wanted = os.environ.get("GEN_OPENCODE_API_KEY_ENV", "")
        if not wanted:
            notes.append("credentials mode=none")
            return {"env_names": [], "container_config": None}
        if wanted not in tuple(OPENCODE_ENV_BY_MODEL_PREFIX.values()):
            raise DockerTransportError("docker_auth_invalid")
        model = _model_from_argv(argv) or ""
        matched = any(model.startswith(prefix)
                      and name == wanted
                      for prefix, name in OPENCODE_ENV_BY_MODEL_PREFIX.items())
        if not matched or not os.environ.get(wanted):
            raise DockerTransportError("docker_auth_invalid")
        notes.append(f"credentials mode=env name={wanted}")
        return {"env_names": [wanted], "container_config": None}

    def _write_setup_log(self, output_dir: Path, notes: list, image: str,
                         cli: str, prompt_chars: int) -> Path:
        logs_dir = output_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        body = "".join(
            [f"image={image} cli={cli} prompt_chars={prompt_chars}\n"]
            + [line + "\n" for line in notes]
        )
        return _write_exclusive(logs_dir / SETUP_LOG_NAME, body.encode("utf-8"))

    @contextmanager
    def prepare(self, argv: list, *, cli: str, output_dir):
        """Stage per-job inputs and yield the final ``docker run`` argv.

        Yields ``["docker", "run", ..., IMAGE, *argv]`` with ``/in:ro``,
        ``/out:rw`` (the reserved host output), a private writable HOME,
        ``--user uid:gid`` and ``--workdir /out``. All temporaries are
        removed afterwards, including on setup failure (typed raise).
        """
        if cli not in SUPPORTED_CLIS:
            raise DockerTransportError("docker_setup_failed")
        if (not isinstance(argv, list) or not argv
                or not all(isinstance(a, str) for a in argv)):
            raise DockerTransportError("docker_setup_failed")
        output_dir = Path(output_dir)
        if not output_dir.is_dir():
            raise DockerTransportError("docker_setup_failed")
        if not self.spec_file.is_file():
            raise DockerTransportError("docker_setup_failed")
        prompt = _extract_prompt(cli, argv)
        image = os.environ.get("GEN_IMAGE", GEN_IMAGE_DEFAULT)
        if not image:
            raise DockerTransportError("docker_setup_failed")

        self.log_paths = {}
        tmp_in = tmp_home = None
        notes: list = []
        setup_complete = False
        try:
            self._ensure_image(image, notes)
            root = _default_temp_root()
            tmp_in = Path(tempfile.mkdtemp(prefix="gen-in-", dir=root))
            tmp_home = Path(tempfile.mkdtemp(prefix="gen-home-", dir=root))
            os.chmod(tmp_home, 0o700)
            (tmp_in / "spec.md").write_bytes(self.spec_file.read_bytes())
            (tmp_in / "prompt.txt").write_text(prompt, encoding="utf-8")
            cred = self._provision_credentials(cli, argv, tmp_home, notes)
            notes.append(f"in={tmp_in} out={output_dir} home={tmp_home}")
            log_path = self._write_setup_log(output_dir, notes, image, cli,
                                             len(prompt))
            self.log_paths[SETUP_LOG_NAME] = str(log_path)
            final = ["docker", "run", "--rm",
                     "--user", f"{os.getuid()}:{os.getgid()}",
                     "-v", f"{tmp_in}:/in:ro",
                     "-v", f"{output_dir}:/out:rw",
                     "-v", f"{tmp_home}:{CONTAINER_HOME}:rw"]
            for name in cred["env_names"]:
                final += ["-e", name]
            final += ["-e", f"HOME={CONTAINER_HOME}"]
            if cli == "claude":
                final += ["-e", f"CLAUDE_CONFIG_DIR={cred['container_config']}"]
            elif cli == "codex":
                final += ["-e", f"CODEX_HOME={cred['container_config']}"]
            final += ["--workdir", "/out", image, *argv]
            setup_complete = True
            yield final
        except DockerTransportError:
            raise
        except OSError:
            if setup_complete:
                raise
            raise DockerTransportError("docker_persistence_failed") from None
        except Exception:
            if setup_complete:
                raise
            raise DockerTransportError("docker_setup_failed") from None
        finally:
            self._cleanup(tmp_in, tmp_home)
