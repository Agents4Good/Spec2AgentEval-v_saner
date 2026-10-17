import os
import time
import uuid
import requests
import subprocess
from dotenv import load_dotenv

if os.getenv('SPEC2AGENT_SKIP_DOTENV') != '1':
    load_dotenv(override=True)

SONAR_HOST = os.getenv('SONAR_HOST_URL')
USERNAME = os.getenv('SONAR_LOGIN')
PASSWORD = os.getenv('SONAR_PASSWORD')
USER_TOKEN = os.getenv('USER_TOKEN')
SONAR_COMMAND = os.getenv('SONAR_COMMAND', None)
EXPIRATION_DATE = "2040-01-01"

def start_sonarqube_server():
    if not SONAR_COMMAND: raise ValueError("SONAR_COMMAND environment variable is not set.")
    cmd = [SONAR_COMMAND, "console"]  
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def wait_for_sonarqube_boot(timeout=300):
    start_time = time.monotonic()
    status_url = f"{SONAR_HOST}/api/system/status"
    
    while time.monotonic() - start_time < timeout:
        try:
            response = requests.get(status_url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "UP":
                    return
        except requests.RequestException:
            pass
            
        time.sleep(5)
        
    raise TimeoutError("Time exceeded while waiting for SonarQube to boot.")

def sonar_login(session: requests.Session):
    resp = session.post(
        f"{SONAR_HOST}/api/authentication/login",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        data={"login": USERNAME, "password": PASSWORD},
        timeout=10
    )
    resp.raise_for_status()

    xsrf = session.cookies.get("XSRF-TOKEN")
    if not xsrf:
        raise RuntimeError("XSRF-TOKEN not found after login.")
    return xsrf

def create_project(session: requests.Session, xsrf: str, project_key: str, project_name: str):
    resp = session.post(
        f"{SONAR_HOST}/api/projects/create",
        auth=(USER_TOKEN, ""),
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "x-xsrf-token": xsrf
        },
        data={
            "project": project_key,
            "name": project_name
        },
        timeout=10
    )
    if resp.status_code not in (200, 400):
        resp.raise_for_status()

def generate_project_token(session: requests.Session, xsrf: str, project_key: str) -> str:
    token_name = f"analysis-{project_key}-{uuid.uuid4().hex[:4]}"
    resp = session.post(
        f"{SONAR_HOST}/api/user_tokens/generate",
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "x-xsrf-token": xsrf
        },
        data={
            "name": token_name,
            "type": "PROJECT_ANALYSIS_TOKEN",
            "projectKey": project_key,
            "expirationDate": EXPIRATION_DATE
        },
        timeout=10
    )

    resp.raise_for_status()
    return resp.json()["token"]

def run_pysonar(token: str, target_path: str, project_key: str):
    cmd = [
        "pysonar",
        f"--sonar-host-url={SONAR_HOST}",
        f"--sonar-project-key={project_key}",
        "--sonar-sources=.",
        "--sonar-scm-exclusions-disabled"
    ]
    env = os.environ.copy()
    env['SONAR_TOKEN'] = token
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                   cwd=target_path, check=True, timeout=300, env=env, shell=False)

def wait_for_analysis(session: requests.Session, project_key: str, timeout=300):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        resp = session.get(
            f"{SONAR_HOST}/api/ce/component",
            params={"component": project_key},
            auth=(USER_TOKEN, ""), timeout=10
        )
        resp.raise_for_status()
        data = resp.json()

        queue = data.get("queue", [])
        current = data.get("current", {})
        
        is_pending = len(queue) > 0
        is_in_progress = current.get("status") in ["PENDING", "IN_PROGRESS"]

        if is_pending or is_in_progress:
            time.sleep(2)
            continue
            
        if current.get("status") in ["FAILED", "CANCELED"]:
            raise RuntimeError(f"The analysis processing failed in SonarQube. Status: {current.get('status')}")

        if current.get('status') != 'SUCCESS':
            time.sleep(2)
            continue
        return
    raise TimeoutError('Time exceeded while waiting for Sonar analysis.')

def extract_metrics(session: requests.Session, project_key: str):
    metrics = [
        # correctness proxy
        "bugs",
        "reliability_rating",
        "reliability_remediation_effort",

        # security
        "vulnerabilities",
        "security_rating",
        "security_hotspots",
        "security_hotspots_reviewed",

        # maintainability
        "code_smells",
        "sqale_index",
        "sqale_rating",
        "sqale_debt_ratio",

        # complexity
        "complexity",
        "cognitive_complexity",
        "functions",
        "classes",

        # size / verbosity
        "ncloc",
        "lines",
        "statements",
        "comment_lines_density",

        # redundancy
        "duplicated_lines_density",
        "duplicated_blocks"
    ]

    resp = session.get(
        f"{SONAR_HOST}/api/measures/component",
        params={
            "component": project_key,
            "metricKeys": ",".join(metrics)
        },
        auth=(USER_TOKEN, ""), timeout=10
    )
    
    resp.raise_for_status()
    data = resp.json()

    measures = data.get("component", {}).get("measures", [])
    
    result = {m["metric"]: m.get("value") for m in measures}
    return result
