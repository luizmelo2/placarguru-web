"""Script para atualizar pacotes Python desatualizados."""
import subprocess
import json
import sys

out = subprocess.check_output(
    [sys.executable, "-m", "pip", "list", "--outdated", "--format=json"]
)
for pkg in json.loads(out):
    name = pkg["name"]
    subprocess.call([sys.executable, "-m", "pip", "install", "-U", name])

# pip install pip-review
# pip-review --auto
