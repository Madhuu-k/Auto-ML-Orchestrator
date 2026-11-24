from pathlib import Path
import yaml

config_path = Path(__file__).resolve().parents[2] / "configs" / "experiment.yaml"

with open(config_path, "r") as file:
    content = file.read()

config = yaml.safe_load(content)
print(config)