import json
from pathlib import Path
from pydantic import TypeAdapter

from modules.core.config import Config, RLAlgoConfig


def generate_schemas():
    target_dir = Path(__file__).resolve().parent.parent / "config" / "schemas"

    if target_dir.exists() and target_dir.is_dir():
        for file in target_dir.glob("*.json"):
            file.unlink()
            print(f"Deleted old schema: {file.name}")
    else:
        target_dir.mkdir(parents=True, exist_ok=True)

    schema_dict = Config.model_json_schema()

    if "required" in schema_dict:
        del schema_dict["required"]

    output_path = target_dir / "schema.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(schema_dict, f, indent=2, ensure_ascii=False)

    print(f"Schema saved to: {output_path}")

    rl_adapter = TypeAdapter(RLAlgoConfig)
    rl_schema_dict = rl_adapter.json_schema()

    rl_output_path = target_dir / "schema_rl_algo.json"
    with open(rl_output_path, "w", encoding="utf-8") as f:
        json.dump(rl_schema_dict, f, indent=2, ensure_ascii=False)

    print(f"Schema saved to: {rl_output_path}")


if __name__ == "__main__":
    generate_schemas()
