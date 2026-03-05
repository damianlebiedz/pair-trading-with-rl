import json
from pathlib import Path
from pydantic import TypeAdapter

from modules.core.config import Config, RLAlgoConfig
from modules.utils.logger import get_logger

logger = get_logger(__name__)


def generate_schemas():
    target_dir = Path(__file__).resolve().parent.parent / "config" / "schemas"

    if target_dir.exists() and target_dir.is_dir():
        for file in target_dir.glob("*.json"):
            file.unlink()
            logger.debug(f"Deleted old schema: {file.name}")
    else:
        target_dir.mkdir(parents=True, exist_ok=True)

    schema_dict = Config.model_json_schema()

    if "required" in schema_dict:
        del schema_dict["required"]

    output_path = target_dir / "schema.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(schema_dict, f, indent=2, ensure_ascii=False)

    logger.info(f"Schema saved to: {output_path}")

    rl_adapter = TypeAdapter(RLAlgoConfig)
    rl_schema_dict = rl_adapter.json_schema()

    rl_output_path = target_dir / "schema_rl_algo.json"
    with open(rl_output_path, "w", encoding="utf-8") as f:
        json.dump(rl_schema_dict, f, indent=2, ensure_ascii=False)

    logger.info(f"Schema saved to: {rl_output_path}")


def generate_docs():
    schema = Config.model_json_schema()

    md_lines = [
        "# Configuration Documentation (YAML)\n",
        "Below is an automatically generated list of all configuration parameters supported by the system.\n",
        "## Root Parameters\n",
    ]

    for prop_name, prop_info in schema.get("properties", {}).items():
        desc = prop_info.get("description", "*No description provided*")

        if prop_name not in ["name", "defaults"]:
            md_lines.append(f"- **`{prop_name}`**: {desc}")

    md_lines.append("\n---\n")

    md_lines.append("## Configuration Modules\n")

    defs = schema.get("$defs", {})
    for def_name, def_info in defs.items():
        if "properties" in def_info:
            md_lines.append(f"### {def_name}")

            if "description" in def_info:
                md_lines.append(f"_{def_info['description']}_\n")

            for prop_name, prop_info in def_info["properties"].items():
                desc = prop_info.get("description", "*No description provided*")
                md_lines.append(f"- **`{prop_name}`**: {desc}")
            md_lines.append("\n")

    target_dir = Path(__file__).resolve().parent.parent / "docs"
    target_dir.mkdir(parents=True, exist_ok=True)

    output_path = target_dir / "configuration.md"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    logger.info(f"Documentation successfully generated at: {output_path}")


if __name__ == "__main__":
    generate_schemas()
    generate_docs()
