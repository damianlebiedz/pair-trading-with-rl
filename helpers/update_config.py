"""
Configuration Management Helper.

This project utilizes a hybrid Pydantic-Hydra configuration setup.
Whenever you add, modify, or remove parameters in the Pydantic models
located in `modules/core/config.py`, you must run this helper script (`update_config.py`).

Running this script automatically updates the JSON schemas in the `config/schemas/`
directory. This ensures that your YAML configuration files benefit from IDE features
like auto-completion, inline descriptions, and automatic validation against the schema.
"""

import json
from pathlib import Path
from typing import Any
from pydantic import TypeAdapter

from modules.core.config import (
    Config,
    RLAlgoConfig,
    DataFetchingPipeline,
)
from modules.utils.logger import get_logger

logger = get_logger(__name__)


def strip_required(schema: Any) -> None:
    """
    Recursively removes the 'required' key from the generated JSON schema dictionaries.
    """
    if isinstance(schema, dict):
        schema.pop("required", None)
        for value in schema.values():
            strip_required(value)
    elif isinstance(schema, list):
        for item in schema:
            strip_required(item)


def generate_schemas():
    """
    Generates JSON schemas from Pydantic models and exports them to the schemas directory.
    """
    target_dir = Path(__file__).resolve().parent.parent / "config" / "schemas"

    schema_dict = Config.model_json_schema()
    strip_required(schema_dict)

    with open(target_dir / "schema.json", "w", encoding="utf-8") as f:
        json.dump(schema_dict, f, indent=2)
    logger.debug(f"Schema saved to: {target_dir / "schema.json"}")

    rl_adapter = TypeAdapter(RLAlgoConfig)
    with open(target_dir / "schema_rl_algo.json", "w", encoding="utf-8") as f:
        json.dump(rl_adapter.json_schema(), f, indent=2)
    logger.debug(f"Schema saved to: {target_dir / "schema_rl_algo.json"}")

    helper_adapter = TypeAdapter(DataFetchingPipeline)
    with open(target_dir / "schema_helpers.json", "w", encoding="utf-8") as f:
        json.dump(helper_adapter.json_schema(), f, indent=2)
    logger.debug(f"Schema saved to: {target_dir / "schema_helpers.json"}")

    logger.info(f"Schemas saved to: {target_dir}")


def generate_docs():
    """
    Parses the main configuration schema to generate a Markdown documentation file.
    """
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

    logger.info(f"Documentation generated at: {output_path}")


if __name__ == "__main__":
    generate_schemas()
    generate_docs()
