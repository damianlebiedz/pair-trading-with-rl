from pathlib import Path

from modules.core.config import Config


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

    print(f"Documentation successfully generated at: {output_path}")


if __name__ == "__main__":
    generate_docs()
