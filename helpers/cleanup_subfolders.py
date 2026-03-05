import shutil
from pathlib import Path


def cleanup_numeric_subfolders():
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    results_dir = project_root / "results"

    if not results_dir.exists():
        print(f"Dir {results_dir} not found.")
        return

    simulation_dirs = [p.parent for p in results_dir.rglob(".hydra") if p.is_dir()]

    deleted_count = 0
    for sim_dir in simulation_dirs:
        for item in sim_dir.iterdir():
            if item.is_dir() and item.name.isdigit():
                try:
                    shutil.rmtree(item)
                    deleted_count += 1
                except Exception as e:
                    print(f"Error during deleting {item}: {e}")

    print(f"\nDeleted {deleted_count} sub dirs.")


if __name__ == "__main__":
    cleanup_numeric_subfolders()
