# /// script
# requires-python = ">=3.8"
# dependencies = []
# ///

import json
import argparse
from pathlib import Path


def extract(export_path: Path):
    """Parses an Open WebUI functions export and extracts each function into its own subdirectory."""
    if not export_path.exists():
        print(f"Error: Export file not found at {export_path}")
        return

    try:
        # read_text handles open/close and encoding
        functions = json.loads(export_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return

    if not isinstance(functions, list):
        print("Error: Export format is expected to be a list of functions")
        return

    extracted_count = 0
    for func in functions:
        func_type = func.get("type", "other")
        base_dirname = f"{func_type}s" if not func_type.endswith("s") else func_type

        # Fallback if type isn't one of the expected ones
        if base_dirname not in ["actions", "filters", "pipes"]:
            base_dirname = "other"

        func_id = func.get("id", "unknown")
        target_dir = Path(base_dirname) / func_id
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / "__init__.py").touch()

        content = func.get("content", "")
        py_file = target_dir / f"{func_id}.py"
        py_file.write_text(content, encoding="utf-8")

        # Extract metadata (everything except code)
        meta = {k: v for k, v in func.items() if k != "content"}
        json_file = target_dir / f"{func_id}.json"
        json_file.write_text(
            json.dumps(meta, indent=4, ensure_ascii=False), encoding="utf-8"
        )

        extracted_count += 1
        print(f"Extracted: {target_dir}/")

    print(f"\nSuccessfully extracted {extracted_count} function")


def bundle(output_path: Path):
    """Combines functions from the actions/, filters/, and pipes/ directories into a single export JSON."""
    functions = []
    for base_dirname in ["actions", "filters", "pipes", "other"]:
        base_dir = Path(base_dirname)
        if not base_dir.exists():
            continue

        for target_dir in base_dir.iterdir():
            if not target_dir.is_dir():
                continue

            func_id = target_dir.name
            json_path = target_dir / f"{func_id}.json"
            py_path = target_dir / f"{func_id}.py"

            if json_path.exists() and py_path.exists():
                func = json.loads(json_path.read_text(encoding="utf-8"))
                content = py_path.read_text(encoding="utf-8")

                # Reinsert the content
                func["content"] = content
                functions.append(func)
                print(f"Bundled: {target_dir}/")

    output_path.write_text(
        json.dumps(functions, ensure_ascii=False, separators=(',', ':')), encoding="utf-8"
    )
    print(f"\nSuccessfully bundled {len(functions)} functions into {output_path}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Open WebUI Function Manager")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    extract_parser = subparsers.add_parser(
        "extract", help="Extract functions from a JSON export"
    )
    extract_parser.add_argument("path", type=Path, help="Path to the export JSON file")

    bundle_parser = subparsers.add_parser(
        "bundle", help="Bundle local files into a JSON export"
    )
    bundle_parser.add_argument("path", type=Path, help="Path to the output JSON file")

    args = parser.parse_args()

    if args.command == "extract":
        extract(args.path)
    elif args.command == "bundle":
        bundle(args.path)
    else:
        parser.print_help()
