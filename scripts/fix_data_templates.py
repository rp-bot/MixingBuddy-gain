#!/usr/bin/env python3
"""
Fix dataset and config templates:
- Replace short no_error template in train/test JSONL with balanced version
- Ensure very_quiet template matches the requested sentence
- Fix dB range order for quiet and very_quiet in 05_musdb_expanded.yaml
"""

from pathlib import Path
import json
import re

OLD_NO_ERROR = "The mix is well-balanced. No adjustments needed."
NEW_NO_ERROR = (
    "After analyzing the mix, all stems are at appropriate levels and the balance is correct. "
    "The mix is well-balanced. No adjustments needed."
)

VERY_QUIET_SENTENCE = (
    "The {target_stem} is barely audible. Increase the {target_stem} level between "
    "{min_gain_db} and {max_gain_db} dB to balance the mix."
)

QUIET_SENTENCE = (
    "The {target_stem} is a little too quiet. Increase the {target_stem} level between "
    "{min_gain_db} and {max_gain_db} dB to balance the mix."
)


def replace_no_error_in_jsonl(file_path: Path) -> int:
    """Replace OLD_NO_ERROR with NEW_NO_ERROR in common fields: 'response', 'chosen', 'rejected'."""
    if not file_path.exists():
        print(f"- Skipping missing file: {file_path}")
        return 0
    fixed = 0
    tmp_path = file_path.with_suffix(file_path.suffix + ".tmp")
    with file_path.open("r") as infile, tmp_path.open("w") as outfile:
        for line in infile:
            data = json.loads(line)
            # SFT format: response
            if data.get("response") == OLD_NO_ERROR:
                data["response"] = NEW_NO_ERROR
                fixed += 1
            # DPO format: chosen/rejected
            if data.get("chosen") == OLD_NO_ERROR:
                data["chosen"] = NEW_NO_ERROR
                fixed += 1
            if data.get("rejected") == OLD_NO_ERROR:
                data["rejected"] = NEW_NO_ERROR
                fixed += 1
            outfile.write(json.dumps(data) + "\n")
    backup_path = file_path.with_suffix(file_path.suffix + ".bak")
    file_path.rename(backup_path)
    tmp_path.rename(file_path)
    print(f"- {file_path}: replaced {fixed} occurrences (backup at {backup_path.name})")
    return fixed


def normalize_quiet_very_quiet_in_jsonl(file_path: Path) -> int:
    """
    Normalize quiet and very_quiet responses in SFT JSONL:
    - quiet: set to QUIET_SENTENCE with 3 and 6
    - very_quiet: set to VERY_QUIET_SENTENCE with 6 and 12
    """
    if not file_path.exists():
        print(f"- Skipping missing file: {file_path}")
        return 0
    fixed = 0
    tmp_path = file_path.with_suffix(file_path.suffix + ".tmp2")
    with file_path.open("r") as infile, tmp_path.open("w") as outfile:
        for line in infile:
            data = json.loads(line)
            meta = data.get("meta", {})
            error_category = meta.get("error_category")
            target_stem = meta.get("target_stem")
            response = data.get("response")
            if isinstance(response, str) and target_stem:
                if error_category == "quiet":
                    desired = QUIET_SENTENCE.format(target_stem=target_stem, min_gain_db=3, max_gain_db=6)
                    if response != desired:
                        data["response"] = desired
                        fixed += 1
                elif error_category == "very_quiet":
                    desired = VERY_QUIET_SENTENCE.format(target_stem=target_stem, min_gain_db=6, max_gain_db=12)
                    if response != desired:
                        data["response"] = desired
                        fixed += 1
            outfile.write(json.dumps(data) + "\n")
    backup_path = file_path.with_suffix(file_path.suffix + ".bak2")
    file_path.rename(backup_path)
    tmp_path.rename(file_path)
    print(f"- {file_path}: normalized {fixed} quiet/very_quiet responses (backup at {backup_path.name})")
    return fixed

def fix_yaml_file(yaml_path: Path) -> None:
    """Edit YAML text safely with regex replacements while preserving comments/formatting."""
    text = yaml_path.read_text()

    # Ensure very_quiet template line matches requested sentence
    # Replace any existing very_quiet template line content with the exact sentence
    text = re.sub(
        r'(very_quiet:\s*\n\s*-\s*")[^"]*(")',
        r'\1' + VERY_QUIET_SENTENCE.replace("\\", "\\\\") + r'\2',
        text,
        count=1,
        flags=re.MULTILINE,
    )

    # Fix range db order:
    # quiet: [-6, -3] -> quiet: [-3, -6]
    text = re.sub(
        r"(quiet:\s*\[\s*)-6(\s*,\s*)-3(\s*\])",
        r"\1-3\2-6\3",
        text,
        count=1,
        flags=re.MULTILINE,
    )
    # very_quiet: [-12, -6] -> very_quiet: [-6, -12]
    text = re.sub(
        r"(very_quiet:\s*\[\s*)-12(\s*,\s*)-6(\s*\])",
        r"\1-6\2-12\3",
        text,
        count=1,
        flags=re.MULTILINE,
    )

    yaml_backup = yaml_path.with_suffix(yaml_path.suffix + ".bak")
    yaml_backup.write_text(yaml_path.read_text())
    yaml_path.write_text(text)
    print(f"- Updated YAML: {yaml_path} (backup at {yaml_backup.name})")


def main():
    repo_root = Path(__file__).resolve().parent.parent
    train_jsonl = repo_root / "data/musdb18hq_processed/train/training_samples.jsonl"
    test_jsonl = repo_root / "data/musdb18hq_processed/test/test_samples.jsonl"
    yaml_path = repo_root / "configs/data/05_musdb_expanded.yaml"

    print("Fixing JSONL datasets...")
    replace_no_error_in_jsonl(train_jsonl)
    replace_no_error_in_jsonl(test_jsonl)
    normalize_quiet_very_quiet_in_jsonl(train_jsonl)
    normalize_quiet_very_quiet_in_jsonl(test_jsonl)

    print("Fixing YAML templates and ranges...")
    fix_yaml_file(yaml_path)

    print("✅ Done.")


if __name__ == "__main__":
    main()


