import json
from pathlib import Path

# Mapping of all 87 commits: short_hash -> proposed_subject
# Evaluated based on files changed, intent, and prompt standard (2-4 words after prefix, conventional prefix, no day/task references)
MAPPING = {
    "091578e": "feat: initialize project architecture",
    "3e06490": "docs: update development progress",
    "9456b3e": "chore: setup project foundation",
    "1b18c86": "docs: document dataset sources",
    "3cb4625": "docs: update progress tracker",
    "d73b6ec": "feat: add video loader",
    "3e646e5": "docs: document video loader",
    "23b187d": "feat: configure video ingestion",
    "c658599": "feat: update video config",
    "0f275e7": "feat: add frame extractor",
    "f2a43fe": "test: add extraction tests",
    "f000255": "docs: document frame extraction",
    "afb2b52": "feat: refine frame extraction",
    "fa45ea8": "feat: add frame preprocessor",
    "9d938b1": "docs: document frame preprocessing",
    "37065e1": "refactor: optimize frame preprocessor",
    "3cf8f13": "feat: add metadata generator",
    "050f341": "docs: document metadata generation",
    "b66dc48": "feat: add dataset organizer",
    "1e522fc": "feat: structure dataset files",
    "e4b565a": "docs: document dataset structure",
    "8cdbc05": "test: add organizer tests",
    "354993f": "feat: add dataset validator",
    "7b66f2a": "refactor: improve dataset validator",
    "e3c27f8": "docs: document dataset validation",
    "5c2e479": "docs: update validation tracker",
    "dcd574c": "test: add validation tests",
    "d1abe6e": "fix: improve video verification",
    "d65717f": "docs: log video fix",
    "6aad778": "docs: add video audit",
    "feda949": "docs: update video audit",
    "2c4b942": "feat: setup yolo environment",
    "04f511d": "feat: configure yolo loader",
    "d2b0bc6": "feat: add yolo config",
    "c3de873": "feat: add yolo detector",
    "487c47b": "refactor: update yolo config",
    "fc3e981": "feat: add single detection",
    "9dabc22": "feat: add batch detection",
    "48c3935": "docs: update detection progress",
    "b184841": "feat: add visualizer module",
    "142b0d7": "test: add visualizer tests",
    "baf3681": "feat: export visualization package",
    "dc9a15a": "docs: document detection visualizer",
    "77cf4c5": "feat: update detection visualizer",
    "b4bdfb8": "docs: update visualization metadata",
    "21cab9f": "test: verify detection visualizer",
    "1f9611a": "feat: add detection metrics",
    "4b5b386": "docs: document detection metrics",
    "daea055": "docs: update metrics progress",
    "8498daa": "test: add metrics tests",
    "ca83c5c": "chore: finalize pipeline validation",
    "cb0b02d": "docs: add system audit",
    "826a97e": "feat: add tracking environment",
    "cbe6016": "feat: integrate ByteTrack",
    "bcfef69": "feat: add single tracking",
    "413e1a7": "feat: add multi tracking",
    "e51edf6": "feat: add tracking visualization",
    "4eaac9e": "docs: update tracking validation",
    "ea11ec9": "test: validate tracking pipeline",
    "449531c": "docs: document tracking metrics",
    "a9d74a6": "feat: add feature setup",
    "699b6bf": "feat: add detection features",
    "b31517f": "feat: add motion features",
    "2500da0": "feat: add visual features",
    "d2c3689": "feat: add temporal aggregation",
    "5c43d17": "feat: add sequence builder",
    "04c8f48": "docs: document feature pipeline",
    "a7db391": "test: validate feature pipeline",
    "6e2097a": "feat: add forecasting environment",
    "d5f4efa": "feat: add baseline forecaster",
    "64009de": "feat: add lstm architecture",
    "368cb4c": "feat: add lstm training",
    "66b8ac7": "test: add training tests",
    "8210b19": "feat: add training pipeline",
    "f80c59e": "feat: add multi-horizon forecasting",
    "f41445a": "feat: add forecasting evaluator",
    "5223a51": "feat: add model evaluation",
    "022c8f5": "docs: add audit reports",
    "1dabfb6": "feat: add audit metrics",
    "97e3aed": "test: validate forecasting pipeline",
    "11b1669": "docs: update validation tracker",
    "9600f70": "test: add auditor tests",
    "44eed37": "feat: add project auditor",
    "6be8c22": "docs: add audit documentation",
    "6c1ca40": "docs: update audit readme",
    "0918562": "test: add auditor suite",
    "d106cfe": "test: complete project audit",
}

with open("reports/github/commits_with_files.json", "r", encoding="utf-8") as f:
    commits = json.load(f)

assert len(commits) == 87, f"Expected 87 commits, got {len(commits)}"
assert len(MAPPING) == 87, f"Expected 87 mapped commits, got {len(MAPPING)}"

changes_count = 0
preserved_count = 0

for c in commits:
    sh = c["short_hash"]
    orig = c["subject"]
    prop = MAPPING[sh]
    if orig == prop:
        preserved_count += 1
    else:
        changes_count += 1

    # Validate proposed message format
    parts = prop.split(": ", 1)
    assert len(parts) == 2, f"Invalid format in {prop}"
    prefix, words = parts[0], parts[1].split()
    assert prefix in ["feat", "fix", "test", "docs", "refactor", "chore"], f"Invalid prefix {prefix}"
    assert 2 <= len(words) <= 4, f"Word count {len(words)} out of range (2-4) in '{prop}'"
    for forbidden in ["day", "task", "day 1", "day 2", "day 3", "day 4", "day 5", "day 6", "day 7"]:
        assert forbidden not in prop.lower(), f"Forbidden reference '{forbidden}' in '{prop}'"

print(f"Validation successful!")
print(f"Total commits: {len(commits)}")
print(f"Commits to change: {changes_count}")
print(f"Commits preserved: {preserved_count}")
