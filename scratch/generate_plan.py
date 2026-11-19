import json
from pathlib import Path

# Load all commits
with open("reports/github/commits_with_files.json", "r", encoding="utf-8") as f:
    commits = json.load(f)

# Design exact mappings
# Rule: 2-4 meaningful words after conventional prefix [feat, fix, test, docs, refactor, chore].
# No day/task references. Preserve already-short meaningful messages.
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
    "bcfef69": "feat: add single object tracking",
    "413e1a7": "feat: add multi object tracking",
    "e51edf6": "feat: add tracking visualization",
    "4eaac9e": "docs: update tracking validation",
    "ea11ec9": "test: validate tracking pipeline",
    "449531c": "docs: document tracking metrics",
    "a9d74a6": "feat: add feature engineering setup",
    "699b6bf": "feat: add detection feature extraction",
    "b31517f": "feat: add motion feature extraction",
    "2500da0": "feat: add visual quality features",
    "d2c3689": "feat: add temporal feature aggregation",
    "5c43d17": "feat: add temporal sequence builder",
    "04c8f48": "docs: document feature pipeline",
    "a7db391": "test: validate feature pipeline",
    "6e2097a": "feat: add forecasting environment",
    "d5f4efa": "feat: add baseline forecasting model",
    "64009de": "feat: add lstm architecture",
    "368cb4c": "feat: add lstm training",
    "66b8ac7": "test: add training tests",
    "8210b19": "feat: add LSTM training pipeline",
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
    "d106cfe": "test: complete end-to-end project audit",
}

# Compile stats
to_change = []
to_preserve = []

for c in commits:
    sh = c["short_hash"]
    orig = c["subject"]
    prop = MAPPING[sh]
    if orig == prop:
        to_preserve.append((sh, orig, prop))
    else:
        to_change.append((sh, orig, prop))

print(f"Total: {len(commits)}, To Change: {len(to_change)}, To Preserve: {len(to_preserve)}")

# Generate Markdown Plan
md_lines = [
    "# Git Commit History Cleanup Plan",
    "",
    "**Repository:** `https://github.com/Jeeva1117/PerceptGuard-AI`  ",
    "**Target Branch:** `main`  ",
    "**Original HEAD:** `d106cfe952d7583c37ef9748a943411742861f0b`  ",
    "**Backup Reference:** `refs/heads/backup_main_original` (points to `d106cfe`)  ",
    f"**Total Commits Reviewed:** {len(commits)}  ",
    f"**Commits Scheduled for Message Standardization:** {len(to_change)}  ",
    f"**Commits Preserved Unchanged:** {len(to_preserve)}  ",
    "",
    "---",
    "",
    "## 1. Commit Standardization Rules & Policies",
    "",
    "- **Conventional Type Prefixes:** `feat:`, `fix:`, `test:`, `docs:`, `refactor:`, `chore:`.",
    "- **Subject Length:** Strictly 2 to 4 meaningful words after the prefix.",
    "- **Prohibited Content:** Zero development-day (`Day 1`, `Day 2`, etc.) or task-number (`Task 1`, `Task 2`, etc.) references.",
    "- **Style:** Human-written, professional, concise, avoiding promotional AI fluff (`production-grade`, `comprehensive`, `advanced`).",
    "- **No Accidental Code Changes:** Only commit message strings are rewritten; commit trees, code, authors, dates, and file contents remain 100% byte-for-byte identical.",
    "- **No Merge Disruptions:** The repository history is verified to be a completely linear chain (0 merge commits).",
    "",
    "---",
    "",
    "## 2. Complete Commit Mapping Table (Chronological Order)",
    "",
    "| # | Original Commit | Original Message | Proposed Short Message | Action |",
    "| :-: | :---: | :--- | :--- | :---: |",
]

for i, c in enumerate(commits, 1):
    sh = c["short_hash"]
    orig = c["subject"]
    prop = MAPPING[sh]
    action = "**PRESERVE**" if orig == prop else "**REWRITE**"
    md_lines.append(f"| {i:02d} | `{sh}` | {orig} | `{prop}` | {action} |")

md_lines.extend([
    "",
    "---",
    "",
    "## 3. Impact & Safety Analysis",
    "",
    "### 3.1 Merge Commit Considerations",
    "- An inspection with `git log --merges` confirmed that **zero merge commits** exist in this repository.",
    "- The commit graph is a strictly linear chain of 87 commits from `091578e` to `d106cfe`.",
    "- Rewriting commit messages will preserve the linear parent-child relationship perfectly.",
    "",
    "### 3.2 Branch & Tag Impact",
    "- **Tags:** Verified with `git tag`; zero tags exist in the repository.",
    "- **Branches:** The only branches are `main` and `remotes/origin/main`.",
    "- **Backup Safety Branch:** A local branch `backup_main_original` has been created pointing directly to `d106cfe`.",
    "- The backup reference will **never** be deleted or force-pushed, ensuring zero risk of irreversible history loss.",
    "",
    "### 3.3 Code & Metadata Preservation",
    "- Author attribution (`Jeevanantham <arjunjeeva33@gmail.com>`) is strictly preserved across all commits.",
    "- Original author dates and commit dates are preserved.",
    "- Source tree hashes ($Tree$) for every commit remain identical.",
    "- Zero changes to source code, dataset files, model checkpoints, or test files.",
    "",
    "### 3.4 Synchronization & Publishing Policy",
    "- In accordance with Section 5 of the specification, **no history rewriting or force-pushing will occur during this preview stage**.",
    "- Execution will halt immediately after presenting this plan to the user.",
    "- Only upon explicit user approval will the rewrite proceed and subsequently update `origin/main` using `--force-with-lease`.",
    "",
])

report_path = Path("reports/github/commit_cleanup_plan.md")
report_path.parent.mkdir(parents=True, exist_ok=True)
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(md_lines))

print(f"Saved cleanup plan to {report_path}")
