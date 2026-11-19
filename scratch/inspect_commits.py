import json

with open("reports/github/commits_with_files.json", "r", encoding="utf-8") as f:
    commits = json.load(f)

for i, c in enumerate(commits, 1):
    short_h = c["short_hash"]
    subj = c["subject"]
    f_list = c["files"]
    f_str = ", ".join(f_list[:3]) if len(f_list) <= 3 else f"{', '.join(f_list[:2])} (+{len(f_list)-2} more)"
    print(f"{i:02d} | {short_h} | {subj} | [{f_str}]")
