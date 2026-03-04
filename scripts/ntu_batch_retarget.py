import os
import re
import sys
import argparse
import subprocess
from collections import defaultdict
from glob import glob


NTU_PATTERN = re.compile(r"S(\d{3})C(\d{3})P(\d{3})R(\d{3})A(\d{3})", re.IGNORECASE)


def parse_codes(path):
    name = os.path.basename(path)
    m = NTU_PATTERN.search(name)
    if not m:
        return None
    return {
        "S": m.group(1),
        "C": m.group(2),
        "P": m.group(3),
        "R": m.group(4),
        "A": m.group(5),
        "name": m.group(0),
        "path": path,
    }


def scan_ntu(root):
    files = glob(os.path.join(root, "**", "*.skeleton"), recursive=True)
    persons = defaultdict(list)
    actions = defaultdict(list)
    all_actions = set()

    for f in files:
        rec = parse_codes(f)
        if not rec:
            continue
        persons[rec["P"]].append(rec)
        actions[rec["A"]].append(rec)
        all_actions.add(rec["A"])

    return persons, actions, sorted(all_actions)


def choose_target_skeleton(records_for_person):
    if not records_for_person:
        return None
    recs = sorted(records_for_person, key=lambda r: (r["R"], r["C"], r["S"], r["A"]))
    return recs[0]


def choose_source_for_action(action_records, target_p):
    for rec in sorted(action_records, key=lambda r: (r["R"], r["C"], r["S"], r["P"])):
        if rec["P"] != target_p:
            return rec
    return None


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def run_predict(python_exec, repo_root, model_path, ntu_src, ntu_tgt, out_dir, height, width, max_length):
    predict_py = os.path.join(repo_root, "predict.py")
    cmd = [
        python_exec, predict_py,
        "-n", "skeleton",
        "--model_path", model_path,
        "--ntu1", ntu_src,
        "--ntu2", ntu_tgt,
        "-h1", str(height), "-w1", str(width),
        "-h2", str(height), "-w2", str(width),
        "-o", out_dir,
        "--save_skeleton",
        "--max_length", str(max_length),
    ]
    subprocess.run(cmd, check=True)


def expected_out_name_for_2input(src_rec, tgt_rec):
    return f"S{tgt_rec['S']}C{tgt_rec['C']}P{tgt_rec['P']}R{tgt_rec['R']}A{src_rec['A']}.skeleton"


def main():
    parser = argparse.ArgumentParser(description="Batch retarget NTU motions to fill missing actions per person.")
    parser.add_argument("--ntu_root", type=str, required=True, help="NTU skeleton root directory")
    parser.add_argument("--model_path", type=str, required=True, help="Path to pretrained model (.pth)")
    parser.add_argument("--out_root", type=str, default="./outputs/ntu-batch", help="Output root directory")
    parser.add_argument("--height", type=int, default=720, help="Canvas height for visualization")
    parser.add_argument("--width", type=int, default=1280, help="Canvas width for visualization")
    parser.add_argument("--max_length", type=int, default=120, help="Max frames per sample")
    parser.add_argument("--dry_run", action="store_true", help="Only print plan, do not execute")
    parser.add_argument("--python", type=str, default=sys.executable, help="Python executable for running predict.py")

    args = parser.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    persons, actions, all_actions = scan_ntu(args.ntu_root)

    print(f"Found {len(persons)} persons, {len(all_actions)} actions in dataset.")

    total_jobs = 0
    for p, p_records in sorted(persons.items()):
        tgt = choose_target_skeleton(p_records)
        if tgt is None:
            continue
        have_actions = {rec["A"] for rec in p_records}
        missing = [a for a in all_actions if a not in have_actions]
        if not missing:
            continue

        out_dir_person = os.path.join(args.out_root, f"P{p}")
        ensure_dir(out_dir_person)

        for a in missing:
            src = choose_source_for_action(actions.get(a, []), p)
            if src is None:
                continue
            out_name = expected_out_name_for_2input(src, tgt)
            out_path = os.path.join(out_dir_person, out_name)
            if os.path.exists(out_path):
                print(f"[Skip exists] {out_path}")
                continue

            print(f"[Plan] P{p} missing A{a}: src={os.path.basename(src['path'])} -> tgt={os.path.basename(tgt['path'])}")
            if not args.dry_run:
                try:
                    run_predict(args.python, repo_root, args.model_path, src["path"], tgt["path"],
                                out_dir_person, args.height, args.width, args.max_length)
                except subprocess.CalledProcessError as e:
                    print(f"[Error] Retarget failed for P{p} A{a}: {e}")
                    continue
            total_jobs += 1

    print(f"Done. Total retarget jobs executed/planned: {total_jobs}")


if __name__ == "__main__":
    main()
