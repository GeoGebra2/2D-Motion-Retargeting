import os
import re
import sys
import argparse
import subprocess
from collections import defaultdict
from glob import glob
from tqdm import tqdm


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


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def run_predict(python_exec, repo_root, model_path, ntu_src, ntu_tgt, out_dir, height, width, max_length, fname_suffix=None, no_video=False, only_out12=False):
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
    if fname_suffix:
        cmd.extend(["--fname_suffix", fname_suffix])
    if no_video:
        cmd.append("--no_video")
    if only_out12:
        cmd.append("--only_out12")
    subprocess.run(cmd, check=True)


def expected_out_name_for_2input(src_rec, tgt_rec):
    return f"S{tgt_rec['S']}C{tgt_rec['C']}P{tgt_rec['P']}R{tgt_rec['R']}A{src_rec['A']}F{src_rec['P']}.skeleton"


def choose_sources_same_sc(target_rec, all_records):
    out = []
    for rec in all_records:
        if rec["P"] == target_rec["P"]:
            continue
        if rec["S"] == target_rec["S"] and rec["C"] == target_rec["C"]:
            out.append(rec)
    return out


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

    all_records = []
    for recs in persons.values():
        all_records.extend(recs)
    tasks = []
    for p, p_records in sorted(persons.items()):
        out_dir_person = os.path.join(args.out_root, f"P{p}")
        ensure_dir(out_dir_person)
        for tgt in sorted(p_records, key=lambda r: (r["R"], r["C"], r["S"], r["A"])):
            src_list = choose_sources_same_sc(tgt, all_records)
            for src in src_list:
                out_name = expected_out_name_for_2input(src, tgt)
                out_path = os.path.join(out_dir_person, out_name)
                if os.path.exists(out_path):
                    continue
                tasks.append((p, out_dir_person, tgt, src, out_path))

    for (_, out_dir_person, tgt, src, out_path) in tqdm(tasks, desc="Retarget", unit="job"):
        print(f"[Plan] tgt={tgt['name']} <= src={src['name']} -> {out_path}")
        if not args.dry_run:
            try:
                run_predict(
                    args.python, repo_root, args.model_path,
                    src["path"], tgt["path"],
                    out_dir_person, args.height, args.width, args.max_length,
                    fname_suffix=src["P"], no_video=True, only_out12=True
                )
            except subprocess.CalledProcessError as e:
                print(f"[Error] Retarget failed for P{tgt['P']} A{src['A']}: {e}")
                continue

    print(f"Done. Total retarget jobs executed/planned: {len(tasks)}")


if __name__ == "__main__":
    main()
