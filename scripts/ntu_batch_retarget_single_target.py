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
    for f in files:
        rec = parse_codes(f)
        if not rec:
            continue
        persons[rec["P"]].append(rec)
    return persons


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def run_predict(python_exec, repo_root, model_path, ntu_src, ntu_tgt, out_dir, height, width, max_length, r_override):
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
        "--no_video",
        "--only_out12",
        "--fname_r_override", str(r_override),
    ]
    subprocess.run(cmd, check=True)


def expected_out_name(src_rec, tgt_rec, r_code):
    return f"S{tgt_rec['S']}C{tgt_rec['C']}P{src_rec['P']}R{r_code}A{src_rec['A']}.skeleton"


def canon_person(p):
    p = p.strip()
    if p.startswith("P") or p.startswith("p"):
        p = p[1:]
    return p.zfill(3)


def main():
    parser = argparse.ArgumentParser(description="Batch retarget all others to one target person (match S/C, disambiguate by R).")
    parser.add_argument("--ntu_root", type=str, required=True, help="NTU skeleton root directory")
    parser.add_argument("--model_path", type=str, required=True, help="Path to pretrained model (.pth)")
    parser.add_argument("--target_person", type=str, required=False, help="Target person code (e.g., 015 or P015). If not set, choose from 011-040 automatically")
    parser.add_argument("--out_root", type=str, default="./outputs/ntu-batch-single", help="Output root directory")
    parser.add_argument("--height", type=int, default=720, help="Canvas height")
    parser.add_argument("--width", type=int, default=1280, help="Canvas width")
    parser.add_argument("--max_length", type=int, default=120, help="Max frames per sample")
    parser.add_argument("--dry_run", action="store_true", help="Only print plan, do not execute")
    parser.add_argument("--python", type=str, default=sys.executable, help="Python executable for running predict.py")
    args = parser.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    persons = scan_ntu(args.ntu_root)

    eligible_targets = sorted([p for p in persons.keys() if 11 <= int(p) <= 40])
    if args.target_person:
        target_p = canon_person(args.target_person)
        if target_p not in eligible_targets:
            print(f"Target person must be within P011-P040 and exist in dataset. Given: P{target_p}")
            return
    else:
        if not eligible_targets:
            print("No eligible target person found within P011-P040")
            return
        target_p = eligible_targets[0]

    out_dir_person = os.path.join(args.out_root, f"P{target_p}")
    ensure_dir(out_dir_person)

    sc_index = defaultdict(list)
    for p, recs in persons.items():
        # 仅使用 P001–P010 作为来源；排除目标本人
        if p == target_p or not (1 <= int(p) <= 10):
            continue
        for r in recs:
            sc_index[(r["S"], r["C"])].append(r)

    base_max_r = {}
    tasks = []

    for tgt in persons[target_p]:
        key = (tgt["S"], tgt["C"])
        src_list = sc_index.get(key, [])
        if not src_list:
            continue
        for src in src_list:
            base_key = (tgt["S"], tgt["C"], src["P"], src["A"])
            if base_key not in base_max_r:
                pattern = f"S{tgt['S']}C{tgt['C']}P{src['P']}R???A{src['A']}.skeleton"
                existing = glob(os.path.join(out_dir_person, pattern))
                max_r = 0
                for ef in existing:
                    m = re.search(r'R(\d{3})', os.path.basename(ef))
                    if m:
                        max_r = max(max_r, int(m.group(1)))
                base_max_r[base_key] = max_r
            base_max_r[base_key] += 1
            r_code = f"{base_max_r[base_key]:03d}"
            out_name = expected_out_name(src, tgt, r_code)
            out_path = os.path.join(out_dir_person, out_name)
            while os.path.exists(out_path):
                base_max_r[base_key] += 1
                r_code = f"{base_max_r[base_key]:03d}"
                out_name = expected_out_name(src, tgt, r_code)
                out_path = os.path.join(out_dir_person, out_name)
            tasks.append((out_dir_person, tgt, src, out_path, r_code))

    for (out_dir_person, tgt, src, out_path, r_code) in tqdm(tasks, desc="Retarget", unit="job"):
        print(f"[Plan] src={src['name']} => tgt={tgt['name']} (R={r_code}) -> {out_path}")
        if args.dry_run:
            continue
        try:
            run_predict(
                args.python, repo_root, args.model_path,
                src["path"], tgt["path"],
                out_dir_person, args.height, args.width, args.max_length,
                r_override=r_code
            )
        except subprocess.CalledProcessError as e:
            print(f"[Error] Retarget failed for P{src['P']} A{src['A']} -> P{tgt['P']}: {e}")
            continue

    print(f"Done. Total retarget jobs executed/planned: {len(tasks)}")


if __name__ == "__main__":
    main()
