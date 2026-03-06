import os
import re
import sys
import argparse
import subprocess
from collections import defaultdict
from glob import glob
from tqdm import tqdm
import random


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
    return f"S{tgt_rec['S']}C{tgt_rec['C']}P{tgt_rec['P']}R{tgt_rec['R']}A{src_rec['A']}.skeleton"


def by_action_camera(records):
    m = defaultdict(lambda: defaultdict(list))
    for r in records:
        m[r["A"]][r["C"]].append(r)
    return m


def select_targets_for_person(p_records):
    pam = by_action_camera(p_records)
    eligible_actions = list(pam.keys())
    if len(eligible_actions) < 20:
        return [], set()
    random.shuffle(eligible_actions)

    selected_actions = set()
    targets = []
    for a in eligible_actions:
        if len(selected_actions) >= 20:
            break
        recs_any = []
        for cam_id, recs in pam[a].items():
            recs_any.extend(recs)
        if not recs_any:
            continue
        tgt = random.choice(recs_any)
        targets.append(tgt)
        selected_actions.add(a)

    if len(targets) != 20:
        return [], set()

    needed = {"001", "002", "003"}
    present = set(t["C"] for t in targets)
    missing = list(needed - present)
    # 尝试通过替换/调整实现相机覆盖
    for cm in missing:
        done = False
        # 优先用已选动作在该相机上的样本替换
        for idx, t in enumerate(targets):
            a = t["A"]
            if cm in pam.get(a, {}) and len(pam[a][cm]) > 0 and t["C"] != cm:
                targets[idx] = random.choice(pam[a][cm])
                done = True
                break
        if done:
            continue
        # 尝试用未选动作替换一个目标
        repl_action = None
        for a2 in eligible_actions:
            if a2 in selected_actions:
                continue
            if cm in pam.get(a2, {}) and len(pam[a2][cm]) > 0:
                repl_action = a2
                break
        if repl_action is not None:
            # 替换第一个目标
            i = 0
            old_action = targets[i]["A"]
            targets[i] = random.choice(pam[repl_action][cm])
            selected_actions.remove(old_action)
            selected_actions.add(repl_action)
        else:
            return [], set()

    present = set(t["C"] for t in targets)
    if not {"001", "002", "003"}.issubset(present):
        return [], set()

    return targets, selected_actions


def pick_sources_for_target(tgt, persons, avoid_persons_set, avoid_actions_set):
    s = tgt["S"]
    c = tgt["C"]
    candidates_persons = [pp for pp in persons.keys() if pp not in avoid_persons_set and pp != tgt["P"]]
    random.shuffle(candidates_persons)

    used_actions = set()
    selected_sources = []
    selected_persons = []

    for sp in candidates_persons:
        sp_records = persons[sp]
        # 所有满足同 S 同 C 的动作集合
        actions_available = set()
        for r in sp_records:
            if r["C"] == c:
                actions_available.add(r["A"])
        # 过滤：不与目标已选动作重复，且不与已选来源动作重复
        actions_pool = [a for a in actions_available if a not in avoid_actions_set and a not in used_actions]
        random.shuffle(actions_pool)
        if len(actions_pool) < 4:
            continue
        chosen_actions = actions_pool[:4]
        # 为每个动作挑选一条记录
        for a in chosen_actions:
            rr = [r for r in sp_records if r["C"] == c and r["A"] == a]
            if not rr:
                continue
            selected_sources.append(random.choice(rr))
            used_actions.add(a)
        selected_persons.append(sp)
        if len(selected_persons) == 3:
            break

    # 需满足：3 个来源人，每人 4 个不同动作，总计 12 个动作互不重复
    if len(selected_persons) != 3 or len(selected_sources) < 12:
        return []
    # 截断到前 12 条（按每人 4 个）
    return selected_sources[:12]


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

    target_persons = [f"{i:03d}" for i in range(1, 11)]
    tasks = []

    for p in target_persons:
        if p not in persons:
            continue
        p_records = persons[p]
        out_dir_person = os.path.join(args.out_root, f"P{p}")
        ensure_dir(out_dir_person)

        targets, selected_actions = select_targets_for_person(p_records)
        if len(targets) != 20:
            print(f"[Skip] P{p}: insufficient targets for 20 with C1–C3 coverage")
            continue
        avoid_persons_set = set(target_persons)
        avoid_actions_set = set(selected_actions)

        for tgt in targets:
            srcs = pick_sources_for_target(tgt, persons, avoid_persons_set, avoid_actions_set)
            if len(srcs) != 12:
                print(f"[Skip target] {tgt['name']}: insufficient source coverage (need 3 persons × 4 actions)")
                continue
            for src in srcs:
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
                    fname_suffix=None, no_video=True, only_out12=True
                )
            except subprocess.CalledProcessError as e:
                print(f"[Error] Retarget failed for P{tgt['P']} A{src['A']}: {e}")
                continue

    print(f"Done. Total retarget jobs executed/planned: {len(tasks)}")


if __name__ == "__main__":
    main()
