import os, json, csv, math, random, argparse

def list_classes(root_dirs):
    classes = set()
    for root in root_dirs:
        if not os.path.isdir(root): continue
        for c in os.listdir(root):
            if os.path.isdir(os.path.join(root, c)):
                classes.add(c)
    return sorted(classes)

def assign_dummy_co2(classes, lo=2.0, hi=18.0, seed=42):
    random.seed(seed)
    n = max(len(classes), 1)
    base = {}
    # spread classes roughly evenly in [lo, hi]
    for i, c in enumerate(classes):
        frac = i / max(n - 1, 1)
        val = lo + frac * (hi - lo)
        # small class-level random wiggle so adjacent classes aren't perfectly linear
        val *= (0.95 + 0.10 * random.random())  # 0.95x .. 1.05x
        base[c] = round(val, 2)
    return base

def walk_images(split_dir):
    for cls in sorted(os.listdir(split_dir)):
        d = os.path.join(split_dir, cls)
        if not os.path.isdir(d): continue
        for f in sorted(os.listdir(d)):
            if f.lower().endswith((".jpg",".jpeg",".png")):
                yield os.path.join(d, f), cls

def write_split(split_dir, class_to_co2, out_csv, jitter=0.10, seed=123):
    random.seed(seed)
    rows = []
    for p, cls in walk_images(split_dir):
        base = class_to_co2.get(cls)
        if base is None: 
            print(f"[warn] class {cls} missing in mapping; skipping {p}")
            continue
        # per-image jitter: ±jitter
        mult = 1.0 + random.uniform(-jitter, jitter)
        co2 = round(base * mult, 2)
        rows.append({"image_path": p, "co2": co2})
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image_path","co2"])
        w.writeheader(); w.writerows(rows)
    print(f"Wrote {len(rows)} rows -> {out_csv}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", default="dataset_products/train")
    ap.add_argument("--val_dir",   default="dataset_products/validate")
    ap.add_argument("--out_train", default="metadata/co2_train.csv")
    ap.add_argument("--out_val",   default="metadata/co2_val.csv")
    ap.add_argument("--out_mapping", default="metadata/co2_by_product.json")
    ap.add_argument("--min_co2", type=float, default=2.0)
    ap.add_argument("--max_co2", type=float, default=18.0)
    ap.add_argument("--jitter", type=float, default=0.10)  # ±10%
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    classes = list_classes([args.train_dir, args.val_dir])
    if not classes:
        raise SystemExit("No class folders found. Check dataset_products/train and validate.")

    mapping = assign_dummy_co2(classes, args.min_co2, args.max_co2, seed=args.seed)
    os.makedirs(os.path.dirname(args.out_mapping), exist_ok=True)
    with open(args.out_mapping, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)
    print(f"Wrote {args.out_mapping} with {len(mapping)} classes.")

    write_split(args.train_dir, mapping, args.out_train, jitter=args.jitter, seed=args.seed+1)
    write_split(args.val_dir,   mapping, args.out_val,   jitter=args.jitter, seed=args.seed+2)

if __name__ == "__main__":
    main()
