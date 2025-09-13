import csv, os, argparse, json

def walk_images(root):
    for cls in sorted(os.listdir(root)):
        d = os.path.join(root, cls)
        if not os.path.isdir(d): continue
        for f in sorted(os.listdir(d)):
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                yield os.path.join(d, f), cls

def write_csv(split_dir, mapping, out_csv):
    rows = []
    for p, cls in walk_images(split_dir):
        if cls not in mapping:
            print(f"[warn] no CO2 for class {cls}, skipping {p}")
            continue
        rows.append({"image_path": p, "co2": mapping[cls]})
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image_path","co2"])
        w.writeheader(); w.writerows(rows)
    print(f"Wrote {len(rows)} rows -> {out_csv}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", default="dataset_products/train")
    ap.add_argument("--val_dir",   default="dataset_products/validate")
    ap.add_argument("--mapping_json", required=True, help='{"ProductA": 12.7, "ProductB": 5.3, ...}')
    ap.add_argument("--out_train", default="metadata/co2_train.csv")
    ap.add_argument("--out_val",   default="metadata/co2_val.csv")
    args = ap.parse_args()

    with open(args.mapping_json, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    write_csv(args.train_dir, mapping, args.out_train)
    write_csv(args.val_dir,   mapping, args.out_val)
