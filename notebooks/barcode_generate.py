# generate_images_and_barcodes.py
import os
import csv
from PIL import Image, ImageEnhance
import random
import barcode
from barcode.writer import ImageWriter
import glob

# Paths (Windows-safe)
RAW_PATH = r"C:/Users/harsh/desktop/GreenLens/GreenLens/dataset/raw"
PROCESSED_PATH = r"C:/Users/harsh/desktop/GreenLens/GreenLens/dataset/processed"
CSV_PATH = r"C:/Users/harsh/desktop/GreenLens/GreenLens/metadata/products.csv"
BARCODE_PATH = os.path.join(RAW_PATH, "barcodes")

# Categories and products
categories = {
    "fruits_veggies": ["banana","apple","orange","strawberry","grapes","carrot","tomato","broccoli","lettuce","bellpepper"],
    "snacks": ["lays","doritos","pringles","kitkat","snickers","oreo","mms","cheezit","goldfish","ritz"],
    "beverages": ["coke","pepsi","redbull","tropicana","dasani","frappuccino","gatorade","lipton","monster","lacroix"],
    "dairy": ["milk1","milk2","cheddar","yogurt","butter","mozzarella","creamcheese","halfhalf","greekyogurt","cottage"],
    "staples": ["rice","pasta","flour","sugar","salt","oil","peanutbutter","cereal","bread","oats"]
}

# Ensure processed and barcode folders exist
os.makedirs(PROCESSED_PATH, exist_ok=True)
os.makedirs(BARCODE_PATH, exist_ok=True)
for cat in categories:
    os.makedirs(os.path.join(PROCESSED_PATH, cat), exist_ok=True)
    os.makedirs(os.path.join(RAW_PATH, cat), exist_ok=True)

# Load existing products.csv
products = []
with open(CSV_PATH, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        products.append(row)

# Helper function: create 5 image variations
def create_variations(image_path, save_prefix):
    img = Image.open(image_path)
    for i in range(1,6):
        img_var = img.copy()
        angle = random.uniform(-10,10)
        img_var = img_var.rotate(angle, expand=True)
        img_var = ImageEnhance.Brightness(img_var).enhance(random.uniform(0.9,1.1))
        img_var = ImageEnhance.Contrast(img_var).enhance(random.uniform(0.9,1.1))
        w,h = img_var.size
        crop_pct = random.uniform(0,0.1)
        left = int(w*crop_pct)
        top = int(h*crop_pct)
        right = w-left
        bottom = h-top
        img_var = img_var.crop((left, top, right, bottom))
        img_var = img_var.resize((w,h))
        img_var.save(f"{save_prefix}_var{i}.jpg")

# Generate barcode images and update CSV
for row in products:
    pid = row['product_id']
    category = row['category']
    name = row['product_name']
    
    # Generate synthetic 12-digit barcode
    barcode_number = pid.zfill(12)
    row['barcode'] = barcode_number

    # Generate barcode image
    CODE128 = barcode.get_barcode_class('code128')
    code = CODE128(barcode_number, writer=ImageWriter())
    barcode_file = os.path.join(BARCODE_PATH, f"{pid}_{name}_barcode")
    code.save(barcode_file)

    # Generate image variations
    # Look for any original image matching the pattern: pid_name_img*.jpg
    orig_images = glob.glob(os.path.join(RAW_PATH, category, f"{pid}_{name}_img*.jpg"))
    if orig_images:
        for img_path in orig_images:
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            save_prefix = os.path.join(PROCESSED_PATH, category, base_name)
            create_variations(img_path, save_prefix)
    else:
        print(f"Warning: No images found for {pid}_{name} in {category}.")

# Save updated CSV
with open(CSV_PATH, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=products[0].keys())
    writer.writeheader()
    writer.writerows(products)

print("All images, barcodes, and CSV updated successfully!")
