import os
import csv
import random
import barcode
from barcode.writer import ImageWriter

# Paths
CSV_PATH = r"C:/Users/harsh/desktop/GreenLens/GreenLens/metadata/products.csv"
BARCODE_PATH = r"C:/Users/harsh/desktop/GreenLens/GreenLens/dataset/raw/barcodes"
os.makedirs(BARCODE_PATH, exist_ok=True)

# Load CSV
products = []
with open(CSV_PATH, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        products.append(row)

# Keep track of already used barcodes
used_barcodes = set()

# Helper to generate unique 12-digit barcode
def generate_unique_barcode():
    while True:
        barcode_num = "".join([str(random.randint(0, 9)) for _ in range(12)])
        if barcode_num not in used_barcodes:
            used_barcodes.add(barcode_num)
            return barcode_num

# Generate barcode images and update CSV
for row in products:
    barcode_number = generate_unique_barcode()
    row['barcode'] = barcode_number

    # Create barcode image
    CODE128 = barcode.get_barcode_class('code128')
    code = CODE128(barcode_number, writer=ImageWriter())
    barcode_file = os.path.join(BARCODE_PATH, f"{row['product_id']}_{row['product_name']}_barcode")
    code.save(barcode_file)

# Save updated CSV
with open(CSV_PATH, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=products[0].keys())
    writer.writeheader()
    writer.writerows(products)

print("All barcodes generated and CSV updated successfully!")