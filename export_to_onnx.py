import torch
import torch.nn as nn
from torchvision import models
import json
import os

def convert_model_to_onnx():
    """Convert the trained PyTorch model to ONNX format for deployment"""
    
    # Load the trained model checkpoint
    model_path = 'product_classifier_model.pth'
    if not os.path.exists(model_path):
        print("❌ Model file not found. Please train the model first.")
        return
    
    print("Loading trained model...")
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Create model architecture
    model = models.mobilenet_v3_small(pretrained=False)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, checkpoint['num_classes'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create dummy input for ONNX export
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Export to ONNX
    onnx_path = 'greenlens_model.onnx'
    print(f"Exporting to ONNX format: {onnx_path}")
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    # Create metadata file for the app
    metadata = {
        'model_info': {
            'input_shape': [1, 3, 224, 224],
            'input_name': 'input',
            'output_name': 'output',
            'num_classes': checkpoint['num_classes'],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225]
        },
        'class_mapping': checkpoint['idx_to_class'],
        'carbon_footprint': {
            # Sample carbon data - replace with actual data from products.csv
            'Apple': {'co2_kg': 0.15, 'eco_rating': 'low', 'alternative': 'Banana'},
            'Banana': {'co2_kg': 0.12, 'eco_rating': 'low', 'alternative': 'Apple'},
            'Coca-Cola_12oz_Can': {'co2_kg': 0.20, 'eco_rating': 'medium', 'alternative': 'Dasani_Water_16oz'},
            'Doritos_Nacho_Cheese': {'co2_kg': 0.28, 'eco_rating': 'high', 'alternative': 'Apple'},
            'Whole_Milk_1_Gallon': {'co2_kg': 0.30, 'eco_rating': 'medium', 'alternative': '2%_Milk_1_Gallon'}
        }
    }
    
    # Load actual carbon data from products.csv if available
    try:
        import pandas as pd
        products_df = pd.read_csv('metadata/products.csv')
        carbon_data = {}
        
        for _, row in products_df.iterrows():
            product_name = row['product_name'].replace(' ', '_').replace('(', '').replace(')', '')
            if '&' in product_name:
                product_name = product_name.replace('&', '_')
            
            # Determine eco rating based on CO2
            co2_value = float(row['co2_kg'])
            if co2_value <= 0.15:
                eco_rating = 'low'
            elif co2_value <= 0.25:
                eco_rating = 'medium'
            else:
                eco_rating = 'high'
            
            carbon_data[product_name] = {
                'co2_kg': co2_value,
                'eco_rating': eco_rating,
                'alternative': row.get('alternative_1', 'N/A'),
                'category': row['category']
            }
        
        metadata['carbon_footprint'] = carbon_data
        print(f"✅ Loaded carbon data for {len(carbon_data)} products")
        
    except Exception as e:
        print(f"⚠️ Using sample carbon data: {e}")
    
    # Save metadata
    with open('model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Model exported successfully!")
    print(f"   ONNX Model: {onnx_path}")
    print(f"   Metadata: model_metadata.json")
    print(f"   Classes: {checkpoint['num_classes']}")
    print(f"   Best Accuracy: {checkpoint.get('best_acc', 'N/A')}")

if __name__ == '__main__':
    convert_model_to_onnx()