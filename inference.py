import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json
import os

class ProductClassifier:
    def __init__(self, model_path='product_classifier_model.pth'):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.load_model(model_path)
        
    def load_model(self, model_path):
        # Load saved model
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Create model architecture
        model = models.mobilenet_v3_small(pretrained=False)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, checkpoint['num_classes'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        self.model = model
        self.class_to_idx = checkpoint['class_to_idx']
        self.idx_to_class = checkpoint['idx_to_class']
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
    def predict(self, image_path, top_k=5):
        """Predict the product class for an image"""
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            top_probs, top_indices = torch.topk(probabilities, top_k)
            
        results = []
        for i in range(top_k):
            class_name = self.idx_to_class[top_indices[i].item()]
            confidence = top_probs[i].item()
            results.append((class_name, confidence))
            
        return results

# Example usage
if __name__ == "__main__":
    classifier = ProductClassifier()
    
    # Test with a sample image
    if os.path.exists('dataset_products/validate'):
        sample_dirs = os.listdir('dataset_products/validate')
        if sample_dirs:
            sample_dir = sample_dirs[0]
            sample_images = [f for f in os.listdir(f'dataset_products/validate/{sample_dir}') 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if sample_images:
                sample_path = f'dataset_products/validate/{sample_dir}/{sample_images[0]}'
                print(f"Testing with: {sample_path}")
                results = classifier.predict(sample_path)
                print("Top predictions:")
                for i, (class_name, confidence) in enumerate(results):
                    print(f"  {i+1}. {class_name}: {confidence:.4f}")
