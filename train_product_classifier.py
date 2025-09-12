import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import time
import json
from collections import defaultdict
import pandas as pd

# --- Configuration ---
DATA_DIR = 'dataset_products'
NUM_EPOCHS = 20  # Number of times to loop through the training data
BATCH_SIZE = 16  # Number of images to process at once
LEARNING_RATE = 0.001
MOMENTUM = 0.9

def get_class_mapping():
    """Create a mapping from product names to class indices"""
    # Read the products.csv to get all unique products
    products_df = pd.read_csv('metadata/products.csv')
    product_names = sorted(products_df['product_name'].unique())
    
    # Create mapping from product name to class index
    class_to_idx = {name: idx for idx, name in enumerate(product_names)}
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}
    
    print(f"Found {len(product_names)} unique products:")
    for i, name in enumerate(product_names):
        print(f"  {i}: {name}")
    
    return class_to_idx, idx_to_class, len(product_names)

def create_dataset_info():
    """Analyze the dataset structure and create info"""
    train_dir = os.path.join(DATA_DIR, 'train')
    val_dir = os.path.join(DATA_DIR, 'validate')
    
    # Get all product folders
    train_products = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    val_products = [d for d in os.listdir(val_dir) if os.path.isdir(os.path.join(val_dir, d))]
    
    # Count images per product
    train_counts = {}
    val_counts = {}
    
    for product in train_products:
        product_path = os.path.join(train_dir, product)
        image_files = [f for f in os.listdir(product_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        train_counts[product] = len(image_files)
    
    for product in val_products:
        product_path = os.path.join(val_dir, product)
        image_files = [f for f in os.listdir(product_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        val_counts[product] = len(image_files)
    
    print("\nDataset Analysis:")
    print("=" * 50)
    print(f"Training products: {len(train_products)}")
    print(f"Validation products: {len(val_products)}")
    print(f"Total training images: {sum(train_counts.values())}")
    print(f"Total validation images: {sum(val_counts.values())}")
    
    return train_counts, val_counts

def train_model():
    print("Initializing product classifier training...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Get class mapping and dataset info
    class_to_idx, idx_to_class, num_classes = get_class_mapping()
    train_counts, val_counts = create_dataset_info()

    # 1. Define data transformations
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'validate': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # 2. Load the datasets
    print(f"\nLoading datasets from {DATA_DIR}...")
    image_datasets = {x: datasets.ImageFolder(os.path.join(DATA_DIR, x), data_transforms[x])
                      for x in ['train', 'validate']}
    
    # Update the class_to_idx mapping to match the actual folder structure
    actual_class_to_idx = image_datasets['train'].class_to_idx
    actual_idx_to_class = {v: k for k, v in actual_class_to_idx.items()}
    
    print(f"Actual classes found in dataset: {len(actual_class_to_idx)}")
    print("Class mapping:")
    for idx, class_name in actual_idx_to_class.items():
        print(f"  {idx}: {class_name}")

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], 
                                                 batch_size=BATCH_SIZE, 
                                                 shuffle=True, 
                                                 num_workers=4,
                                                 pin_memory=True)
                   for x in ['train', 'validate']}
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'validate']}
    class_names = image_datasets['train'].classes
    
    print(f"\nDataset sizes:")
    print(f"  Training: {dataset_sizes['train']} images")
    print(f"  Validation: {dataset_sizes['validate']} images")

    # 3. Load the pre-trained model
    print(f"\nLoading pre-trained MobileNetV3...")
    model = models.mobilenet_v3_small(pretrained=True)
    
    # Replace the final layer to match the number of product classes
    num_ftrs = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(num_ftrs, len(class_names))
    model = model.to(device)

    # 4. Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # 5. Training loop
    print(f"\nStarting training for {NUM_EPOCHS} epochs...")
    print("=" * 60)
    
    since = time.time()
    best_acc = 0.0
    best_model_wts = None
    training_history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(NUM_EPOCHS):
        print(f'\nEpoch {epoch+1}/{NUM_EPOCHS}')
        print('-' * 40)

        for phase in ['train', 'validate']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for batch_idx, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Store training history
            if phase == 'train':
                training_history['train_loss'].append(epoch_loss)
                training_history['train_acc'].append(epoch_acc.item())
            else:
                training_history['val_loss'].append(epoch_loss)
                training_history['val_acc'].append(epoch_acc.item())

            # Save best model
            if phase == 'validate' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict().copy()
                print("✨ New best model found!")

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning rate: {current_lr:.6f}")

    time_elapsed = time.time() - since
    print(f'\nTraining completed in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best Validation Accuracy: {best_acc:.4f}')

    # 6. Save the best model and metadata
    print(f"\nSaving model and metadata...")
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_to_idx': actual_class_to_idx,
        'idx_to_class': actual_idx_to_class,
        'num_classes': len(class_names),
        'best_acc': best_acc.item(),
        'training_history': training_history
    }, 'product_classifier_model.pth')
    
    # Save class mapping as JSON for easy loading
    with open('class_mapping.json', 'w') as f:
        json.dump({
            'class_to_idx': actual_class_to_idx,
            'idx_to_class': actual_idx_to_class
        }, f, indent=2)
    
    print("✅ Model saved as 'product_classifier_model.pth'")
    print("✅ Class mapping saved as 'class_mapping.json'")
    
    # Print final statistics
    print(f"\nFinal Statistics:")
    print(f"  Total products: {len(class_names)}")
    print(f"  Best validation accuracy: {best_acc:.4f}")
    print(f"  Training images: {dataset_sizes['train']}")
    print(f"  Validation images: {dataset_sizes['validate']}")

def create_inference_script():
    """Create a simple inference script for testing the trained model"""
    inference_code = '''import torch
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
'''
    
    with open('inference.py', 'w') as f:
        f.write(inference_code)
    
    print("✅ Inference script created as 'inference.py'")

if __name__ == '__main__':
    try:
        train_model()
        create_inference_script()
        print("\n🎉 Training completed successfully!")
        print("You can now use 'inference.py' to test the trained model.")
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
