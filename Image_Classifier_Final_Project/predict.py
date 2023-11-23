import torch
from torchvision import models, transforms
from PIL import Image
import json
import numpy as np

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = models.vgg16(pretrained=True)
    print(checkpoint.keys())
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model

def process_image(image_path):
    # Open the image
    img = Image.open(image_path)
    
    # Define transformations for the image
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Preprocess the image
    img_tensor = preprocess(img)
    
    # Convert to NumPy array
    img_numpy = np.array(img_tensor)
    
    return img_numpy

def predict(image_path, model, topk=5):
    model.eval()
    img = process_image(image_path)
    
    # Convert NumPy array to PyTorch Tensor
    img_tensor = torch.from_numpy(img)
    img_tensor = img_tensor.unsqueeze(0) # Add batch size of 1
    
    # Predict the class probabilities
    with torch.no_grad():
        output = model(img_tensor)
    
    # Calculate probabilities and class indices
    probabilities = torch.exp(output).topk(topk)[0][0].numpy()
    class_indices = torch.exp(output).topk(topk)[1][0].numpy()
    
    # Convert class indices to class labels using class_to_idx mapping
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_classes = [idx_to_class[idx] for idx in class_indices]
    
    return probabilities, top_classes

def visualize_prediction(image_path, model, cat_to_name):
    # Make predictions
    probabilities, top_classes = predict(image_path, model)
    
    # Map class indices to class names
    class_names = [cat_to_name[idx] for idx in top_classes]
    
    # Display the prediction results
    print("Predicted Flower Classes:")
    for i in range(len(class_names)):
        print(f"{class_names[i]}: {probabilities[i]*100:.2f}%")
    
# Load cat_to_name dictionary
with open('/home/workspace/ImageClassifier/cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

# Specify the path to the saved checkpoint file
checkpoint_path = '/home/workspace/ImageClassifier/checkpoint.pth'

# Load the trained model from the checkpoint
model = load_checkpoint(checkpoint_path)

# Specify the path to the test image you want to predict
test_image_path = '/home/workspace/ImageClassifier/flowers/test/2/image_05100.jpg'

# Visualize predictions for the test image
visualize_prediction(test_image_path, model, cat_to_name)
