'''Followed this guide initially to create the demo
https://medium.com/data-science-in-your-pocket/getting-started-with-dinov2-installation-setup-and-inference-made-easy-be37b07d32e7'''

from transformers import AutoImageProcessor, AutoModel, Dinov2ForImageClassification
import torch
from PIL import Image
import requests

# Load the pre-trained model and image processor
processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")

#model = Dinov2ForImageClassification.from_pretrained("facebook/dinov2-base") # <-- With classifier
model = AutoModel.from_pretrained("facebook/dinov2-base")  # <-- No classifier


# Set the model to evaluation mode
model.eval()

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

# Load an image
img = Image.open("eksempel_0002.jpg")

# Preprocess the image
inputs = processor(images=img, return_tensors="pt")

# Perform inference
'''with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = logits.argmax(-1).item()
'''

with torch.no_grad():
    outputs = model(**inputs)
    embedding = outputs.last_hidden_state[:, 0]  # CLS token embedding
    print("Embedding shape:", embedding.shape)

# Output the predicted class
''' print(f"Predicted class: {predicted_class}")



import matplotlib.pyplot as plt

# Visualize the image with segmentation
plt.imshow(img)
plt.title(f"Predicted class: {predicted_class}")
plt.show() '''