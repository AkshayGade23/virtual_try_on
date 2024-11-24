import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import os
import numpy as np
# Load CLIP model
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


def get_image_embedding(image_path):
    """
    Generate an embedding for a given image using CLIP.
    Args:
        image_path (str): Path to the image.

    Returns:
        Tensor: Normalized image embedding.
    """
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        embedding = model.get_image_features(**inputs)
    return embedding / embedding.norm(dim=-1)  # Normalize the embedding

def load_inventory_embeddings(inventory_folder="inventory"):
    """
    Generate embeddings for all images in the inventory folder.
    Args:
        inventory_folder (str): Path to the inventory folder.

    Returns:
        list: List of tuples with image names and their embeddings.
    """
    inventory_embeddings = []
    for filename in os.listdir(inventory_folder):
        if filename.endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(inventory_folder, filename)
            embedding = get_image_embedding(image_path)
            inventory_embeddings.append((filename, embedding))
    return inventory_embeddings

def find_best_matches(generated_embedding, inventory_embeddings, top_n=3):
    """
    Find the top matching images from the inventory.
    Args:
        generated_embedding (Tensor): Embedding of the generated image.
        inventory_embeddings (list): List of inventory embeddings.
        top_n (int): Number of top matches to return.

    Returns:
        list: List of top matching inventory items and their similarity scores.
    """
    similarities = []
    for image_name, inventory_embedding in inventory_embeddings:
        similarity = torch.nn.functional.cosine_similarity(generated_embedding, inventory_embedding).item()
        similarities.append((image_name, similarity))
    sorted_matches = sorted(similarities, key=lambda x: x[1], reverse=True)
    return sorted_matches[:top_n]

invetory_embeddings = load_inventory_embeddings()
np.save('embeddings.npy',invetory_embeddings)
