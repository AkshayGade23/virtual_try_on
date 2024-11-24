import requests
import os
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from image_embeddings import get_image_embedding, find_best_matches
import numpy as np

# from main  import inventory_embeddings
# Replace with your Stability AI API key
API_KEY = ""

def generate_image(prompt, save_path="static/generated/generated_image.jpeg"):
    """
    Generate an image using Stability AI's API.
    Args:
        prompt (str): Text prompt for image generation.
        save_path (str): Path to save the generated image.

    Returns:
        str: Path of the saved image if successful, raises an exception otherwise.
    """
    # Define the API endpoint
    url = "https://api.stability.ai/v2beta/stable-image/generate/sd3"

    # Headers for the API request
    headers = {
        "authorization": f"Bearer {API_KEY}",
        "accept": "image/*",
    }

    # Data for the request
    data = {
        "prompt": prompt,
        "output_format": "jpeg",  # Set output format as JPEG
    }

    # Send the POST request
    response = requests.post(url, headers=headers, files={"none": ''},data=data)

    # Check the response status
    if response.status_code == 200:
        # Save the image
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as file:
            file.write(response.content)
        return save_path
    else:
        # Raise an exception for unsuccessful requests
        raise Exception(f"Error: {response.status_code}, Details: {response.json()}")
    




def  prompt_generation_similarity(prompt,gender,age):

    # Combine gender and age into the prompt
    complete_prompt = f"{age} {gender} wearing {prompt}"

    # Generate the image
    generated_image_path = generate_image(complete_prompt)
    generated_image = Image.open(generated_image_path)

    # Get embedding for generated image
    generated_embedding = get_image_embedding(generated_image_path)
    inventory_embeddings = np.load('embeddings.npy')
    # Find best matching inventory items
    best_matches = find_best_matches(generated_embedding, inventory_embeddings)

    # Prepare gallery of matching images
    matched_images = []
    for match in best_matches:
        matched_image_path = f"inventory/{match[0]}"
        matched_image = Image.open(matched_image_path)
        matched_images.append((matched_image, f"Similarity: {match[1]:.2f}"))

    return generated_image, matched_images
