import gradio as gr
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import os
import requests
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model
from keras.utils import load_img, img_to_array
from sklearn.metrics.pairwise import cosine_similarity

from image_embeddings import get_image_embedding, find_best_matches

# Global variables for Tkinter functionality
rectangles = []
masked_image_path = ''
canvas = None
img_tk = None
rect_id = None
start_x = start_y = 0

STABILITY_KEY = ''
default_folder = 'white'  # Default folder to find similar images


def select_rectangle(event):
    global start_x, start_y, rect_id
    start_x = event.x
    start_y = event.y
    rect_id = canvas.create_rectangle(start_x, start_y, start_x, start_y, outline='red')

def update_rectangle(event):
    canvas.coords(rect_id, start_x, start_y, event.x, event.y)

def save_rectangle(event):
    global rectangles
    end_x = event.x
    end_y = event.y
    rectangles.append((start_x, start_y, end_x, end_y))
    canvas.unbind("<B1-Motion>")
    canvas.unbind("<ButtonRelease-1>")
    canvas.bind("<Button-1>", select_rectangle)
    canvas.bind("<B1-Motion>", update_rectangle)
    canvas.bind("<ButtonRelease-1>", save_rectangle)

def reset_mask():
    global rectangles
    rectangles = []
    canvas.delete("all")
    canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)

def send_generation_request(host, params):
    headers = {
        "Accept": "image/*",
        "Authorization": f"Bearer {STABILITY_KEY}"
    }

    files = {}
    image = params.pop("image", None)
    mask = params.pop("mask", None)
    if image is not None and image != '':
        files["image"] = open(image, 'rb')
    if mask is not None and mask != '':
        files["mask"] = open(mask, 'rb')
    if len(files) == 0:
        files["none"] = ''

    print(f"Sending REST request to {host}...")
    response = requests.post(
        host,
        headers=headers,
        files=files,
        data=params
    )
    if not response.ok:
        raise Exception(f"HTTP {response.status_code}: {response.text}")

    return response

def edit_image_with_stability(image_path, prompt, output_format='jpeg', strength=0.75, seed=0):
    global edited_image_path

    host = f"https://api.stability.ai/v2beta/stable-image/edit/inpaint"

    params = {
        "image": image_path,
        "prompt": prompt,
        "strength": strength,
        "seed": seed,
        "output_format": output_format,
        "mode": "image-to-image",
        "model": "sd3-medium"
    }

    response = send_generation_request(host, params)

    output_image = response.content
    finish_reason = response.headers.get("finish-reason")
    seed = response.headers.get("seed")

    if finish_reason == 'CONTENT_FILTERED':
        raise Warning("Generation failed NSFW classifier")

    filename, _ = os.path.splitext(os.path.basename(image_path))
    edited_image_path = f"generated_{seed}.{output_format}"
    with open(edited_image_path, "wb") as f:
        f.write(output_image)
    print(f"Saved image {edited_image_path}")

    return edited_image_path

def done_masking():
    global masked_image_path  # Use the global masked image path for continuity
    create_masked_image(mask_window.image_path)  # Save the masked image

    try:
        # Define prompt and other parameters for image generation
        prompt = "style outfit in a western dress"  # Update the prompt as needed
        strength = 0.75
        seed = 42  # Optional: Use a specific seed for reproducibility

        # Call the Stability API with the masked image
        edited_image = edit_image_with_stability(
            image_path=masked_image_path,  # Use the masked image as input
            prompt=prompt,
            output_format="jpeg",  # Change to desired format
            strength=strength,
            seed=seed
        )

        print(f"Edited image saved at: {edited_image}")
    except Exception as e:
        print(f"Error during image editing: {e}")

    mask_window.destroy()  # Close the Tkinter window


def create_masked_image(image_path):
    global masked_image_path

    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

    canvas_width, canvas_height = canvas.winfo_width(), canvas.winfo_height()
    img_width, img_height = img.shape[1], img.shape[0]
    scale_x = img_width / canvas_width
    scale_y = img_height / canvas_height

    for rect in rectangles:
        start_x, start_y, end_x, end_y = rect
        start_x = int(start_x * scale_x)
        start_y = int(start_y * scale_y)
        end_x = int(end_x * scale_x)
        end_y = int(end_y * scale_y)
        start_x, start_y = max(start_x, 0), max(start_y, 0)
        end_x, end_y = min(end_x, img.shape[1]), min(end_y, img.shape[0])
        img[start_y:end_y, start_x:end_x, 3] = 0

    masked_image_path = 'output_image.png'
    cv2.imwrite(masked_image_path, img)
    print(f"Image saved to {masked_image_path}")
    return masked_image_path

def show_masking_window(image_path):
    global mask_window, canvas, img_tk
    rectangles.clear()
    mask_window = tk.Tk()
    mask_window.title("Choose a section to mask")
    mask_window.image_path = image_path

    # Bring the window to the front
    mask_window.attributes('-topmost', True)
    mask_window.update()
    mask_window.attributes('-topmost', False)

    img = Image.open(image_path)
    img.thumbnail((300, 300))
    img_tk = ImageTk.PhotoImage(img)
    canvas = tk.Canvas(mask_window, width=img.width, height=img.height)
    canvas.pack()
    canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)

    canvas.bind("<Button-1>", select_rectangle)
    canvas.bind("<B1-Motion>", update_rectangle)
    canvas.bind("<ButtonRelease-1>", save_rectangle)

    frame_buttons = tk.Frame(mask_window)
    frame_buttons.pack(fill=tk.X, padx=10, pady=10)

    btn_reset = ttk.Button(frame_buttons, text="Reset", command=reset_mask)
    btn_reset.pack(side=tk.LEFT, padx=5)

    btn_done = ttk.Button(frame_buttons, text="Done", command=done_masking)
    btn_done.pack(side=tk.RIGHT, padx=5)

    mask_window.mainloop()


def upload_image(image_path, prompt):
    global default_folder
    default_folder = 'white'  # Ensure the default folder is set
    
    # Step 1: Show the masking window for the user to mask parts of the image
    show_masking_window(image_path)

    # Step 2: Generate the modified image using Stability API
    try:
        strength = 0.75
        seed = 42  # Optional: seed for reproducibility

        # Generate the new image
        generated_image_path = edit_image_with_stability(
            image_path=masked_image_path,
            prompt=prompt,
            output_format="jpeg",
            strength=strength,
            seed=seed
        )

        print(f"Generated image saved at: {generated_image_path}")
    except Exception as e:
        print(f"Error during image generation: {e}")
        return "Error during image generation.", None  # Return an error message if needed

    # Step 3: Perform image similarity search
    # similar_images = find_similar_images(generated_image_path, default_folder)
    generated_embedding = get_image_embedding(generated_image_path)

    # Find best matching inventory items
    inventory_embeddings = np.load('embeddings.npy')
    best_matches = find_best_matches(generated_embedding, inventory_embeddings)

    # Prepare gallery of matching images
    matched_images = []
    for match in best_matches:
        matched_image_path = f"inventory/{match[0]}"
        matched_image = Image.open(matched_image_path)
        matched_images.append((matched_image, f"Similarity: {match[1]:.2f}"))

    return generated_image_path, matched_images  # Return both the generated image and similar images

# Function to load and preprocess the image
def load_and_preprocess_image(img_path, target_size=(224, 224)):
    img = load_img(img_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

# Function to extract features using VGG16
def extract_features(model, img_path):
    img_array = load_and_preprocess_image(img_path)
    features = model.predict(img_array)
    return features.flatten()

# Function to find similar images
def find_similar_images(reference_img_path, folder_path, top_n=9):
    base_model = VGG16(weights='imagenet')
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)

    reference_features = extract_features(model, reference_img_path)

    image_paths = [os.path.join(folder_path, fname) for fname in os.listdir(folder_path) if fname.lower().endswith(('png', 'jpg', 'jpeg'))]
    similarities = []

    for img_path in image_paths:
        features = extract_features(model, img_path)
        similarity = cosine_similarity([reference_features], [features])[0][0]
        similarities.append((img_path, similarity))

    similarities.sort(key=lambda x: x[1], reverse=True)
    similar_images = [img[0] for img in similarities[:top_n]]

    return similar_images

# ui = gr.Interface(
#     fn=upload_image,
#     inputs=[
#         gr.Image(type="filepath", label="Upload an Image"),  # Image upload field
#         gr.Textbox(label="Enter a prompt for image generation")  # Input for prompt
#     ],
#     outputs=[
#         gr.Image(type="filepath", label="Generated Image"),  # Display the generated image
#         gr.Gallery(label="Similar Images")  # Display similar images in a gallery
#     ],
#     title="Image Generation and Similarity Search",
#     description="Upload an image, select regions to mask, provide a prompt, and generate a modified version of the image. Find similar images from a predefined folder."
# )

# ui.launch()


