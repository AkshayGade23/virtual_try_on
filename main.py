import os
import gradio as gr
import base64
from gradio_prompt_tryon import upload_image  # Import the Feature 1 function
from gradio_client import Client, handle_file
import shutil
from PIL import Image

from promt_generation import  prompt_generation_similarity
# from image_embeddings import find_best_matches, get_image_embedding,load_inventory_embeddings


# inventory_embeddings =  load_inventory_embeddings()
# Feature 2: Process Images
def process_images(vton_img_path, garm_img_path, category):
    # Initialize the client
    client = Client("levihsu/OOTDiffusion")
    
    # Call the model
    result = client.predict(
        vton_img=handle_file(vton_img_path),
        garm_img=handle_file(garm_img_path),
        category=category,
        n_samples=1,
        n_steps=20,
        image_scale=2,
        seed=-1,
        api_name="/process_dc"
    )
    
    # Get the local file path from result
    output_image_path = result[0]["image"]
    
    # Define destination path
    destination_path = "output_image.webp"
    
    # Copy the image file to the current directory
    shutil.copy(output_image_path, destination_path)
    
    return destination_path


def create_ui():
    # Convert logo to Base64
    with open("icon.png", "rb") as logo_file:
        base64_logo = base64.b64encode(logo_file.read()).decode("utf-8")

    with gr.Blocks() as app:
        # Header Section with Logo and Tabs
        gr.HTML(f"""
        <div style="display: flex; align-items: center; justify-content: space-between; background-color: #288dd7; padding: 10px 20px; color: white; font-family: Arial, sans-serif; border-radius: 10px;">
            <div style="display: flex; align-items: center;">
                <img src="data:image/png;base64,{base64_logo}" alt="Logo" style="height: 50px; margin-right: 15px; border-radius: 5px;">
                <h2 style="margin: 0; font-size: 24px; color: white;">Virtual Try On</h2>
            </div>
            <div style="display: flex; gap: 20px; font-size: 18px; align-items: center;">
                <a href="#" style="color: white; text-decoration: none;">Home</a>
                <a href="#" style="color: white; text-decoration: none;">About</a>
                <a href="#" style="color: white; text-decoration: none;">Contact</a>
            </div>
        </div>
        """)
        gr.Image(value="banner.jpg", label="App Banner")

        # Tabs for Features
        with gr.Tabs():
            # Feature 1 Tab
            with gr.Tab("Feature 1"):
                gr.HTML("<h3 style='text-align: center;'>Feature 1: Image Generation and Similarity Search</h3>")
                gr.Interface(
                    fn=upload_image,
                    inputs=[
                        gr.Image(type="filepath", label="Upload an Image"),
                        gr.Textbox(label="Enter a prompt for image generation"),
                    ],
                    outputs=[
                        gr.Image(type="filepath", label="Generated Image"),
                        gr.Gallery(label="Similar Images"),
                    ],
                    live=True
                )

            # Feature 2 Tab
            with gr.Tab("Feature 2"):
                gr.Markdown("## Prompt-Driven Virtual Try-On")

                with gr.Row():
                    vton_img = gr.Image(label="Upload Your Picture", type="filepath")
                    garm_img = gr.Image(label="Upload Outfit Image", type="filepath")
                    output_image = gr.Image(label="Output Image")

                with gr.Row():
                    category = gr.Dropdown(
                        choices=["Upper-body", "Lower-body"],
                        label="Select Category"
                    )
                    run_button = gr.Button("Generate", elem_id="generate-btn")

                run_button.click(
                    fn=process_images,
                    inputs=[vton_img, garm_img, category],
                    outputs=output_image
                )
            
            # Feature 3 Tab
            with gr.Tab("Feature 3"):
                gr.Markdown("## Prompt-Driven Image Generation with Similarity Search")
                
                with gr.Row():


                    prompt_input = gr.Textbox(label="Enter a prompt for image generation")
                    gender_option = gr.Dropdown(
                        choices=["Male", "Female"],
                        label="Select Gender"
                    )
                    age_group_option = gr.Dropdown(
                        choices=["Child", "Young", "Adult"],
                        label="Select Age Group"
                    )

                with gr.Row():
                    generate_button = gr.Button("Generate", elem_id="generate-btnn")

                output_image = gr.Image(label="Generated Image")
                similar_images = gr.Gallery(label="Similar Images")

                generate_button.click(
                    fn=prompt_generation_similarity,
                    inputs=[prompt_input, gender_option, age_group_option],
                    outputs=[output_image, similar_images]
                )

        # Footer
        gr.HTML("""
        <div style="text-align: center; margin-top: 20px; color: #777; font-size: 14px;">
            <hr style="border: 1px solid #eee; margin-bottom: 10px;">
            &copy; 2024 Group 15. All Rights Reserved.
        </div>
        """)
    return app



if __name__ == "__main__":
    # inventory_embeddings = load_inventory_embeddings()
    app = create_ui()
    app.launch(server_name="127.0.0.1", server_port=7860, share=False)
