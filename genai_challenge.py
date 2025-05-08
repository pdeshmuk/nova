import time
import json
import base64
from io import BytesIO
import boto3
import streamlit as st
from PIL import Image
from datetime import datetime
import io

from streamlit import session_state as state

REGION = "us-east-1"

# List of Stable Diffusion Preset Styles
sd_presets = [
    "None",
    "3d-model",
    "analog-film",
    "anime",
    "cinematic",
    "comic-book",
    "digital-art",
    "enhance",
    "fantasy-art",
    "isometric",
    "line-art",
    "low-poly",
    "modeling-compound",
    "neon-punk",
    "origami",
    "photographic",
    "pixel-art",
    "tile-texture",
]

# bucket name
S3_BUCKET_NAME = "aci-techexpo-genaichallenge"  

# Define bedrock
bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name=REGION,
)

def base64_to_image(base64_string):
    """
    Convert base64 string to an image that can be displayed in Streamlit
    
    Args:
        base64_string: Base64 encoded image string
    Returns:
        PIL Image object
    """
    # Decode base64 string to bytes
    image_bytes = base64.b64decode(base64_string)
    # Convert bytes to PIL Image
    image = Image.open(BytesIO(image_bytes))
    return image

# generate_image_sd, generate_image_titan, & generate_image_nova functions here...
# Bedrock api call to stable diffusion
def generate_image_sd(text, style):
    """
    Purpose:
        Uses Bedrock API to generate an Image
    Args/Requests:
         text: Prompt
         style: style for image
    Return:
        image: base64 string of image
    """
    body = {
        "text_prompts": [{"text": text}],
        "cfg_scale": 10,
        "seed": 0,
        "steps": 50,
        "style_preset": style,
    }

    if style == "None":
        del body["style_preset"]

    body = json.dumps(body)

    modelId = "stability.stable-diffusion-xl-v1"
    accept = "application/json"
    contentType = "application/json"

    try:
        response = bedrock_runtime.invoke_model(
            body=body, modelId=modelId, accept=accept, contentType=contentType
        )
        response_body = json.loads(response.get("body").read())

        results = response_body.get("artifacts")[0].get("base64")
        return results
    except Exception as e:
        st.sidebar.error(f"Error with model {modelId}: {str(e)}")

# Bedrock api call to titan
def generate_image_titan(text):
    """
    Purpose:
        Uses Bedrock API to generate an Image using Titan
    Args/Requests:
         text: Prompt
    Return:
        image: base64 string of image
    """
    body = {
        "textToImageParams": {"text": text},
        "taskType": "TEXT_IMAGE",
        "imageGenerationConfig": {
            "cfgScale": 8.0,
            "seed": 0,
            "quality": "standard",
            "width": 1024,
            "height": 1024,
            "numberOfImages": 1,
        },
    }

    body = json.dumps(body)

    modelId = "amazon.titan-image-generator-v2:0"
    #modelId = "amazon.titan-image-generator-v1"
   
    accept = "application/json"
    contentType = "application/json"

    try: 
        response = bedrock_runtime.invoke_model(
            body=body, modelId=modelId, accept=accept, contentType=contentType
        )
        response_body = json.loads(response.get("body").read())

        results = response_body.get("images")[0]
        return results
    except Exception as e:
        st.sidebar.error(f"Error with model {modelId}: {str(e)}")
        # Fallback to v1 if v2 fails
        modelId = "amazon.titan-image-generator-v1"
        st.sidebar.write(f"Trying fallback model: {modelId}")
        response = bedrock_runtime.invoke_model(
            body=body,
            modelId=modelId,
            accept=accept,
            contentType=contentType
        )
        response_body = json.loads(response.get("body").read())
        return response_body.get("images")[0]

# Bedrock api call to nova
def generate_image_nova(text):
    """
    Purpose:
        Uses Bedrock API to generate an Image using Nova
    Args/Requests:
         text: Prompt
    Return:
        image: base64 string of image
    """
    body = {
        "textToImageParams": {"text": text},
        "taskType": "TEXT_IMAGE",
        "imageGenerationConfig": {
            "cfgScale": 8.0,
            "seed": 42,
            "quality": "standard",
            "width": 1024,
            "height": 1024,
            "numberOfImages": 1,
        },
    }

    body = json.dumps(body)

    modelId = "amazon.nova-canvas-v1:0"
    
    accept = "application/json"
    contentType = "application/json"

    try:
        response = bedrock_runtime.invoke_model(
                body=body, modelId=modelId, accept=accept, contentType=contentType
            )
        response_body = json.loads(response.get("body").read())

        results = response_body.get("images")[0]
        return results
    except Exception as e:
        st.sidebar.error(f"Error with model {modelId}: {str(e)}")


# Add these functions after your existing functions
def save_image(image):
    """
    Save the PIL Image object with a timestamp-based filename
    
    Args:
        image: PIL Image object
    Returns:
        bytes: Image in bytes format for download
    """
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return buffered.getvalue()

def save_to_s3(image_bytes, bucket_name, filename):
    """
    Save image to S3 bucket
    """
    try:
        s3_client = boto3.client('s3', region_name=REGION)
        response = s3_client.put_object(
            Bucket=bucket_name,
            Key=filename,
            Body=image_bytes,
            ContentType='image/png'
        )        
        return True
    except Exception as e:
        st.error(f"Error saving to S3: {str(e)}")
        return False

def list_images_in_s3(bucket_name, prefix='images/'):
    """
    List all images in the specified S3 bucket and prefix
    """
    try:
        s3_client = boto3.client('s3', region_name=REGION)
        paginator = s3_client.get_paginator('list_objects_v2')
        
        image_list = []
        for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
            if 'Contents' in page:
                for obj in page['Contents']:
                    if obj['Key'].lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_list.append(obj['Key'])
        
        return image_list
    except Exception as e:
        st.error(f"Error listing images from S3: {str(e)}")
        return []

@st.cache_data(ttl=300)
def get_image_from_s3(bucket_name, key):
    """
    Retrieve image from S3 and convert to PIL Image
    """
    try:
        s3_client = boto3.client('s3', region_name=REGION)
        response = s3_client.get_object(Bucket=bucket_name, Key=key)
        image_bytes = response['Body'].read()
        return Image.open(BytesIO(image_bytes))
    except Exception as e:
        st.error(f"Error retrieving image from S3: {str(e)}")
        return None

# UI Elements
st.set_page_config(layout="wide")
#st.logo("awslogo.png")
st.image("awslogo.png")

st.title("ACI Tech Expo")  # Title of the application

st.subheader("AWS GenAI Image Generation Challenge")

model = st.selectbox("Select model", ["Nova Canvas", "Amazon Titan", "Stable Diffusion"])

# Style selection for Stable Diffusion
style = None
if model == "Stable Diffusion":
    style = st.selectbox("Select style preset:", sd_presets)


# Text inputs for prompt, fname, lname
prompt = st.text_input("Enter your image prompt:")
fname = st.text_input("Enter your first name:")
lname = st.text_input("Enter your last name:")

# Check if all required fields are filled
all_fields_filled = (
    prompt.strip() != "" and 
    fname.strip() != "" and 
    lname.strip() != ""
)

# Show warning if any field is empty
if not all_fields_filled:
    st.warning("* All fields are required")

# Generate button section - disabled if any field is empty
if st.button("Generate Image", disabled=not all_fields_filled):
    with st.spinner("Generating image..."):
        try:
            if model == "Nova Canvas":
                image_base64 = generate_image_nova(prompt)
            elif model == "Amazon Titan":
                image_base64 = generate_image_titan(prompt)
            else:  # Stable Diffusion
                image_base64 = generate_image_sd(prompt, style)
            
            # Convert base64 to image, display outside the generate button block
            image = base64_to_image(image_base64)
            
            # Create timestamp for filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{fname}_{lname}_{timestamp}.png"
            
            # Get image bytes
            image_bytes = save_image(image)
            
            # Store the image bytes and filename in session state
            state.image_bytes = image_bytes
            state.filename = filename
            state.image_base64 = image_base64
                        
        except Exception as e:
            st.error(f"Error generating image: {str(e)}")


# Handle the S3 save & download operations independently of the Generate button
if 'image_bytes' in state and 'filename' in state:

    # Convert base64 to image and display
    image = base64_to_image(state.image_base64)
    st.image(image, caption="Generated Image", use_container_width=True)
    #st.image(image, caption="Generated Image", use_column_width=True)

    # Create columns for buttons
    col1, col2 = st.columns(2)
            
    # Download button in first column
    with col1:
        st.download_button(
            label="Download Image",
            data=state.image_bytes,
            file_name=state.filename,
            mime="image/png"
        )
            
    # Save to S3 button in second column
    with col2:
        if st.button("Save to S3"):
            if save_to_s3(state.image_bytes, S3_BUCKET_NAME, f"images/{state.filename}"):
                st.success(f"Image saved to S3 bucket: {state.filename}")
            else:
                st.error("Failed to save image to S3")



st.markdown("---")  # Add a visual separator
st.subheader("Image Carousel")

# Initialize session states if not exists
if 'image_index' not in st.session_state:
    st.session_state.image_index = 0
if 'loaded_images' not in st.session_state:
    st.session_state.loaded_images = {}

# Get list of images
image_list = list_images_in_s3(S3_BUCKET_NAME)

if image_list:
    # Create placeholder for image
    image_placeholder = st.empty()
    
    # Create columns for navigation
    col1, col2, col3 = st.columns([1, 6, 1])
    
    # Add navigation buttons
    with col1:
        if st.button("⬅️"):
            st.session_state.image_index = (st.session_state.image_index - 1) % len(image_list)
    
    with col3:
        if st.button("➡️"):
            st.session_state.image_index = (st.session_state.image_index + 1) % len(image_list)
    
    # Add auto-scroll toggle
    auto_scroll = st.checkbox("Enable auto-scroll", value=True)
    scroll_speed = st.slider("Scroll Speed (seconds)", min_value=1, max_value=10, value=3)
    
    # Display current image index
    st.caption(f"Image {st.session_state.image_index + 1} of {len(image_list)}")
    
    # Function to display current image
    def display_current_image():
        current_image_key = image_list[st.session_state.image_index]
        
        # Check if image is already loaded in session state
        if current_image_key not in st.session_state.loaded_images:
            # Load image and store in session state
            image = get_image_from_s3(S3_BUCKET_NAME, current_image_key)
            if image:
                st.session_state.loaded_images[current_image_key] = image
        
        # Get image from session state
        if current_image_key in st.session_state.loaded_images:
            image = st.session_state.loaded_images[current_image_key]
            filename = current_image_key.split('/')[-1]
            image_placeholder.image(image, caption=filename, use_container_width=True)
            #image_placeholder.image(image, caption=filename, use_column_width=True)
    
    # Add preload button
    if st.button("Preload All Images"):
        with st.spinner("Loading all images..."):
            for key in image_list:
                if key not in st.session_state.loaded_images:
                    image = get_image_from_s3(S3_BUCKET_NAME, key)
                    if image:
                        st.session_state.loaded_images[key] = image
            st.success(f"Loaded {len(st.session_state.loaded_images)} images")
    
        # Add option to clear cached images
    if st.button("Clear Cached Images"):
        st.session_state.loaded_images = {}
        st.success("Cleared image cache")
        st.rerun()
    
    # Show loading status
    st.caption(f"Loaded {len(st.session_state.loaded_images)} of {len(image_list)} images")
    
    # Auto-scroll logic
    if auto_scroll:
        # Display current image
        display_current_image()
        
        # Wait for specified duration
        time.sleep(scroll_speed)
        
        # Update index for next iteration
        st.session_state.image_index = (st.session_state.image_index + 1) % len(image_list)
        
        # Rerun the app to show next image
        st.rerun()
    else:
        # Just display current image without auto-scroll
        display_current_image()
    
    # Add image counter/progress bar
    progress = st.progress(0)
    progress.progress((st.session_state.image_index + 1) / len(image_list))
    

else:
    st.info("No images found in the S3 bucket.")