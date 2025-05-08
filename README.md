# Bedrock models prompt engineering challenge for image generation
This is a Bedrock prompt engineering challenge application. It lets users pick one of our image generation models (Nova Canvas, Amazon Titan, Stable Diffusion) and then generate images based on their prompts. The application also includes a dynamic rotating image carousel of all the images saved so far. 

# S3 bucket
Currently this program saves images to my S3 bucket.  You should change S3 bucket name to one of your buckets before running the program.  Program should have rights to read from and save images to the specified bucket

S3_BUCKET_NAME = "my_bucket_name"  

# Bedrock models
This program should have the ability to invoke the following models:

1. amazon.titan-image-generator-v2:0
2. amazon.nova-canvas-v1:0
3. stability.stable-diffusion-xl-v1
You are welcome to change the models based on your requirements.



# Pre-requisites
1. Python 3.11 or above
2. streamlit framework (pip install streamlit)
3. Other python libraries

# Run command:
streamlit run image_examples/genai_challenge.py
