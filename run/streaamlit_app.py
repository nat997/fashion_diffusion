import streamlit as st
import os
from pathlib import Path
import sys
import torch
from PIL import Image, ImageOps

from utils_ootd import get_mask_location

PROJECT_ROOT = Path(__file__).absolute().parents[1].absolute()
sys.path.insert(0, str(PROJECT_ROOT))

import time
from preprocess.openpose.run_openpose import OpenPose
from preprocess.humanparsing.run_parsing import Parsing
from ootd.inference_ootd_hd import OOTDiffusionHD
from ootd.inference_ootd_dc import OOTDiffusionDC

# Model initialization
openpose_model_hd = OpenPose(0)
parsing_model_hd = Parsing(0)
ootd_model_hd = OOTDiffusionHD(0)

openpose_model_dc = openpose_model_hd
parsing_model_dc = parsing_model_hd
ootd_model_dc = ootd_model_hd

category_dict = ['upperbody', 'lowerbody', 'dress']
category_dict_utils = ['upper_body', 'lower_body', 'dresses']

example_path = os.path.join(os.path.dirname(__file__), 'examples')
model_hd = os.path.join(example_path, 'model/model_1.png')
garment_hd = os.path.join(example_path, 'garment/03244_00.jpg')
model_dc = os.path.join(example_path, 'model/model_8.png')
garment_dc = os.path.join(example_path, 'garment/048554_1.jpg')

def process_hd(vton_img, garm_img, n_samples, n_steps, image_scale, seed):
    model_type = 'hd'
    category = 0 # 0:upperbody; 1:lowerbody; 2:dress

    with torch.no_grad():
        garm_img = Image.open(garm_img).resize((768, 1024))
        vton_img = Image.open(vton_img).resize((768, 1024))
        keypoints = openpose_model_hd(vton_img.resize((384, 512)))
        model_parse, _ = parsing_model_hd(vton_img.resize((384, 512)))

        mask, mask_gray = get_mask_location(model_type, category_dict_utils[category], model_parse, keypoints)
        mask = mask.resize((768, 1024), Image.NEAREST)
        mask_gray = mask_gray.resize((768, 1024), Image.NEAREST)
        
        masked_vton_img = Image.composite(mask_gray, vton_img, mask)

        images = ootd_model_hd(
            model_type=model_type,
            category=category_dict[category],
            image_garm=garm_img,
            image_vton=masked_vton_img,
            mask=mask,
            image_ori=vton_img,
            num_samples=n_samples,
            num_steps=n_steps,
            image_scale=image_scale,
            seed=seed,
        )

    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)
    for i, image in enumerate(images):
        output_path = os.path.join(output_dir, f'image_{i}.png')
        image.save(output_path)

    return images

def process_dc(vton_img, garm_img, category, n_samples, n_steps, image_scale, seed):
    model_type = 'dc'
    if category == 'Upper-body':
        category = 0
    elif category == 'Lower-body':
        category = 1
    else:
        category =2

    with torch.no_grad():
        garm_img = Image.open(garm_img).resize((768, 1024))
        vton_img = Image.open(vton_img).resize((768, 1024))
        keypoints = openpose_model_dc(vton_img.resize((384, 512)))
        model_parse, _ = parsing_model_dc(vton_img.resize((384, 512)))

        mask, mask_gray = get_mask_location(model_type, category_dict_utils[category], model_parse, keypoints)
        mask = mask.resize((768, 1024), Image.NEAREST)
        mask_gray = mask_gray.resize((768, 1024), Image.NEAREST)
        
        masked_vton_img = Image.composite(mask_gray, vton_img, mask)

        images = ootd_model_dc(
            model_type=model_type,
            category=category_dict[category],
            image_garm=garm_img,
            image_vton=masked_vton_img,
            mask=mask,
            image_ori=vton_img,
            num_samples=n_samples,
            num_steps=n_steps,
            image_scale=image_scale,
            seed=seed,
        )

    return images
# Streamlit app
st.title("OOTDiffusion Demo")

# Half-body
st.subheader("Half-body")
st.markdown("***Support upper-body garments***")

col1, col2, col3 = st.columns(3)
with col1:
    vton_img = st.file_uploader("Model", type=["png", "jpg"], help="Upload a model image")
    if vton_img:
        model_hd = Image.open(vton_img)
        st.image(model_hd, width=384)
with col2:
    garm_img = st.file_uploader("Garment", type=["png", "jpg"], help="Upload a garment image")
    if garm_img:
        garment_hd = Image.open(garm_img)
        st.image(garment_hd, width=384)
with col3:
    if st.button("Run"):
        n_samples = st.slider("Images", min_value=1, max_value=4, value=1, step=1, help="Number of output images")
        n_steps = st.slider("Steps", min_value=20, max_value=40, value=20, step=1, help="Number of diffusion steps")
        image_scale = st.slider("Guidance scale", min_value=1.0, max_value=5.0, value=2.0, step=0.1, help="Guidance scale for the diffusion process")
        seed = st.slider("Seed", min_value=-1, max_value=2147483647, value=-1, step=1, help="Random seed for the diffusion process")

        if vton_img and garm_img:
            with st.spinner("Processing..."):
                images = process_hd(vton_img, garm_img, n_samples, n_steps, image_scale, seed)
            for i, image in enumerate(images):
                st.image(image, width=384, caption=f"Output {i+1}")

# Full-body
st.subheader("Full-body")
st.markdown("***Support upper-body/lower-body/dresses; garment category must be paired!!!***")

col1, col2, col3 = st.columns(3)
with col1:
    vton_img_dc = st.file_uploader("Model", type=["png", "jpg"], help="Upload a model image")
    if vton_img_dc:
        model_dc = Image.open(vton_img_dc)
        st.image(model_dc, width=384)
with col2:
    garm_img_dc = st.file_uploader("Garment", type=["png", "jpg"], help="Upload a garment image")
    category_dc = st.selectbox("Garment category", ["Upper-body", "Lower-body", "Dress"], index=0, help="Select garment category")
    if garm_img_dc:
        garment_dc = Image.open(garm_img_dc)
        st.image(garment_dc, width=384)