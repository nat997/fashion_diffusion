import gradio as gr
import os
from pathlib import Path
import sys
import torch
from PIL import Image, ImageOps
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import uvicorn

# other imports and initializations

app = FastAPI()

class InputData(BaseModel):
    vton_img: str
    garm_img: str
    n_samples: int
    n_steps: int
    image_scale: float
    seed: int

@app.post("/process_hd/")
async def process_hd_api(input_data: InputData):
    vton_img = input_data.vton_img
    garm_img = input_data.garm_img
    n_samples = input_data.n_samples
    n_steps = input_data.n_steps
    image_scale = input_data.image_scale
    seed = input_data.seed
    images = process_hd(vton_img, garm_img, n_samples, n_steps, image_scale, seed)
    return {"images": [image.tolist() for image in images]}

@app.post("/process_dc/")
async def process_dc_api(input_data: InputData, category: str):
    vton_img = input_data.vton_img
    garm_img = input_data.garm_img
    n_samples = input_data.n_samples
    n_steps = input_data.n_steps
    image_scale = input_data.image_scale
    seed = input_data.seed
    images = process_dc(vton_img, garm_img, category, n_samples, n_steps, image_scale, seed)
    return {"images": [image.tolist() for image in images]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)