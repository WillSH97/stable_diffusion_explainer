import torch
from diffusers import AutoencoderKL, LMSDiscreteScheduler, UNet2DConditionModel

from PIL import Image
import requests
from io import BytesIO
import numpy as np

from matplotlib import pyplot as plt
from torchvision import transforms as tfms
from transformers import CLIPTextModel, CLIPTokenizer, logging
import os

# Supress some unnecessary warnings when loading the CLIPTextModel
logging.set_verbosity_error()


# Set device
def get_torch_device():
    torch_device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if "mps" == torch_device: 
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = "1"
    return torch_device

torch_device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
if "mps" == torch_device: 
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = "1"

def load_all_models(torch_device: str):
    # Load the tokenizer and text encoder to tokenize and encode the text.
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder = text_encoder.to(torch_device).half()
    
    # Load the autoencoder model which will be used to decode the latents into image space.
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
    vae = vae.to(torch_device).half()
    
    # UNet
    unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")
    unet=unet.to(torch_device).half()
    
    # Scheduler
    scheduler = LMSDiscreteScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")

    return

# Load the tokenizer and text encoder to tokenize and encode the text.
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = text_encoder.to(torch_device).half()

# Load the autoencoder model which will be used to decode the latents into image space.
vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
vae = vae.to(torch_device).half()

# UNet
unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")
unet=unet.to(torch_device).half()

# Scheduler
scheduler = LMSDiscreteScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")





# Text embedder funcs
def tokenize(stringlist):
    tokenized = tokenizer(stringlist, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    return tokenized

# def detokenizer(tokenized): ## - Do I need this?
#     return tokenizer.decode(tokenized['input_ids'][0], skip_special_tokens=True)

def sentence_embedder(stringlist):
    inputs=tokenize(stringlist)
    inputs=inputs.to(torch_device)
    with torch.no_grad():
        sentence_embeddings=text_encoder(**inputs)
    return sentence_embeddings

#Q do I need this here????
uncond_embeddings=sentence_embedder([''])[0]



## VAE ###

# Function for turning an image into a VAE input-- Do I need htis?
def img2VAETensor(PILImage): 
    PILImage=PILImage.convert('RGB')
    ImgTensor=tfms.ToTensor()(PILImage).unsqueeze(0) * 2.0 - 1.0
    return ImgTensor

# VAE encoder function -- Do I need this???
def VAE_encode(VAETensor):
    init_encode=vae.encode(VAETensor)
    encoded_sample=init_encode.latent_dist.sample()*0.18215
    return encoded_sample


def VAE_decode(VAE_encoded_sample):
    rescaled_sample=(1/0.18215)*VAE_encoded_sample

    with torch.no_grad():
        decodedTensor=vae.decode(rescaled_sample).sample
    decodedTensor=(decodedTensor/2+0.5).clamp(0,1)
    decodedTensor=decodedTensor.detach().cpu().permute(0, 2, 3, 1).numpy()
    decodedTensor=(decodedTensor * 255).round().astype("uint8")
    decodedTensorImg=Image.fromarray(decodedTensor[0])

    return decodedTensorImg





# Image Utils -- Q do I need these???
elonURL = requests.get('https://preview.redd.it/twsliey41yk21.jpg?width=640&crop=smart&auto=webp&s=19e34b31877d410d141c019faad30d967caffa16')
elonImg = Image.open(BytesIO(elonURL.content))
elonImg=elonImg.crop((0,50,640,690))
elonImg=elonImg.resize((512,512))

elonTensor=img2VAETensor(elonImg).to(torch_device).half()

elonEncodedSample=VAE_encode(elonTensor)


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

def get_image(URL):
    ImgURL = requests.get(URL)
    Img = Image.open(BytesIO(ImgURL.content))
    
    lrdiff=Img.size[0]-min(Img.size)
    tbdiff=Img.size[1]-min(Img.size)
    Img = Img.crop((0,0,Img.size[0]-lrdiff, Img.size[1]-tbdiff))
    Img = Img.resize((512,512))
    return Img

    
# def downres(Img):
#     Img=Img.resize((64,64))
#     Img=Img.resize((512,512))
#     return Img


# def VAE_demo(URL):
#     orig_img=get_image(URL)
#     downres_img=downres(orig_img)

#     tensor_Img=img2VAETensor(orig_img).to(torch_device).half()
#     encodedSample=VAE_encode(tensor_Img)
#     decoded_img=VAE_decode(encodedSample)
    return image_grid([orig_img, downres_img, decoded_img], 1, 3)

# # this is setting the number of theoretical denoising steps we're doing. This number is arbitrary - ignore it for now.
# scheduler.set_timesteps(100)

# # this is setting the step which we're denoising from - also ignore it for now.
# demo_timestep=80