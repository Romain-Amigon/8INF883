import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Générateur de Visages (CGAN / CVAE)", layout="wide")
st.title("Projet Vision Artificielle - 8INF887 : Génération de Visages")

Z_DIM = 100
C_DIM = 40
FEATURES_GEN = 64
CHANNELS = 3

CELEBA_ATTRIBUTES = [
    "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes", "Bald",
    "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair",
    "Blurry", "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin",
    "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones",
    "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard",
    "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline", "Rosy_Cheeks",
    "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair", "Wearing_Earrings",
    "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace", "Wearing_Necktie", "Young"
]

class Generator(nn.Module):
    def __init__(self, z_dim, c_dim, features_g):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            nn.ConvTranspose2d(z_dim + c_dim, features_g * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(features_g * 8),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(features_g * 8, features_g * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(features_g * 4),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(features_g * 4, features_g * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(features_g * 2),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(features_g * 2, features_g, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(features_g),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(features_g, CHANNELS, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, x, labels):
        labels = labels.unsqueeze(2).unsqueeze(3) 
        x = torch.cat([x, labels], dim=1)
        return self.gen(x)

class CVAE(nn.Module):
    def __init__(self, latent_dim=512, num_classes=40):
        super(CVAE, self).__init__()
        self.num_classes = num_classes
        
        self.enc_conv1 = nn.Conv2d(3 + num_classes, 64, kernel_size=4, stride=2, padding=1)
        self.enc_bn1 = nn.BatchNorm2d(64)
        self.enc_conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.enc_bn2 = nn.BatchNorm2d(128)
        self.enc_conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.enc_bn3 = nn.BatchNorm2d(256)
        self.enc_conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
        self.enc_bn4 = nn.BatchNorm2d(512)
        self.enc_conv5 = nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1)
        self.enc_bn5 = nn.BatchNorm2d(1024)
        
        self.fc_mu = nn.Linear(1024 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(1024 * 4 * 4, latent_dim)
        
        self.dec_fc = nn.Linear(latent_dim + num_classes, 1024 * 4 * 4)
        
        self.dec_conv0 = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1)
        self.dec_bn0 = nn.BatchNorm2d(512)
        self.dec_conv1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.dec_bn1 = nn.BatchNorm2d(256)
        self.dec_conv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.dec_bn2 = nn.BatchNorm2d(128)
        self.dec_conv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.dec_bn3 = nn.BatchNorm2d(64)
        self.dec_conv4 = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1)

    def encode(self, x, c):
        batch_size, _, h, w = x.size()
        c_expanded = c.view(batch_size, self.num_classes, 1, 1).expand(-1, -1, h, w)
        x = torch.cat([x, c_expanded], dim=1)
        
        x = F.leaky_relu(self.enc_bn1(self.enc_conv1(x)), 0.2)
        x = F.leaky_relu(self.enc_bn2(self.enc_conv2(x)), 0.2)
        x = F.leaky_relu(self.enc_bn3(self.enc_conv3(x)), 0.2)
        x = F.leaky_relu(self.enc_bn4(self.enc_conv4(x)), 0.2)
        x = F.leaky_relu(self.enc_bn5(self.enc_conv5(x)), 0.2)
        x = x.view(x.size(0), -1)
        
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c):
        z = torch.cat([z, c], dim=1)
        z = self.dec_fc(z)
        z = z.view(z.size(0), 1024, 4, 4)
        
        z = F.relu(self.dec_bn0(self.dec_conv0(z)))
        z = F.relu(self.dec_bn1(self.dec_conv1(z)))
        z = F.relu(self.dec_bn2(self.dec_conv2(z)))
        z = F.relu(self.dec_bn3(self.dec_conv3(z)))
        z = torch.sigmoid(self.dec_conv4(z))
        return z

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z, c)
        return x_recon, mu, logvar

@st.cache_resource
def load_cgan_model():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    model = Generator(Z_DIM, C_DIM, FEATURES_GEN).to(device)
    try:
        model.load_state_dict(torch.load("./GAN/saved_models/cgan_generator_15.pth", map_location=device))
        model.eval()
        return model, device
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {e}")
        return None, device

@st.cache_resource
def load_cvae_model():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    model = CVAE().to(device)
    try:
        model.load_state_dict(torch.load("./CVAE/models/CNN_features_128_model_20260403-080330.pth", map_location=device))
        model.eval()
        return model, device
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle CVAE : {e}")
        return None, device

cgan_model, device = load_cgan_model()
cvae_model, device_cvae = load_cvae_model()

tab_cvae, tab_gan = st.tabs(["CVAE", "GAN"])

with tab_cvae:
    st.header("Modèle CVAE")
    st.write("Sélectionnez les attributs faciaux désirés et générez une image de visage aléatoire via l'espace latent.")
    
    if cvae_model is not None:
        cols_cvae = st.columns(4)
        selected_attributes_cvae = {}
        
        for i, attr in enumerate(CELEBA_ATTRIBUTES):
            with cols_cvae[i % 4]:
                selected_attributes_cvae[attr] = st.checkbox(attr, value=False, key=f"cvae_{attr}")
                
        st.divider()
        
        num_images_cvae = st.number_input("Nombre d'images à générer (CVAE)", min_value=1, max_value=16, value=1, step=1)
        
        if st.button("Lancer la génération CVAE", type="primary"):
            with st.spinner("Génération en cours..."):
                condition_list_cvae = [1.0 if selected_attributes_cvae[attr] else 0.0 for attr in CELEBA_ATTRIBUTES]
                labels_tensor_cvae = torch.tensor([condition_list_cvae] * num_images_cvae, dtype=torch.float32).to(device_cvae)
                
                noise_cvae = torch.randn(num_images_cvae, 512, device=device_cvae)
                
                with torch.no_grad():
                    generated_images_cvae = cvae_model.decode(noise_cvae, labels_tensor_cvae)
                
                generated_images_cvae = generated_images_cvae.cpu().numpy()
                
                fig_cvae, axes_cvae = plt.subplots(1, num_images_cvae, figsize=(num_images_cvae * 3, 3))
                if num_images_cvae == 1:
                    axes_cvae = [axes_cvae]
                    
                for idx, ax in enumerate(axes_cvae):
                    img_to_show_cvae = np.transpose(generated_images_cvae[idx], (1, 2, 0))
                    ax.imshow(img_to_show_cvae)
                    ax.axis("off")
                    
                st.pyplot(fig_cvae)
    else:
        st.warning("Le modèle CVAE n'est pas chargé. Vérifiez le chemin d'accès.")

with tab_gan:
    st.header("Modèle CGAN")
    st.write("Sélectionnez les attributs faciaux désirés et générez une image de visage.")

    if cgan_model is not None:
        cols = st.columns(4)
        selected_attributes = {}
        
        for i, attr in enumerate(CELEBA_ATTRIBUTES):
            with cols[i % 4]:
                selected_attributes[attr] = st.checkbox(attr, value=False)
                
        st.divider()
        
        num_images = st.number_input("Nombre d'images à générer", min_value=1, max_value=16, value=1, step=1)
        
        if st.button("Lancer la génération CGAN", type="primary"):
            with st.spinner("Génération en cours..."):
                condition_list = [1.0 if selected_attributes[attr] else 0.0 for attr in CELEBA_ATTRIBUTES]
                
                labels_tensor = torch.tensor([condition_list] * num_images, dtype=torch.float32).to(device)
                
                noise = torch.randn(num_images, Z_DIM, 1, 1, device=device)
                
                with torch.no_grad():
                    generated_images = cgan_model(noise, labels_tensor)
                
                generated_images = (generated_images + 1) / 2.0
                generated_images = generated_images.cpu().numpy()
                
                fig, axes = plt.subplots(1, num_images, figsize=(num_images * 3, 3))
                if num_images == 1:
                    axes = [axes]
                    
                for idx, ax in enumerate(axes):
                    img_to_show = np.transpose(generated_images[idx], (1, 2, 0))
                    ax.imshow(img_to_show)
                    ax.axis("off")
                    
                st.pyplot(fig)
    else:
        st.warning("Le modèle 'cgan_generator_15.pth' n'est pas chargé. Assurez-vous qu'il est bien présent.")