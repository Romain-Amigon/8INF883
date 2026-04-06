import streamlit as st
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# --- Configuration de la page ---
st.set_page_config(page_title="Générateur de Visages (CGAN / CVAE)", layout="wide")
st.title("Projet Vision Artificielle - 8INF887 : Génération de Visages")

# --- Définition des constantes GAN ---
Z_DIM = 100
C_DIM = 40
FEATURES_GEN = 64
CHANNELS = 3

# Liste officielle des 40 attributs de CelebA
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

# --- Définition de l'architecture du Générateur CGAN ---
class Generator(nn.Module):
    def __init__(self, z_dim, c_dim, features_g):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            # Input: (z_dim + c_dim) x 1 x 1 -> State: (features_g*8) x 4 x 4
            nn.ConvTranspose2d(z_dim + c_dim, features_g * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(features_g * 8),
            nn.ReLU(True),
            
            # State: (features_g*8) x 4 x 4 -> State: (features_g*4) x 8 x 8
            nn.ConvTranspose2d(features_g * 8, features_g * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(features_g * 4),
            nn.ReLU(True),
            
            # State: (features_g*4) x 8 x 8 -> State: (features_g*2) x 16 x 16
            nn.ConvTranspose2d(features_g * 4, features_g * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(features_g * 2),
            nn.ReLU(True),
            
            # State: (features_g*2) x 16 x 16 -> State: (features_g) x 32 x 32
            nn.ConvTranspose2d(features_g * 2, features_g, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(features_g),
            nn.ReLU(True),
            
            # State: (features_g) x 32 x 32 -> Output: 3 x 64 x 64
            nn.ConvTranspose2d(features_g, CHANNELS, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh() # Pour mapper les valeurs entre [-1, 1]
        )

    def forward(self, x, labels):
        # Ajout des dimensions pour concaténer avec le vecteur de bruit
        labels = labels.unsqueeze(2).unsqueeze(3) 
        x = torch.cat([x, labels], dim=1)
        return self.gen(x)

# --- Chargement du modèle GAN---
@st.cache_resource
def load_cgan_model():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    model = Generator(Z_DIM, C_DIM, FEATURES_GEN).to(device)
    try:
        # Pensez à ajuster le chemin d'accès ici si le fichier pth est dans un sous-dossier
        model.load_state_dict(torch.load("./GAN/saved_models/cgan_generator_15.pth", map_location=device))
        model.eval()
        return model, device
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {e}")
        return None, device

cgan_model, device = load_cgan_model()

# --- Création des onglets Streamlit ---
tab_cvae, tab_gan = st.tabs(["CVAE", "GAN"])

with tab_cvae:
    st.header("Modèle CVAE")

with tab_gan:
    st.header("Modèle CGAN")
    st.write("Sélectionnez les attributs faciaux désirés et générez une image de visage.")

    if cgan_model is not None:
        cols = st.columns(4)
        selected_attributes = {}
        
        for i, attr in enumerate(CELEBA_ATTRIBUTES):
            with cols[i % 4]:
                # On coche la checkbox si l'utilisateur souhaite cet attribut
                selected_attributes[attr] = st.checkbox(attr, value=False)
                
        st.divider()
        
        # Nombre d'images à générer
        num_images = st.number_input("Nombre d'images à générer", min_value=1, max_value=16, value=1, step=1)
        
        # Bouton de génération
        if st.button("Lancer la génération CGAN", type="primary"):
            with st.spinner("Génération en cours..."):
                # Préparer le vecteur des conditions (labels)
                # 1 si coché, 0 si non coché (à adapter selon si le modèle a été entraîné avec 0/1 ou -1/1)
                condition_list = [1.0 if selected_attributes[attr] else 0.0 for attr in CELEBA_ATTRIBUTES]
                
                # Dupliquer la condition pour le nombre d'images demandé
                labels_tensor = torch.tensor([condition_list] * num_images, dtype=torch.float32).to(device)
                
                # Générer des vecteurs de bruit aléatoires Z
                noise = torch.randn(num_images, Z_DIM, 1, 1, device=device)
                
                # Inférence avec le modèle CGAN
                with torch.no_grad():
                    generated_images = cgan_model(noise, labels_tensor)
                
                # Dénormalisation (de [-1, 1] vers [0, 1])
                generated_images = (generated_images + 1) / 2.0
                generated_images = generated_images.cpu().numpy()
                
                # Affichage dynamique sous forme de grille
                fig, axes = plt.subplots(1, num_images, figsize=(num_images * 3, 3))
                if num_images == 1:
                    axes = [axes]
                    
                for idx, ax in enumerate(axes):
                    # Transposer la forme PyTorch (C, H, W) -> (H, W, C)
                    img_to_show = np.transpose(generated_images[idx], (1, 2, 0))
                    ax.imshow(img_to_show)
                    ax.axis("off")
                    
                st.pyplot(fig)
    else:
        st.warning("Le modèle 'cgan_generator_15.pth' n'est pas chargé. Assurez-vous qu'il est bien présent.")