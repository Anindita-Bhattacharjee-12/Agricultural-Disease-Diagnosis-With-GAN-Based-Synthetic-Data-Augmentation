import torch
import matplotlib.pyplot as plt
import os
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from stargan_tomatoleaves import Model, CelebA, ImgForPlot, device, nf, nz, nd, sdim  # Import from main file

# ==========================
# Configuration
# ==========================
dataset_dir = 'D:/ArkoStudyMaterials/CVPR/Model/CVPR_Dataset'  # Update path
model_path = './stargan_tomato_v1'  # Path to saved model
output_dir = './generated_results'
os.makedirs(output_dir, exist_ok=True)

batch_size = 1  # Generate images one by one for precise control
num_images_per_class = 5
classes = ["Healthy", "Early Blight", "Late Blight", "Leaf Mold", "Septoria"]

# ==========================
# Load dataset
# ==========================
test_dataset = CelebA('test', dataset_dir)
test_ref_dataset = CelebA('test_ref', dataset_dir)

test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
ref_dataloader = DataLoader(test_ref_dataset, batch_size=batch_size, shuffle=True)

# ==========================
# Load model
# ==========================
model = Model(nf, nz, nd, sdim, lr=1e-4, lr_f=1e-6, betas=(0.0, 0.99),
              weight_decay=1e-4, device=device)
model.load(model_path)
model.eval()

print("Model loaded successfully!")

# ==========================
# Generate images
# ==========================
generated_images = {cls: [] for cls in classes}

with torch.no_grad():
    for target_class, class_name in enumerate(classes):
        print(f"\nGenerating images for category: {class_name}")
        images_generated = 0

        for (x, y), (x_ref, y_ref) in zip(test_dataloader, ref_dataloader):
            x, x_ref, y_ref = x.to(device), x_ref.to(device), y_ref.to(device).squeeze(1)

            # Encode style and generate image
            s_ = model.E(x_ref, y_ref)
            x_generated = model.G(x, s_)

            generated_images[class_name].append(x_generated.cpu())

            images_generated += 1
            if images_generated >= num_images_per_class:
                break  # Stop after generating 5 images for this class

# ==========================
# Visualize & Save
# ==========================
for class_name in classes:
    imgs_tensor = torch.cat(generated_images[class_name], dim=0)
    grid = make_grid(imgs_tensor, nrow=num_images_per_class, normalize=True, value_range=(-1, 1))

    plt.figure(figsize=(15, 5))
    plt.imshow(ImgForPlot(grid))
    plt.axis("off")
    plt.title(f"Generated Images - {class_name}")
    plt.savefig(os.path.join(output_dir, f"{class_name}_generated.png"))
    plt.close()

print(f"\nGenerated images saved to: {output_dir}")
