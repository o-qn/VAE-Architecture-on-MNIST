# milestone 4 - vae evaluation & analysis on mnist

import zipfile
import struct
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from sklearn.manifold import TSNE


zip_file = "archive.zip"

batch = 128
epochs = 5
z_size = 20
lr = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using:", device)


# read images from the mnist zip
def get_images(file_name):
    z = zipfile.ZipFile(zip_file, "r")
    f = z.open(file_name)

    # first 16 bytes are just info about the file
    magic, number, rows, cols = struct.unpack(">IIII", f.read(16))

    data = f.read()
    imgs = torch.frombuffer(data, dtype=torch.uint8)
    imgs = imgs.float()
    imgs = imgs.view(number, 1, rows, cols)
    imgs = imgs / 255

    f.close()
    z.close()
    return imgs


# read labels from the mnist zip
def get_labels(file_name):
    z = zipfile.ZipFile(zip_file, "r")
    f = z.open(file_name)

    magic, number = struct.unpack(">II", f.read(8))

    data = f.read()
    labels = torch.frombuffer(data, dtype=torch.uint8)
    labels = labels.long()

    f.close()
    z.close()
    return labels


x_train = get_images("train-images.idx3-ubyte")
y_train = get_labels("train-labels.idx1-ubyte")

x_test = get_images("t10k-images.idx3-ubyte")
y_test = get_labels("t10k-labels.idx1-ubyte")

print("train images shape:", x_train.shape)
print("train labels shape:", y_train.shape)
print("test images shape:", x_test.shape)
print("test labels shape:", y_test.shape)

train_data = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_data, batch_size=batch, shuffle=True)

test_data = TensorDataset(x_test, y_test)
test_loader = DataLoader(test_data, batch_size=batch, shuffle=False)


class MyVAE(nn.Module):
    def __init__(self):
        super().__init__()

        # encoder part
        self.layer1 = nn.Linear(784, 400)
        self.mean_layer = nn.Linear(400, z_size)
        self.log_layer = nn.Linear(400, z_size)

        # decoder part
        self.layer2 = nn.Linear(z_size, 400)
        self.layer3 = nn.Linear(400, 784)

    def encoder(self, x):
        x = torch.relu(self.layer1(x))
        mean = self.mean_layer(x)
        log_var = self.log_layer(x)
        return mean, log_var

    def sample(self, mean, log_var):
        # reparameterization trick
        std = torch.exp(0.5 * log_var)
        random_noise = torch.randn_like(std)
        z = mean + random_noise * std
        return z

    def decoder(self, z):
        z = torch.relu(self.layer2(z))
        z = torch.sigmoid(self.layer3(z))
        return z

    def forward(self, img):
        img = img.view(-1, 784)

        mean, log_var = self.encoder(img)
        z = self.sample(mean, log_var)
        result = self.decoder(z)

        return result, mean, log_var


def loss_function(new_img, old_img, mean, log_var):
    old_img = old_img.view(-1, 784)

    # reconstruction error
    loss1 = nn.functional.binary_cross_entropy(new_img, old_img, reduction="sum")

    # kl divergence
    loss2 = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    return loss1 + loss2


model = MyVAE()
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=lr)

train_losses = []

for e in range(epochs):
    total = 0

    for img, label in train_loader:
        img = img.to(device)

        optimizer.zero_grad()

        output, mean, log_var = model(img)

        loss = loss_function(output, img, mean, log_var)

        loss.backward()
        optimizer.step()

        total += loss.item()

    epoch_loss = total / len(train_loader.dataset)
    train_losses.append(epoch_loss)
    print("epoch", e + 1, "loss =", epoch_loss)


# -------------------------------------------------------
# milestone 4: evaluation & analysis
# -------------------------------------------------------

model.eval()

# --- 1. plot training loss curve ---
plt.figure(figsize=(7, 4))
plt.plot(range(1, epochs + 1), train_losses, marker="o")
plt.title("VAE Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.tight_layout()
plt.savefig("loss_curve.png")
plt.close()
print("saved: loss_curve.png")


# --- 2. reconstruction quality: original vs reconstructed ---
# grab one batch from the test set
test_imgs, test_labels = next(iter(test_loader))
test_imgs = test_imgs.to(device)

with torch.no_grad():
    recon, _, _ = model(test_imgs)

recon = recon.view(-1, 1, 28, 28)

# show 8 pairs side by side
n = 8
fig, axes = plt.subplots(2, n, figsize=(n * 1.5, 3))
for i in range(n):
    axes[0, i].imshow(test_imgs[i].cpu().squeeze(), cmap="gray")
    axes[0, i].axis("off")
    if i == 0:
        axes[0, i].set_title("original", fontsize=8)

    axes[1, i].imshow(recon[i].cpu().squeeze(), cmap="gray")
    axes[1, i].axis("off")
    if i == 0:
        axes[1, i].set_title("reconstructed", fontsize=8)

plt.tight_layout()
plt.savefig("reconstruction_comparison.png")
plt.close()
print("saved: reconstruction_comparison.png")


# --- 3. latent space visualization using t-sne ---
# collect all test latent means and their labels
all_means = []
all_labels = []

with torch.no_grad():
    for img, label in test_loader:
        img = img.to(device)
        mean, _ = model.encoder(img.view(-1, 784))
        all_means.append(mean.cpu())
        all_labels.append(label)

all_means = torch.cat(all_means, dim=0).numpy()
all_labels = torch.cat(all_labels, dim=0).numpy()

# t-sne down to 2d
print("running t-sne on latent space (this may take a moment)...")
tsne = TSNE(n_components=2, random_state=42)
z_2d = tsne.fit_transform(all_means)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(z_2d[:, 0], z_2d[:, 1], c=all_labels, cmap="tab10", s=2, alpha=0.6)
plt.colorbar(scatter, label="digit class")
plt.title("Latent Space Visualization (t-SNE)")
plt.xlabel("dim 1")
plt.ylabel("dim 2")
plt.tight_layout()
plt.savefig("latent_space_tsne.png")
plt.close()
print("saved: latent_space_tsne.png")


# --- 4. latent space 2d grid traversal (only works well when z_size == 2) ---
# since z_size = 20, we interpolate along the first two latent dims
# and keep the rest at 0 to show how those dims affect output
n_steps = 12
z_min, z_max = -3, 3

fig = plt.figure(figsize=(n_steps, n_steps))
gs = gridspec.GridSpec(n_steps, n_steps, wspace=0.05, hspace=0.05)

with torch.no_grad():
    for row_i, dim1_val in enumerate(np.linspace(z_max, z_min, n_steps)):
        for col_i, dim0_val in enumerate(np.linspace(z_min, z_max, n_steps)):
            z = torch.zeros(1, z_size).to(device)
            z[0, 0] = dim0_val
            z[0, 1] = dim1_val
            img_out = model.decoder(z).cpu().view(28, 28)

            ax = fig.add_subplot(gs[row_i, col_i])
            ax.imshow(img_out, cmap="gray")
            ax.axis("off")

plt.suptitle("Latent Space Grid (dim 0 x dim 1)", y=1.01, fontsize=12)
plt.savefig("latent_grid_traversal.png", bbox_inches="tight")
plt.close()
print("saved: latent_grid_traversal.png")


# --- 5. per-class reconstruction error ---
class_errors = {i: [] for i in range(10)}

with torch.no_grad():
    for img, label in test_loader:
        img = img.to(device)
        recon_batch, mean, log_var = model(img)

        for i in range(len(label)):
            orig = img[i].view(784)
            rec = recon_batch[i]
            err = nn.functional.binary_cross_entropy(rec, orig, reduction="sum").item()
            class_errors[label[i].item()].append(err)

avg_errors = [np.mean(class_errors[c]) for c in range(10)]

plt.figure(figsize=(8, 4))
plt.bar(range(10), avg_errors)
plt.xticks(range(10))
plt.xlabel("Digit Class")
plt.ylabel("Avg Reconstruction Error (BCE)")
plt.title("Per-Class Reconstruction Error on Test Set")
plt.tight_layout()
plt.savefig("per_class_error.png")
plt.close()
print("saved: per_class_error.png")


# --- 6. random samples from prior ---
with torch.no_grad():
    random_z = torch.randn(16, z_size).to(device)
    samples = model.decoder(random_z)
    samples = samples.view(-1, 1, 28, 28)
    save_image(samples, "generated_digits.png", nrow=4)

print("saved: generated_digits.png")
print("finished - check all saved images for the evaluation report")