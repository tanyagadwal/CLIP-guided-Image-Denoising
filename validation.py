# import torch
# import torchvision.transforms as transforms
# import torch.nn as nn
# from PIL import Image
# import numpy as np
# import matplotlib.pyplot as plt
# import os
# import glob
# from tqdm import tqdm
# from torch.utils.data import Dataset, DataLoader

# class UNetBlock(nn.Module):
#     # ... (no changes) ...
#     def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
#         super(UNetBlock, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )
#     def forward(self, x):
#         return self.conv(x)
    
# class UNetDenoisingAutoencoder(nn.Module):
#     # ... (no changes - uses direct mapping without residual weight) ...
#     def __init__(self):
#         super(UNetDenoisingAutoencoder, self).__init__()
#         # Encoder
#         self.enc1 = UNetBlock(3, 64)
#         self.enc2 = UNetBlock(64, 128)
#         self.enc3 = UNetBlock(128, 256)
#         self.enc4 = UNetBlock(256, 512)
#         # Downsampling
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
#         # Bottleneck
#         self.bottleneck = UNetBlock(512, 1024)
#         # Upsampling
#         self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
#         self.dec4 = UNetBlock(1024, 512) # Input: 512 (upconv) + 512 (skip)
#         self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
#         self.dec3 = UNetBlock(512, 256) # Input: 256 (upconv) + 256 (skip)
#         self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
#         self.dec2 = UNetBlock(256, 128) # Input: 128 (upconv) + 128 (skip)
#         self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
#         self.dec1 = UNetBlock(128, 64) # Input: 64 (upconv) + 64 (skip)
#         # Output layer
#         self.final_conv = nn.Conv2d(64, 3, kernel_size=1)


#     def forward(self, x):
#         # Encoding path
#         enc1 = self.enc1(x)
#         enc2 = self.enc2(self.pool(enc1))
#         enc3 = self.enc3(self.pool(enc2))
#         enc4 = self.enc4(self.pool(enc3))
#         # Bottleneck
#         bottleneck = self.bottleneck(self.pool(enc4))
#         # Decoding path with skip connections
#         dec4 = self.upconv4(bottleneck)
#         dec4 = torch.cat((dec4, enc4), dim=1); dec4 = self.dec4(dec4)
#         dec3 = self.upconv3(dec4)
#         dec3 = torch.cat((dec3, enc3), dim=1); dec3 = self.dec3(dec3)
#         dec2 = self.upconv2(dec3)
#         dec2 = torch.cat((dec2, enc2), dim=1); dec2 = self.dec2(dec2)
#         dec1 = self.upconv1(dec2)
#         dec1 = torch.cat((dec1, enc1), dim=1); dec1 = self.dec1(dec1)
#         # Final output
#         output = torch.sigmoid(self.final_conv(dec1))
#         return output

# class SSIM(nn.Module):
#     # ... (no changes) ...
#     def __init__(self, window_size=11, size_average=True, data_range=1.0):
#         super(SSIM, self).__init__()
#         self.window_size = window_size; self.size_average = size_average
#         self.data_range = data_range; self.channel = 1
#         self.window = self.create_window(window_size)
#     def gaussian(self, window_size, sigma):
#         gauss = torch.Tensor([np.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
#         return gauss/gauss.sum()
#     def create_window(self, window_size):
#         _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
#         _2D_window = _1D_window.mm(_1D_window.t()).unsqueeze(0).unsqueeze(0)
#         window = _2D_window.expand(1, 1, window_size, window_size).contiguous()
#         return window
#     def forward(self, img1, img2):
#         (_, channel, _, _) = img1.size()
#         if self.window.device != img1.device or self.window.size(0) != channel:
#             window = self.create_window(self.window_size).to(img1.device)
#             self.window = window.expand(channel, 1, self.window_size, self.window_size)
#             self.channel = channel
#         return self._ssim(img1, img2, self.window, self.window_size, channel, self.data_range, self.size_average)
#     def _ssim(self, img1, img2, window, window_size, channel, data_range=1.0, size_average=True):
#         C1 = (0.01 * data_range)**2; C2 = (0.03 * data_range)**2
#         mu1 = nn.functional.conv2d(img1, window, padding=window_size//2, groups=channel)
#         mu2 = nn.functional.conv2d(img2, window, padding=window_size//2, groups=channel)
#         mu1_sq = mu1.pow(2); mu2_sq = mu2.pow(2); mu1_mu2 = mu1 * mu2
#         sigma1_sq = nn.functional.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
#         sigma2_sq = nn.functional.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
#         sigma12 = nn.functional.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2
#         sigma1_sq = torch.clamp(sigma1_sq, min=0); sigma2_sq = torch.clamp(sigma2_sq, min=0)
#         ssim_map = ((2*mu1_mu2+C1)*(2*sigma12+C2))/((mu1_sq+mu2_sq+C1)*(sigma1_sq+sigma2_sq+C2))
#         return ssim_map.mean() if size_average else ssim_map.mean([1,2,3])
    
# def calculate_psnr(img1, img2, data_range=1.0):
#     # ... (no changes) ...
#     mse = torch.mean((img1 - img2) ** 2)
#     if mse == 0: return float('inf')
#     return 20 * torch.log10(data_range / torch.sqrt(mse))

# class EnhancedGenerator(nn.Module):
#     def __init__(self, autoencoder):
#         super(EnhancedGenerator, self).__init__()
#         self.autoencoder = autoencoder

#         # Slightly deeper refinement network
#         self.refinement = nn.Sequential(
#             nn.Conv2d(3, 32, kernel_size=3, padding=1), # Reduced initial channels
#             nn.ReLU(inplace=True),
#             nn.Conv2d(32, 32, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#              nn.Conv2d(32, 3, kernel_size=3, padding=1),
#             # Removed final Sigmoid, autoencoder likely outputs near [0,1]
#             # Clamping or a final Tanh might be needed depending on autoencoder output range
#         )

#         # Learnable blend factor - initialize closer to 1 to favor autoencoder initially
#         self.blend_factor = nn.Parameter(torch.tensor(0.9))

#     def forward(self, x):
#         # Autoencoder denoising
#         denoised = self.autoencoder(x)

#         # Refinement path
#         refined = self.refinement(denoised)

#         # Blend - Use Sigmoid on blend factor to keep it naturally between 0 and 1
#         # Adjust the range slightly to favor autoencoder more (e.g., 0.7 to 0.95)
#         blend = torch.sigmoid(self.blend_factor) * 0.25 + 0.7 # Scale to [0.7, 0.95]
#         output = blend * denoised + (1 - blend) * refined

#         # Clamp output to valid range if necessary (depends on autoencoder's final activation)
#         # output = torch.clamp(output, 0, 1)
#         # Or use Tanh if autoencoder uses Tanh:
#         # output = torch.tanh(output) * 0.5 + 0.5 # Rescale Tanh output from [-1, 1] to [0, 1]


#         return output

# def add_gaussian_noise(image_tensor, sigma=25, device='cpu'):
#     """Adds Gaussian noise to an image tensor."""
#     sigma_scaled = sigma / 255.0 # Assuming input tensor is [0, 1]
#     noise = torch.randn_like(image_tensor) * sigma_scaled
#     noisy_image = image_tensor + noise
#     noisy_image = torch.clamp(noisy_image, 0.0, 1.0)
#     return noisy_image

# def load_ensemble_model(model_path, device):
#     """Loads the trained EnhancedGenerator (ensemble) model."""
#     # Initialize the base autoencoder (needed by EnhancedGenerator)
#     autoencoder = UNetDenoisingAutoencoder()
#     # Initialize the generator which wraps the autoencoder
#     generator = EnhancedGenerator(autoencoder)

#     if not os.path.exists(model_path):
#         raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")

#     print(f"Loading ensemble model from {model_path}...")
#     # Load the checkpoint dictionary saved by the ensemble training script
#     checkpoint = torch.load(model_path, map_location=device)

#     # Check what keys are in the checkpoint
#     print(f"Available keys in checkpoint: {list(checkpoint.keys())}")
    
#     # Get the generator's state dict using the correct key
#     if 'generator_state_dict' in checkpoint:
#         generator_state_dict = checkpoint['generator_state_dict']
#         print("Using 'generator_state_dict' key")
#     elif 'generator' in checkpoint:
#         generator_state_dict = checkpoint['generator']
#         print("Using 'generator' key")
#     else:
#         raise KeyError(f"Checkpoint at '{model_path}' does not contain appropriate generator key. "
#                       f"Available keys: {list(checkpoint.keys())}")

#     # Print metrics if available
#     if 'psnr' in checkpoint:
#         print(f"Loaded model trained with PSNR: {checkpoint.get('psnr', 'N/A'):.2f}dB, SSIM: {checkpoint.get('ssim', 'N/A'):.4f}")

#     try:
#         generator.load_state_dict(generator_state_dict, strict=True)
#         print("Ensemble model state loaded successfully (strict=True).")
#     except RuntimeError as e:
#         print(f"Strict loading failed: {e}. Attempting non-strict loading...")
#         try:
#             generator.load_state_dict(generator_state_dict, strict=False)
#             print("Ensemble model state loaded successfully (strict=False).")
#         except Exception as final_e:
#             print(f"ERROR: Failed to load state_dict even with strict=False: {final_e}")
#             raise final_e

#     generator.to(device)
#     generator.eval()  # Set the model to evaluation mode
#     print("Ensemble model moved to device and set to evaluation mode.")
#     return generator
    
# class BenchmarkDataset(Dataset):
#     """Dataset for loading clean images from benchmark sets like Kodak24."""
#     def __init__(self, dataset_dir, transform=None):
#         self.transform = transform
#         self.image_paths = []
#         # Find common image extensions
#         extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif']
#         for ext in extensions:
#             self.image_paths.extend(glob.glob(os.path.join(dataset_dir, ext)))
#             # Also check common subdirectories like 'clean', 'original', 'HR'
#             for subdir in ['clean', 'original', 'HR', 'ground_truth']:
#                  self.image_paths.extend(glob.glob(os.path.join(dataset_dir, subdir, ext)))

#         self.image_paths = sorted(list(set(self.image_paths))) # Remove duplicates and sort

#         if not self.image_paths:
#             raise FileNotFoundError(f"No image files found in {dataset_dir} or common subdirectories.")
#         print(f"Found {len(self.image_paths)} clean images in {dataset_dir}")

#     def __len__(self):
#         return len(self.image_paths)

#     def __getitem__(self, idx):
#         img_path = self.image_paths[idx]
#         try:
#             # Handle different image modes (like grayscale in some benchmarks)
#             image = Image.open(img_path).convert('RGB')
#         except Exception as e:
#             print(f"Error loading image {img_path}: {e}")
#             # Return a dummy tensor if loading fails
#             return torch.zeros(3, 256, 256), os.path.basename(img_path)

#         if self.transform:
#             image = self.transform(image)
#         return image, os.path.basename(img_path) # Return image tensor and filename

# # --- HARDCODED PARAMETERS (Replace with your desired values) ---
# DATASET_DIR = "./Kodak24/clean"
# MODEL_PATH = "./ensemble_denoising_best.pth"
# NOISE_SIGMAS = [15, 25, 35, 40, 50]
# VISUALIZE_COUNT = 5
# OUTPUT_DIR = "./Kodak24_enhanced_clean"

# def main():
#     # Use hardcoded parameters instead of command line args
#     dataset_dir = DATASET_DIR
#     model_path = MODEL_PATH
#     noise_sigmas = NOISE_SIGMAS
#     visualize_count = VISUALIZE_COUNT
#     output_dir = OUTPUT_DIR
    
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")

#     # --- Define Transforms (MUST match the training transforms, esp. size) ---
#     transform = transforms.Compose([
#         transforms.Resize((256, 256)), # Ensure size matches training
#         transforms.ToTensor()          # Converts to [0, 1] tensor
#     ])

#     # --- Load Trained Ensemble Model ---
#     try:
#         generator = load_ensemble_model(model_path, device)
#     except Exception as e:
#         print(f"Error loading model: {e}")
#         return

#     # --- Prepare Benchmark Dataset ---
#     try:
#         benchmark_dataset = BenchmarkDataset(dataset_dir, transform=transform)
#         # Use batch_size=1 for validation to process images individually
#         benchmark_loader = DataLoader(benchmark_dataset, batch_size=1, shuffle=False, num_workers=2)
#     except FileNotFoundError as e:
#         print(e)
#         return
#     except Exception as e:
#         print(f"Error creating dataset/loader: {e}")
#         return

#     # --- Define Noise Levels for Evaluation ---
#     print(f"Evaluating on noise sigmas: {noise_sigmas}")

#     ssim_criterion = SSIM().to(device)
#     results = {} # Dictionary to store results per sigma

#     # --- Evaluation Loop ---
#     generator.eval() # Ensure model is in eval mode

#     for sigma in noise_sigmas:
#         print(f"\n--- Evaluating for Sigma = {sigma} ---")
#         total_psnr_noisy = 0.0
#         total_ssim_noisy = 0.0
#         total_psnr_denoised = 0.0
#         total_ssim_denoised = 0.0
#         image_count = 0

#         # Create output directory for visualizations if needed
#         vis_dir = None
#         if visualize_count > 0 and output_dir:
#             vis_dir = os.path.join(output_dir, f"sigma_{sigma}")
#             os.makedirs(vis_dir, exist_ok=True)

#         with torch.no_grad(): # Disable gradient calculations
#             for i, (clean_tensor, filenames) in enumerate(tqdm(benchmark_loader, desc=f"Sigma {sigma}")):
#                 # DataLoader returns a batch, but batch_size=1, so access [0]
#                 filename = filenames[0]
#                 clean_tensor = clean_tensor.to(device) # shape [1, C, H, W]

#                 # Add synthetic noise
#                 noisy_tensor = add_gaussian_noise(clean_tensor, sigma, device)

#                 # Denoise the image
#                 denoised_tensor = generator(noisy_tensor)
#                 denoised_tensor = torch.clamp(denoised_tensor, 0.0, 1.0)

#                 # --- Calculate Metrics ---
#                 psnr_noisy_val = calculate_psnr(noisy_tensor, clean_tensor)
#                 ssim_noisy_val = ssim_criterion(noisy_tensor, clean_tensor).item()
#                 psnr_denoised_val = calculate_psnr(denoised_tensor, clean_tensor)
#                 ssim_denoised_val = ssim_criterion(denoised_tensor, clean_tensor).item()

#                 total_psnr_noisy += psnr_noisy_val
#                 total_ssim_noisy += ssim_noisy_val
#                 total_psnr_denoised += psnr_denoised_val
#                 total_ssim_denoised += ssim_denoised_val
#                 image_count += 1

#                 # --- (Optional) Visualize and Save Results for first few images ---
#                 if i < visualize_count and vis_dir:
#                     noisy_img_np = noisy_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
#                     clean_img_np = clean_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
#                     denoised_img_np = denoised_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)

#                     noisy_img_np = np.clip(noisy_img_np, 0, 1)
#                     clean_img_np = np.clip(clean_img_np, 0, 1)
#                     denoised_img_np = np.clip(denoised_img_np, 0, 1)

#                     fig, axs = plt.subplots(1, 3, figsize=(18, 6))
#                     fig.suptitle(f"{filename} (Sigma={sigma})", fontsize=14)

#                     axs[0].imshow(noisy_img_np)
#                     axs[0].set_title(f'Noisy\nPSNR: {psnr_noisy_val:.2f}dB\nSSIM: {ssim_noisy_val:.4f}')
#                     axs[0].axis('off')

#                     axs[1].imshow(denoised_img_np)
#                     axs[1].set_title(f'Denoised\nPSNR: {psnr_denoised_val:.2f}dB\nSSIM: {ssim_denoised_val:.4f}')
#                     axs[1].axis('off')

#                     axs[2].imshow(clean_img_np)
#                     axs[2].set_title('Clean Ground Truth')
#                     axs[2].axis('off')

#                     plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#                     output_path = os.path.join(vis_dir, f"{os.path.splitext(filename)[0]}_result.png")
#                     try:
#                         plt.savefig(output_path)
#                     except Exception as e:
#                          print(f"Error saving visualization {output_path}: {e}")
#                     plt.close(fig) # Close the figure to save memory


#         # --- Calculate and Store Average Metrics for this Sigma ---
#         if image_count > 0:
#             avg_psnr_noisy = total_psnr_noisy / image_count
#             avg_ssim_noisy = total_ssim_noisy / image_count
#             avg_psnr_denoised = total_psnr_denoised / image_count
#             avg_ssim_denoised = total_ssim_denoised / image_count
#             results[sigma] = {
#                 'psnr_noisy': avg_psnr_noisy, 'ssim_noisy': avg_ssim_noisy,
#                 'psnr_denoised': avg_psnr_denoised, 'ssim_denoised': avg_ssim_denoised
#             }
#             print(f"\nAverage Results for Sigma = {sigma}:")
#             print(f"  Noisy:   PSNR={avg_psnr_noisy:.4f} dB, SSIM={avg_ssim_noisy:.4f}")
#             print(f"  Denoised: PSNR={avg_psnr_denoised:.4f} dB, SSIM={avg_ssim_denoised:.4f}")
#             print(f"  Improvement: PSNR={avg_psnr_denoised - avg_psnr_noisy:+.4f} dB, SSIM={avg_ssim_denoised - avg_ssim_noisy:+.4f}")
#         else:
#             print("No images processed.")

#     # --- Print Final Summary ---
#     print("\n--- Overall Benchmark Results ---")
#     for sigma, metrics in results.items():
#         print(f"Sigma={sigma}: PSNR={metrics['psnr_denoised']:.4f} dB, SSIM={metrics['ssim_denoised']:.4f}")
#     print("-" * 30)

#     # --- (Optional) Save results to a file ---
#     if output_dir:
#         results_path = os.path.join(output_dir, "benchmark_results.txt")
#         os.makedirs(output_dir, exist_ok=True)
#         try:
#             with open(results_path, 'w') as f:
#                 f.write(f"Benchmark Dataset: {dataset_dir}\n")
#                 f.write(f"Model: {model_path}\n")
#                 f.write("-" * 30 + "\n")
#                 for sigma, metrics in results.items():
#                      f.write(f"Sigma={sigma}:\n")
#                      f.write(f"  Noisy:   PSNR={metrics['psnr_noisy']:.4f} dB, SSIM={metrics['ssim_noisy']:.4f}\n")
#                      f.write(f"  Denoised: PSNR={metrics['psnr_denoised']:.4f} dB, SSIM={metrics['ssim_denoised']:.4f}\n")
#                      f.write(f"  Improvement: PSNR={metrics['psnr_denoised'] - metrics['psnr_noisy']:+.4f} dB, SSIM={metrics['ssim_denoised'] - metrics['ssim_noisy']:+.4f}\n\n")
#             print(f"Results saved to {results_path}")
#         except Exception as e:
#             print(f"Error saving results file: {e}")


# if __name__ == "__main__":
#     # Perform basic checks before running main
#     if not os.path.isdir(DATASET_DIR):
#         print(f"ERROR: Benchmark dataset directory not found: {DATASET_DIR}")
#     elif not os.path.isfile(MODEL_PATH):
#          print(f"ERROR: Model file not found: {MODEL_PATH}")
#     else:
#         main()

import torch
import torchvision.transforms as transforms
import torch.nn as nn
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

class UNetResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(UNetResBlock, self).__init__()
        # Optional: Projection shortcut if in_channels != out_channels
        self.shortcut = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False) # 1x1 conv

        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_channels), # BN before Conv can sometimes be better
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False), # No bias if using BN
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding, bias=False)
        )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv(x)
        return out + residual # Add residual connection

    
class UNetDenoisingAutoencoderRes(nn.Module):
    def __init__(self):
        super(UNetDenoisingAutoencoderRes, self).__init__()
        # Encoder using ResBlocks
        self.enc1 = UNetResBlock(3, 64)
        self.enc2 = UNetResBlock(64, 128)
        self.enc3 = UNetResBlock(128, 256)
        self.enc4 = UNetResBlock(256, 512)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck using ResBlock
        self.bottleneck = UNetResBlock(512, 1024)

        # Upsampling (keep ConvTranspose2d)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        # Decoder using ResBlocks - Adjust input channels for concatenation
        self.dec4 = UNetResBlock(1024, 512) # Input: 512 (upconv) + 512 (skip)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = UNetResBlock(512, 256) # Input: 256 (upconv) + 256 (skip)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = UNetResBlock(256, 128) # Input: 128 (upconv) + 128 (skip)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = UNetResBlock(128, 64) # Input: 64 (upconv) + 64 (skip)

        # Final layer
        self.final_conv = nn.Conv2d(64, 3, kernel_size=1)



    def forward(self, x):
        # Encoding path
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))

        # Decoding path with skip connections
        dec4 = self.upconv4(bottleneck)
        # Ensure spatial dimensions match before concatenation (might need padding adjustment or check sizes)
        # If Resize((256, 256)) is used, max pooling and conv transpose should align perfectly.
        dec4 = torch.cat((dec4, enc4), dim=1); dec4 = self.dec4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1); dec3 = self.dec3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1); dec2 = self.dec2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1); dec1 = self.dec1(dec1)

        # Final output - Keep Sigmoid if targets are [0, 1]
        output = torch.sigmoid(self.final_conv(dec1))
        return output

class SSIM(nn.Module):
    # ... (no changes) ...
    def __init__(self, window_size=11, size_average=True, data_range=1.0):
        super(SSIM, self).__init__()
        self.window_size = window_size; self.size_average = size_average
        self.data_range = data_range; self.channel = 1
        self.window = self.create_window(window_size)
    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([np.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()
    def create_window(self, window_size):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(1, 1, window_size, window_size).contiguous()
        return window
    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        if self.window.device != img1.device or self.window.size(0) != channel:
            window = self.create_window(self.window_size).to(img1.device)
            self.window = window.expand(channel, 1, self.window_size, self.window_size)
            self.channel = channel
        return self._ssim(img1, img2, self.window, self.window_size, channel, self.data_range, self.size_average)
    def _ssim(self, img1, img2, window, window_size, channel, data_range=1.0, size_average=True):
        C1 = (0.01 * data_range)**2; C2 = (0.03 * data_range)**2
        mu1 = nn.functional.conv2d(img1, window, padding=window_size//2, groups=channel)
        mu2 = nn.functional.conv2d(img2, window, padding=window_size//2, groups=channel)
        mu1_sq = mu1.pow(2); mu2_sq = mu2.pow(2); mu1_mu2 = mu1 * mu2
        sigma1_sq = nn.functional.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = nn.functional.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = nn.functional.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2
        sigma1_sq = torch.clamp(sigma1_sq, min=0); sigma2_sq = torch.clamp(sigma2_sq, min=0)
        ssim_map = ((2*mu1_mu2+C1)*(2*sigma12+C2))/((mu1_sq+mu2_sq+C1)*(sigma1_sq+sigma2_sq+C2))
        return ssim_map.mean() if size_average else ssim_map.mean([1,2,3])
    
def calculate_psnr(img1, img2, data_range=1.0):
    # ... (no changes) ...
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0: return float('inf')
    return 20 * torch.log10(data_range / torch.sqrt(mse))

class EnhancedGenerator(nn.Module):
    def __init__(self, autoencoder):
        super(EnhancedGenerator, self).__init__()
        self.autoencoder = autoencoder

        # Slightly deeper refinement network
        self.refinement = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), # Reduced initial channels
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
             nn.Conv2d(32, 3, kernel_size=3, padding=1),
            # Removed final Sigmoid, autoencoder likely outputs near [0,1]
            # Clamping or a final Tanh might be needed depending on autoencoder output range
        )

        # Learnable blend factor - initialize closer to 1 to favor autoencoder initially
        self.blend_factor = nn.Parameter(torch.tensor(0.9))

    def forward(self, x):
        # Autoencoder denoising
        denoised = self.autoencoder(x)

        # Refinement path
        refined = self.refinement(denoised)

        # Blend - Use Sigmoid on blend factor to keep it naturally between 0 and 1
        # Adjust the range slightly to favor autoencoder more (e.g., 0.7 to 0.95)
        blend = torch.sigmoid(self.blend_factor) * 0.25 + 0.7 # Scale to [0.7, 0.95]
        output = blend * denoised + (1 - blend) * refined

        # Clamp output to valid range if necessary (depends on autoencoder's final activation)
        # output = torch.clamp(output, 0, 1)
        # Or use Tanh if autoencoder uses Tanh:
        # output = torch.tanh(output) * 0.5 + 0.5 # Rescale Tanh output from [-1, 1] to [0, 1]


        return output

def add_gaussian_noise(image_tensor, sigma=25, device='cpu'):
    """Adds Gaussian noise to an image tensor."""
    sigma_scaled = sigma / 255.0 # Assuming input tensor is [0, 1]
    noise = torch.randn_like(image_tensor) * sigma_scaled
    noisy_image = image_tensor + noise
    noisy_image = torch.clamp(noisy_image, 0.0, 1.0)
    return noisy_image

def load_ensemble_model(model_path, device):
    """Loads the trained EnhancedGenerator (ensemble) model."""
    # Initialize the base autoencoder (needed by EnhancedGenerator)
    autoencoder = UNetDenoisingAutoencoderRes()
    # Initialize the generator which wraps the autoencoder
    generator = EnhancedGenerator(autoencoder)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")

    print(f"Loading ensemble model from {model_path}...")
    # Load the checkpoint dictionary saved by the ensemble training script
    checkpoint = torch.load(model_path, map_location=device)

    # Check what keys are in the checkpoint
    print(f"Available keys in checkpoint: {list(checkpoint.keys())}")
    
    # Get the generator's state dict using the correct key
    if 'generator_state_dict' in checkpoint:
        generator_state_dict = checkpoint['generator_state_dict']
        print("Using 'generator_state_dict' key")
    elif 'generator' in checkpoint:
        generator_state_dict = checkpoint['generator']
        print("Using 'generator' key")
    else:
        raise KeyError(f"Checkpoint at '{model_path}' does not contain appropriate generator key. "
                      f"Available keys: {list(checkpoint.keys())}")

    # Print metrics if available
    if 'psnr' in checkpoint:
        print(f"Loaded model trained with PSNR: {checkpoint.get('psnr', 'N/A'):.2f}dB, SSIM: {checkpoint.get('ssim', 'N/A'):.4f}")

    try:
        generator.load_state_dict(generator_state_dict, strict=True)
        print("Ensemble model state loaded successfully (strict=True).")
    except RuntimeError as e:
        print(f"Strict loading failed: {e}. Attempting non-strict loading...")
        try:
            generator.load_state_dict(generator_state_dict, strict=False)
            print("Ensemble model state loaded successfully (strict=False).")
        except Exception as final_e:
            print(f"ERROR: Failed to load state_dict even with strict=False: {final_e}")
            raise final_e

    generator.to(device)
    generator.eval()  # Set the model to evaluation mode
    print("Ensemble model moved to device and set to evaluation mode.")
    return generator
    
class BenchmarkDataset(Dataset):
    """Dataset for loading clean images from benchmark sets like Kodak24."""
    def __init__(self, dataset_dir, transform=None):
        self.transform = transform
        self.image_paths = []
        # Find common image extensions
        extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif']
        for ext in extensions:
            self.image_paths.extend(glob.glob(os.path.join(dataset_dir, ext)))
            # Also check common subdirectories like 'clean', 'original', 'HR'
            for subdir in ['clean', 'original', 'HR', 'ground_truth']:
                 self.image_paths.extend(glob.glob(os.path.join(dataset_dir, subdir, ext)))

        self.image_paths = sorted(list(set(self.image_paths))) # Remove duplicates and sort

        if not self.image_paths:
            raise FileNotFoundError(f"No image files found in {dataset_dir} or common subdirectories.")
        print(f"Found {len(self.image_paths)} clean images in {dataset_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            # Handle different image modes (like grayscale in some benchmarks)
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a dummy tensor if loading fails
            return torch.zeros(3, 256, 256), os.path.basename(img_path)

        if self.transform:
            image = self.transform(image)
        return image, os.path.basename(img_path) # Return image tensor and filename

# --- HARDCODED PARAMETERS (Replace with your desired values) ---
DATASET_DIR = "./CBSD68/original_png"
MODEL_PATH = "./ensemble_denoising_best.pth"
NOISE_SIGMAS = [15, 25, 35, 40, 50]
VISUALIZE_COUNT = 5
OUTPUT_DIR = "./CBSD68_clean"

def main():
    # Use hardcoded parameters instead of command line args
    dataset_dir = DATASET_DIR
    model_path = MODEL_PATH
    noise_sigmas = NOISE_SIGMAS
    visualize_count = VISUALIZE_COUNT
    output_dir = OUTPUT_DIR
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Define Transforms (MUST match the training transforms, esp. size) ---
    transform = transforms.Compose([
        transforms.Resize((256, 256)), # Ensure size matches training
        transforms.ToTensor()          # Converts to [0, 1] tensor
    ])

    # --- Load Trained Ensemble Model ---
    try:
        generator = load_ensemble_model(model_path, device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # --- Prepare Benchmark Dataset ---
    try:
        benchmark_dataset = BenchmarkDataset(dataset_dir, transform=transform)
        # Use batch_size=1 for validation to process images individually
        benchmark_loader = DataLoader(benchmark_dataset, batch_size=1, shuffle=False, num_workers=2)
    except FileNotFoundError as e:
        print(e)
        return
    except Exception as e:
        print(f"Error creating dataset/loader: {e}")
        return

    # --- Define Noise Levels for Evaluation ---
    print(f"Evaluating on noise sigmas: {noise_sigmas}")

    ssim_criterion = SSIM().to(device)
    results = {} # Dictionary to store results per sigma

    # --- Evaluation Loop ---
    generator.eval() # Ensure model is in eval mode

    for sigma in noise_sigmas:
        print(f"\n--- Evaluating for Sigma = {sigma} ---")
        total_psnr_noisy = 0.0
        total_ssim_noisy = 0.0
        total_psnr_denoised = 0.0
        total_ssim_denoised = 0.0
        image_count = 0

        # Create output directory for visualizations if needed
        vis_dir = None
        if visualize_count > 0 and output_dir:
            vis_dir = os.path.join(output_dir, f"sigma_{sigma}")
            os.makedirs(vis_dir, exist_ok=True)

        with torch.no_grad(): # Disable gradient calculations
            for i, (clean_tensor, filenames) in enumerate(tqdm(benchmark_loader, desc=f"Sigma {sigma}")):
                # DataLoader returns a batch, but batch_size=1, so access [0]
                filename = filenames[0]
                clean_tensor = clean_tensor.to(device) # shape [1, C, H, W]

                # Add synthetic noise
                noisy_tensor = add_gaussian_noise(clean_tensor, sigma, device)

                # Denoise the image
                denoised_tensor = generator(noisy_tensor)
                denoised_tensor = torch.clamp(denoised_tensor, 0.0, 1.0)

                # --- Calculate Metrics ---
                psnr_noisy_val = calculate_psnr(noisy_tensor, clean_tensor)
                ssim_noisy_val = ssim_criterion(noisy_tensor, clean_tensor).item()
                psnr_denoised_val = calculate_psnr(denoised_tensor, clean_tensor)
                ssim_denoised_val = ssim_criterion(denoised_tensor, clean_tensor).item()

                total_psnr_noisy += psnr_noisy_val
                total_ssim_noisy += ssim_noisy_val
                total_psnr_denoised += psnr_denoised_val
                total_ssim_denoised += ssim_denoised_val
                image_count += 1

                # --- (Optional) Visualize and Save Results for first few images ---
                if i < visualize_count and vis_dir:
                    noisy_img_np = noisy_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
                    clean_img_np = clean_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
                    denoised_img_np = denoised_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)

                    noisy_img_np = np.clip(noisy_img_np, 0, 1)
                    clean_img_np = np.clip(clean_img_np, 0, 1)
                    denoised_img_np = np.clip(denoised_img_np, 0, 1)

                    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
                    fig.suptitle(f"{filename} (Sigma={sigma})", fontsize=14)

                    axs[0].imshow(noisy_img_np)
                    axs[0].set_title(f'Noisy\nPSNR: {psnr_noisy_val:.2f}dB\nSSIM: {ssim_noisy_val:.4f}')
                    axs[0].axis('off')

                    axs[1].imshow(denoised_img_np)
                    axs[1].set_title(f'Denoised\nPSNR: {psnr_denoised_val:.2f}dB\nSSIM: {ssim_denoised_val:.4f}')
                    axs[1].axis('off')

                    axs[2].imshow(clean_img_np)
                    axs[2].set_title('Clean Ground Truth')
                    axs[2].axis('off')

                    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                    output_path = os.path.join(vis_dir, f"{os.path.splitext(filename)[0]}_result.png")
                    try:
                        plt.savefig(output_path)
                    except Exception as e:
                         print(f"Error saving visualization {output_path}: {e}")
                    plt.close(fig) # Close the figure to save memory


        # --- Calculate and Store Average Metrics for this Sigma ---
        if image_count > 0:
            avg_psnr_noisy = total_psnr_noisy / image_count
            avg_ssim_noisy = total_ssim_noisy / image_count
            avg_psnr_denoised = total_psnr_denoised / image_count
            avg_ssim_denoised = total_ssim_denoised / image_count
            results[sigma] = {
                'psnr_noisy': avg_psnr_noisy, 'ssim_noisy': avg_ssim_noisy,
                'psnr_denoised': avg_psnr_denoised, 'ssim_denoised': avg_ssim_denoised
            }
            print(f"\nAverage Results for Sigma = {sigma}:")
            print(f"  Noisy:   PSNR={avg_psnr_noisy:.4f} dB, SSIM={avg_ssim_noisy:.4f}")
            print(f"  Denoised: PSNR={avg_psnr_denoised:.4f} dB, SSIM={avg_ssim_denoised:.4f}")
            print(f"  Improvement: PSNR={avg_psnr_denoised - avg_psnr_noisy:+.4f} dB, SSIM={avg_ssim_denoised - avg_ssim_noisy:+.4f}")
        else:
            print("No images processed.")

    # --- Print Final Summary ---
    print("\n--- Overall Benchmark Results ---")
    for sigma, metrics in results.items():
        print(f"Sigma={sigma}: PSNR={metrics['psnr_denoised']:.4f} dB, SSIM={metrics['ssim_denoised']:.4f}")
    print("-" * 30)

    # --- (Optional) Save results to a file ---
    if output_dir:
        results_path = os.path.join(output_dir, "benchmark_results.txt")
        os.makedirs(output_dir, exist_ok=True)
        try:
            with open(results_path, 'w') as f:
                f.write(f"Benchmark Dataset: {dataset_dir}\n")
                f.write(f"Model: {model_path}\n")
                f.write("-" * 30 + "\n")
                for sigma, metrics in results.items():
                     f.write(f"Sigma={sigma}:\n")
                     f.write(f"  Noisy:   PSNR={metrics['psnr_noisy']:.4f} dB, SSIM={metrics['ssim_noisy']:.4f}\n")
                     f.write(f"  Denoised: PSNR={metrics['psnr_denoised']:.4f} dB, SSIM={metrics['ssim_denoised']:.4f}\n")
                     f.write(f"  Improvement: PSNR={metrics['psnr_denoised'] - metrics['psnr_noisy']:+.4f} dB, SSIM={metrics['ssim_denoised'] - metrics['ssim_noisy']:+.4f}\n\n")
            print(f"Results saved to {results_path}")
        except Exception as e:
            print(f"Error saving results file: {e}")


if __name__ == "__main__":
    # Perform basic checks before running main
    if not os.path.isdir(DATASET_DIR):
        print(f"ERROR: Benchmark dataset directory not found: {DATASET_DIR}")
    elif not os.path.isfile(MODEL_PATH):
         print(f"ERROR: Model file not found: {MODEL_PATH}")
    else:
        main()