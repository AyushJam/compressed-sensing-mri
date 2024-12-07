import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift, ifftshift
import scipy.io
from skimage.data import shepp_logan_phantom
from skimage.transform import resize
from skimage.metrics import structural_similarity as ssim
import pywt
import imageio
import os
import shutil

# ============================================================
# Utility Functions
# ============================================================

def ensure_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

def fft2c(img):
    return fftshift(fft2(img))

def ifft2c(ksp):
    return ifft2(ifftshift(ksp))

def soft_threshold(x, threshold):
    return np.sign(x)*np.maximum(np.abs(x)-threshold,0)

def wavelet_threshold(img, wavelet='db4', level=4, lam=0.01):
    coeffs = pywt.wavedec2(img, wavelet=wavelet, level=level)
    thresh_coeffs = []
    for c in coeffs:
        if isinstance(c, tuple):
            thresh_coeffs.append(tuple(soft_threshold(ci, lam) for ci in c))
        else:
            thresh_coeffs.append(soft_threshold(c, lam))
    return pywt.waverec2(thresh_coeffs, wavelet=wavelet)

def total_variation_gradient(img):
    dx = np.roll(img, -1, axis=1) - img
    dy = np.roll(img, -1, axis=0) - img
    dx_t = dx - np.roll(dx, 1, axis=1)
    dy_t = dy - np.roll(dy, 1, axis=0)
    grad = dx_t + dy_t
    return grad

def mse_metric(x, ref):
    return np.mean((x - ref)**2)

def psnr_metric(x, ref):
    mse_val = mse_metric(x, ref)
    if mse_val == 0:
        return 100
    max_val = np.max(ref)
    return 20 * np.log10(max_val / np.sqrt(mse_val))

# ============================================================
# Sampling Masks
# ============================================================

def mask_uniform(size, factor=4):
    mask = np.zeros(size)
    mask[::factor, :] = 1
    return mask

def mask_variable_density(size, center_fraction=0.08, prob=0.3):
    mask = np.zeros(size)
    Ny, Nx = size
    center_size = int(Ny * center_fraction)
    center_start = Ny//2 - center_size//2
    center_end = center_start + center_size
    mask[center_start:center_end, :] = 1
    random_mask = np.random.rand(Ny, Nx) < prob
    mask = np.logical_or(mask, random_mask)
    return mask.astype(float)

def mask_radial(size, num_spokes=30):
    Ny, Nx = size
    mask = np.zeros(size)
    center = (Ny//2, Nx//2)
    angles = np.linspace(0, np.pi, num_spokes, endpoint=False)
    radius = max(Ny, Nx)//2
    for theta in angles:
        x_line = np.cos(theta)*np.arange(-radius, radius) + center[1]
        y_line = np.sin(theta)*np.arange(-radius, radius) + center[0]
        x_line = x_line.astype(int)
        y_line = y_line.astype(int)
        valid = (x_line >=0)&(x_line<Nx)&(y_line>=0)&(y_line<Ny)
        mask[y_line[valid], x_line[valid]] = 1
    return mask

def mask_random_incoherent(size, sampling_rate=0.3):
    # Pure random sampling of k-space points with given probability
    Ny, Nx = size
    mask = (np.random.rand(Ny, Nx) < sampling_rate).astype(float)
    return mask

# ============================================================
# Reconstruction Methods (Iterative with Metric Tracking)
# ============================================================

def reconstruct_zero_filled(undersampled_k, mask):
    return np.abs(ifft2c(undersampled_k))

def iterative_recon_wavelet(undersampled_k, mask, outdir, original, max_iter=50, lam=0.01, wavelet='db4', level=4):
    rec = reconstruct_zero_filled(undersampled_k, mask)
    mses = []
    for i in range(max_iter):
        # Data consistency
        ksp_rec = fft2c(rec)
        ksp_rec = undersampled_k + (1 - mask)*ksp_rec
        rec = np.abs(ifft2c(ksp_rec))
        # Wavelet thresholding
        rec = wavelet_threshold(rec, wavelet=wavelet, level=level, lam=lam)
        mses.append(mse_metric(rec, original))

        # Save iteration image
        plt.imshow(rec, cmap='gray')
        plt.title(f'Wavelet CS Iteration {i+1}')
        plt.axis('off')
        plt.savefig(f'{outdir}/iter_{i+1}.png', bbox_inches='tight')
        plt.close()
    return rec, mses

def iterative_recon_tv(undersampled_k, mask, outdir, original, max_iter=50, lam=0.001, lr=0.1):
    rec = reconstruct_zero_filled(undersampled_k, mask)
    mses = []
    for i in range(max_iter):
        # Data consistency
        ksp_rec = fft2c(rec)
        ksp_rec = undersampled_k + (1 - mask)*ksp_rec
        rec = np.abs(ifft2c(ksp_rec))

        # TV gradient step
        grad_tv = total_variation_gradient(rec)
        rec = rec - lr * lam * grad_tv
        mses.append(mse_metric(rec, original))

        # Save iteration image
        plt.imshow(rec, cmap='gray')
        plt.title(f'TV CS Iteration {i+1}')
        plt.axis('off')
        plt.savefig(f'{outdir}/iter_{i+1}.png', bbox_inches='tight')
        plt.close()
    return rec, mses

# ============================================================
# Visualization of Sampling Patterns
# ============================================================

def visualize_sampling_pattern(full_k_space, mask, outdir):
    sampled_k = full_k_space * mask

    plt.imshow(np.log(1+np.abs(full_k_space)), cmap='gray')
    plt.title('Full k-space (log magnitude)')
    plt.axis('off')
    plt.savefig(f'{outdir}/pattern_full_kspace.png', bbox_inches='tight')
    plt.close()

    plt.imshow(mask, cmap='gray')
    plt.title('Sampling Mask')
    plt.axis('off')
    plt.savefig(f'{outdir}/pattern_mask.png', bbox_inches='tight')
    plt.close()

    plt.imshow(np.log(1+np.abs(sampled_k)), cmap='gray')
    plt.title('Sampled k-space (log magnitude)')
    plt.axis('off')
    plt.savefig(f'{outdir}/pattern_sampled_kspace.png', bbox_inches='tight')
    plt.close()

    # Create GIF
    images = []
    for fname in ['pattern_full_kspace.png', 'pattern_mask.png', 'pattern_sampled_kspace.png']:
        images.append(imageio.imread(f'{outdir}/{fname}'))
    imageio.mimsave(f'{outdir}/sampling_pattern.gif', images, fps=1)

# ============================================================
# Main Demonstration
# ============================================================

mat_file_path = r"C:\Users\admin\Downloads\brain.mat"

# Load the .mat file
data = scipy.io.loadmat(mat_file_path)

# Load original image
image_size = 256
#original_image = shepp_logan_phantom()
original_image = np.abs(data['im'])
original_image = resize(original_image, (image_size, image_size), mode='reflect', anti_aliasing=True)

# Full k-space (clean)
full_k_space_clean = fft2c(original_image)

# Add noise level
# For demonstration, we add AWGN in k-space.
noise_level = 0.000  # adjust as needed
# Note: With noise added, reconstructions might be degraded.

sampling_methods = {
    'uniform': lambda sz: mask_uniform(sz, factor=4),
    'variable_density': lambda sz: mask_variable_density(sz, center_fraction=0.08, prob=0.3),
    'radial': lambda sz: mask_radial(sz, num_spokes=30),
    'random_incoherent': lambda sz: mask_random_incoherent(sz, sampling_rate=0.3)
}

recon_methods = {
    'zero_filled': None,
    'cs_wavelet': iterative_recon_wavelet,
    'cs_tv': iterative_recon_tv
}

result_dir = 'results'
ensure_dir(result_dir)

# Save original image
plt.imshow(original_image, cmap='gray')
plt.title('Original Image')
plt.axis('off')
plt.savefig(f'{result_dir}/original_image.png', bbox_inches='tight')
plt.close()

num_iterations = 50

# We'll store final results in a structured manner for a table
# Structure: results[(sampling_method, recon_method)] = { 'fraction_data': , 'MSE': , 'PSNR': , 'SSIM': }
final_results = {}

from skimage.metrics import structural_similarity as ssim

for sm_name, sm_func in sampling_methods.items():
    # Directory for this sampling method
    sm_dir = f'{result_dir}/{sm_name}'
    ensure_dir(sm_dir)

    # Generate mask
    mask = sm_func(original_image.shape)
    fraction_data = np.mean(mask)  # fraction of sampled points

    # Apply mask and add noise to k-space
    undersampled_k = full_k_space_clean * mask
    # Add Gaussian noise
    noise = (np.random.randn(*undersampled_k.shape) + 1j*np.random.randn(*undersampled_k.shape)) * noise_level * np.max(np.abs(full_k_space_clean))
    undersampled_k_noisy = undersampled_k + noise

    # Visualize sampling pattern
    visualize_sampling_pattern(full_k_space_clean, mask, sm_dir)

    for rm_name, rm_func in recon_methods.items():
        outdir = f'{sm_dir}/{rm_name}'
        ensure_dir(outdir)

        # Zero-filled baseline
        zero_filled_recon = reconstruct_zero_filled(undersampled_k_noisy, mask)
        plt.imshow(zero_filled_recon, cmap='gray')
        plt.title(f'{sm_name} - {rm_name} Zero-filled Recon')
        plt.axis('off')
        plt.savefig(f'{outdir}/zero_filled_recon.png', bbox_inches='tight')
        plt.close()

        if rm_name == 'zero_filled':
            final_rec = zero_filled_recon
            mses = []
        elif rm_name == 'cs_wavelet':
            final_rec, mses = rm_func(undersampled_k_noisy, mask, outdir, original_image, max_iter=num_iterations, lam=0.01)
        elif rm_name == 'cs_tv':
            final_rec, mses = rm_func(undersampled_k_noisy, mask, outdir, original_image, max_iter=num_iterations, lam=0.001, lr=0.1)
        else:
            final_rec = zero_filled_recon
            mses = []

        # Save final reconstruction
        plt.imshow(final_rec, cmap='gray')
        plt.title(f'Final Reconstruction ({sm_name}, {rm_name})')
        plt.axis('off')
        plt.savefig(f'{outdir}/final_recon.png', bbox_inches='tight')
        plt.close()

        # If iterative, create GIF and plot MSE
        if rm_name in ['cs_wavelet', 'cs_tv']:
            images = []
            for i in range(1, num_iterations+1):
                fname = f'{outdir}/iter_{i}.png'
                if os.path.exists(fname):
                    images.append(imageio.imread(fname))
            imageio.mimsave(f'{outdir}/iterations.gif', images, fps=5)

            # Plot MSE over iterations
            plt.plot(range(1, len(mses)+1), mses, marker='o')
            plt.xlabel('Iteration')
            plt.ylabel('MSE')
            plt.title(f'MSE over Iterations ({sm_name}, {rm_name})')
            plt.grid(True)
            plt.savefig(f'{outdir}/mse_plot.png', bbox_inches='tight')
            plt.close()

        # Compute final metrics
        final_mse = mse_metric(final_rec, original_image)
        final_psnr = psnr_metric(final_rec, original_image)
        final_ssim = ssim(final_rec, original_image, data_range=final_rec.max()-final_rec.min())

        final_results[(sm_name, rm_name)] = {
            'fraction_data': fraction_data,
            'MSE': final_mse,
            'PSNR': final_psnr,
            'SSIM': final_ssim
        }

# Compare methods for each sampling pattern
for sm_name in sampling_methods.keys():
    fig, axes = plt.subplots(1, len(recon_methods), figsize=(15,5))
    for idx, rm_name in enumerate(recon_methods.keys()):
        outdir = f'{result_dir}/{sm_name}/{rm_name}'
        final_img_path = f'{outdir}/final_recon.png'
        if os.path.exists(final_img_path):
            img = plt.imread(final_img_path)
            axes[idx].imshow(img, cmap='gray')
            axes[idx].set_title(f'{rm_name}')
        axes[idx].axis('off')
    plt.suptitle(f"Sampling: {sm_name}")
    plt.savefig(f'{result_dir}/{sm_name}_comparison.png', bbox_inches='tight')
    plt.close()

# Compare sampling patterns for each reconstruction method
for rm_name in recon_methods.keys():
    fig, axes = plt.subplots(1, len(sampling_methods), figsize=(15,5))
    for idx, sm_name in enumerate(sampling_methods.keys()):
        outdir = f'{result_dir}/{sm_name}/{rm_name}'
        final_img_path = f'{outdir}/final_recon.png'
        if os.path.exists(final_img_path):
            img = plt.imread(final_img_path)
            axes[idx].imshow(img, cmap='gray')
            axes[idx].set_title(f'{sm_name}')
        axes[idx].axis('off')
    plt.suptitle(f"Reconstruction: {rm_name}")
    plt.savefig(f'{result_dir}/{rm_name}_comparison.png', bbox_inches='tight')
    plt.close()

# ============================================================
# Create a Results Table
# ============================================================
# We create a table that shows fraction_data, MSE, PSNR, SSIM for each combo

sampling_list = list(sampling_methods.keys())
recon_list = list(recon_methods.keys())

table_header = ["Sampling\\Reconstruction"] + recon_list
rows = []

for sm_name in sampling_list:
    row = [sm_name]
    for rm_name in recon_list:
        res = final_results.get((sm_name, rm_name), None)
        if res is not None:
            row.append(f"Frac: {res['fraction_data']:.3f}, MSE: {res['MSE']:.3e}, PSNR: {res['PSNR']:.2f}, SSIM: {res['SSIM']:.3f}")
        else:
            row.append("N/A")
    rows.append(row)

# Print table as text
print("Final Results Table (Fraction of Data, MSE, PSNR, SSIM):")
col_widths = [max(len(r[i]) for r in ([table_header]+rows)) for i in range(len(table_header))]
format_str = " | ".join(["{{:<{}}}".format(w) for w in col_widths])
print(format_str.format(*table_header))
print("-"*sum(col_widths) + "-"*(3*(len(table_header)-1)))
for row in rows:
    print(format_str.format(*row))

print("Processing complete. Results saved in the 'results' directory.")
