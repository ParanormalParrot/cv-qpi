import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter
import trimesh

from skimage.restoration import unwrap_phase

def mask_center(img, radius = 10, rotation90 = 0):
    h, w = np.rot90(img, rotation90).shape
    cy, cx = h//2, w//2
    Y, X = np.ogrid[:h, :w]
    mask = ((X-cx)**2 + (Y-cy)**2 <= radius**2) | ~((X >= cx) & (Y <= cy))
    mask = np.rot90(mask, rotation90)
    img_masked = img.copy()
    img_masked[mask] = 0
    return img_masked


def find_plus_one_order_position(img, mask_radius=15, rotation90 = 0):
    h, w = img.shape
    cy, cx = h//2, w//2
    spectrum_img = img.copy()
    spectrum_masked = mask_center(spectrum_img, rotation90=rotation90)
    f_y, f_x = np.unravel_index(np.argmax(spectrum_masked), spectrum_masked.shape)
    return f_x, f_y

def get_spectrum_image_from_fft(fft):
    return np.log(1 + np.abs(fft))

def filter_spectrum(fft, f_x, f_y, radius=20):
    h, w = fft.shape
    cy, cx = h//2, w//2

    filter_mask = np.zeros((h, w), dtype=bool)
    Y, X = np.ogrid[:h, :w]
    dist_sq = (Y - f_y)**2 + (X - f_x)**2
    filter_mask = dist_sq < radius**2
    fft_filtered = np.zeros_like(fft)
    fft_filtered[filter_mask] = fft[filter_mask]
    return fft_filtered

def center_spectrum(fft, f_x, f_y):
    h, w = fft.shape
    c_y, c_x = h//2, w//2
    spectrum_centred = np.roll(fft, -int(f_y - c_y), axis=0)
    spectrum_centred = np.roll(spectrum_centred, -int(f_x - c_x), axis=1)
    return spectrum_centred


def enhance_fringes_contrast(img, boost_factor=2.0, mask_radius=20, filter_radius=15):
    if img.dtype != np.float32 and img.dtype != np.float64:
        img_float = img.astype(np.float32)
    else:
        img_float = img
    
    fft = np.fft.fftshift(np.fft.fft2(img_float))
    h, w = img_float.shape
    cy, cx = h//2, w//2

    Y, X = np.ogrid[:h, :w]
    center_mask = (X-cx)**2 + (Y-cy)**2 <= mask_radius**2
    spectrum = np.abs(fft)
    spectrum_masked = spectrum.copy()
    spectrum_masked[center_mask] = 0

    max_y, max_x = np.unravel_index(np.argmax(spectrum_masked), spectrum_masked.shape)

    pos_mask = (Y - max_y)**2 + (X - max_x)**2 < filter_radius**2
    fft[pos_mask] *= boost_factor

    enhanced_img = np.abs(np.fft.ifft2(np.fft.ifftshift(fft)))
    
    enhanced_img = np.clip(enhanced_img, 0, 255)
    enhanced_img = enhanced_img.astype(np.uint8)
    
    return enhanced_img


def get_phase_from_image(img, mask_radius = 15, filter_radius = 15):
    fft = np.fft.fftshift(np.fft.fft2(img))
    spectrum = np.log(1 + np.abs(fft))
    f_x, f_y = find_plus_one_order_position(spectrum, mask_radius = mask_radius)
    filtered_spectrum = filter_spectrum(fft, f_x, f_y, radius=filter_radius)
    centered_spectrum = center_spectrum(filtered_spectrum, f_x, f_y)
    complex_field = np.fft.ifft2(np.fft.ifftshift(centered_spectrum))
    phase_wrapped = np.angle(complex_field)
    phase_unwrapped = unwrap_phase(phase_wrapped)
    return phase_unwrapped


def get_3d_phase(phase, pixel_size=2.2, wavelength=0.65, zoom=80):
    h, w = phase.shape
    
    step = 1
    x = np.arange(0, w, step)
    y = np.arange(0, h, step)
    X_3d, Y_3d = np.meshgrid(x, y)
    Z_3d = phase[::step, ::step]
    
    X_microns = X_3d * pixel_size / zoom
    Y_microns = Y_3d * pixel_size / zoom
    Z_microns = Z_3d * wavelength / (2 * np.pi)
    
    return X_microns, Y_microns, Z_microns

def plot_3D_phase(phase, pixel_size=2.2, wavelength=0.65, zoom=80,
                  standalone=True, show_colorbar=True):

    X_microns, Y_microns, Z_microns = get_3d_phase(phase, pixel_size=pixel_size,
                                                    wavelength=wavelength, zoom=zoom)

    fig = None
    ax = None
    if standalone:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = plt.gcf()
        ax = plt.gca()

    z_min = Z_microns.min()
    z_max = Z_microns.max()

    surf = ax.plot_surface(X_microns, Y_microns, Z_microns,
                           cmap='turbo',
                           alpha=0.9,
                           linewidth=0.1,
                           antialiased=True,
                           vmin=z_min,
                           vmax=z_max)

    if show_colorbar:
        fig.colorbar(surf, shrink=0.5, aspect=10, label='Height (μm)')

    ax.view_init(elev=30, azim=45)
    ax.set_box_aspect([1, 1, 1])

    ax.set_xlabel('X (μm)')
    ax.set_ylabel('Y (μm)')

def plot_3D_phase_from_image(img, pixel_size = 2.2, wavelength = 0.65, zoom = 80, mask_radius = 15, filter_radius = 15, standalone=True):
    fft = np.fft.fftshift(np.fft.fft2(img))
    spectrum = np.log(1 + np.abs(fft))
    f_x, f_y = find_plus_one_order_position(spectrum, mask_radius = mask_radius)
    filtered_spectrum = filter_spectrum(fft, f_x, f_y, radius=filter_radius)
    centered_spectrum = center_spectrum(filtered_spectrum, f_x, f_y)
    amplitude = np.fft.ifft2(np.fft.ifftshift(centered_spectrum))
    phase_wrapped = np.angle(amplitude)
    phase_unwrapped = unwrap_phase(phase_wrapped)
    plot_3D_phase(phase_unwrapped, pixel_size = pixel_size, wavelength = wavelength, zoom = zoom, standalone=standalone)