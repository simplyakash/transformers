**Types of Noise in Drone Imaging and Methods to Remove Them**
Drone cameras capture images in dynamic environments (wind, vibration, lighting changes, transmission errors).
Because of this, several types of noise can appear in drone images.

1. Gaussian Noise

Meaning

Random variation in pixel intensity following a normal distribution.

Cause

sensor noise
low light conditions
electronic interference

Visual effect

grainy appearance across the image

Mathematical model

I_noisy = I + N(0, σ²)

Removal methods

Gaussian blur
Wiener filter
Bilateral filter
Non-local means denoising

Example filters used
Typical 3×3 Gaussian mask
1 2 1
2 4 2
1 2 1
cv2.GaussianBlur()
cv2.fastNlMeansDenoising()
2. Salt and Pepper Noise

Meaning

random pixels become completely black or white

Cause

sensor malfunction
data transmission errors
bit errors

Visual effect

isolated white and black dots

Example

pixel value → 0 or 255

Removal methods

Median filter
Adaptive median filter
Morphological filtering

Example filter

cv2.medianBlur()

Median filters work well because they remove extreme values while preserving edges.

3. Speckle Noise

Meaning

multiplicative noise affecting intensity values

Model

I_noisy = I + I * noise

Common in

radar imaging
SAR drones
medical ultrasound

Visual effect

granular texture

Removal methods

Lee filter
Frost filter
Kuan filter
Median filter

Deep learning denoising models are also used.

4. Motion Blur (Drone Movement Noise)

Meaning

blur caused by drone movement during exposure

Causes

wind
drone vibration
rapid movement
slow shutter speed

Visual effect

streaking or directional blur

Removal methods

deblurring algorithms
Wiener deconvolution
blind deconvolution
motion stabilization

Prevention

gimbal stabilization
faster shutter speed
image stabilization
5. Thermal Noise (Sensor Noise)

Meaning

noise generated due to heat in the camera sensor

Common in

low-light drone photography
night surveillance drones

Visual effect

random pixel fluctuations

Removal methods

temporal averaging
Gaussian filtering
deep learning denoising
6. Compression Noise

Meaning

artifacts introduced by video/image compression

Cause

JPEG compression
video streaming compression
low bitrate transmission

Visual effect

blocking artifacts
loss of detail

Removal methods

deblocking filters
deep learning super-resolution
bilateral filtering
7. Atmospheric Noise

Meaning

distortions caused by environmental conditions

Examples

fog
dust
haze
smoke

Visual effect

reduced contrast
blurry appearance

Removal methods

dehazing algorithms
contrast enhancement
CLAHE

Example algorithms

Dark Channel Prior
Retinex
8. Transmission Noise

Meaning

noise caused during wireless transmission from drone to ground station

Cause

packet loss
signal interference

Removal methods

error correction codes
packet retransmission
temporal filtering
Common Denoising Techniques Used in Drone Vision Systems

Spatial filtering

Gaussian filter
Median filter
Bilateral filter

Frequency domain filtering

Fourier filtering
Wiener filtering

Advanced algorithms

Non-local means denoising
BM3D denoising

Deep learning methods

DnCNN
Noise2Noise
Noise2Void
Practical Drone Vision Pipeline

Typical preprocessing pipeline in drone vision:

Raw Drone Image
↓
Noise Reduction
↓
Contrast Enhancement
↓
Feature Extraction
↓
Object Detection / Mapping
Summary
Noise Type	Cause	Removal Method
Gaussian noise	sensor noise	Gaussian / Wiener filter
Salt & pepper	transmission error	Median filter
Speckle	multiplicative noise	Lee / Frost filter
Motion blur	drone movement	deconvolution
Thermal noise	sensor heat	temporal averaging
Compression noise	video compression	deblocking filters
Atmospheric noise	haze/fog	dehazing algorithms

If you'd like, I can also explain something very important for drone vision pipelines:

Image stabilization algorithms used in drones (optical + digital stabilization) — these are often asked in robotics / drone CV interviews.

please provide the discreet filters mask also with eah filter

Below is a README-friendly list of common discrete filter masks used to remove different types of noise (especially relevant in drone vision pipelines).
All masks are plain text so they paste correctly into Markdown/README.

Discrete Filter Masks for Noise Removal

Image denoising in drones often uses spatial convolution filters.
Each filter is applied using a kernel (mask) that slides over the image.

General convolution form:

Output(x,y) = Σ Σ  Image(x+i, y+j) * Kernel(i,j)
1. Mean Filter (Average Filter)

Purpose

removes Gaussian noise
smooths image

3×3 mask

1/9  1/9  1/9
1/9  1/9  1/9
1/9  1/9  1/9

Equivalent form

1 1 1
1 1 1
1 1 1   * 1/9

Effect

reduces random noise
but blurs edges
2. Gaussian Filter

Purpose

removes Gaussian noise
preserves structure better than mean filter



Normalized mask

1/16  2/16  1/16
2/16  4/16  2/16
1/16  2/16  1/16

Effect

center pixel has higher weight
smooths noise while keeping edges better
3. Median Filter

Purpose

removes salt and pepper noise
preserves edges

Mask window

3 × 3 window

Example

Input window

10   12   11
255  9    8
7    6    5

Sorted values

5 6 7 8 9 10 11 12 255

Output

median = 9

Note

median filter does not use multiplication mask
it replaces center pixel with median value
4. Bilateral Filter (Edge Preserving)

Purpose

removes Gaussian noise
preserves edges

Concept

weights depend on
spatial distance
intensity difference

Typical spatial mask (3×3 example)

1 2 1
2 4 2
1 2 1

But weights are also adjusted based on:

pixel intensity similarity

Effect

smooths noise
keeps sharp edges
5. Laplacian Filter (Edge Enhancement + Noise Detection)

Purpose

detect edges
used in sharpening and deblurring

Mask

0  -1   0
-1  4  -1
0  -1   0

Alternative mask

-1 -1 -1
-1  8 -1
-1 -1 -1

Effect

highlights intensity changes
useful for edge sharpening
6. Sobel Filter (Edge Detection)

Purpose

detect horizontal and vertical edges

Horizontal Sobel mask

-1  0  1
-2  0  2
-1  0  1

Vertical Sobel mask

-1 -2 -1
 0  0  0
 1  2  1

Effect

detects gradient direction

Used in

object detection
feature extraction
7. Motion Blur Filter

Purpose

simulate or remove motion blur

Example horizontal motion mask

1 1 1 1 1
0 0 0 0 0
0 0 0 0 0
0 0 0 0 0
0 0 0 0 0

Normalized

divide by 5

Effect

blur in motion direction

Deblurring uses inverse filtering or Wiener filtering.

8. Wiener Filter (Adaptive Noise Removal)

Purpose

removes Gaussian noise and blur

Instead of a fixed mask it uses

local mean
local variance
noise variance

Adaptive formula

output = local_mean + (variance - noise_variance)/variance * (pixel - local_mean)

Effect

adaptive noise reduction
9. Sharpening Filter

Purpose

enhance edges after denoising

Mask

0  -1   0
-1  5  -1
0  -1   0

Effect

enhances edges
improves contrast
Practical Drone Vision Filtering Pipeline

Typical preprocessing for drone imagery:

Drone image
↓
Gaussian filter
↓
Median filter
↓
Bilateral filter
↓
Edge detection
↓
Object detection
Summary
Filter	Noise Removed	Mask Type
Mean	Gaussian noise	uniform mask
Gaussian	Gaussian noise	weighted mask
Median	salt & pepper	nonlinear filter
Bilateral	Gaussian noise	adaptive mask
Wiener	Gaussian + blur	adaptive
Laplacian	edge detection	high-pass
Sobel	gradient detection	directional

If you'd like, I can also show you something very useful for interviews and drone CV systems:

Difference between low-pass filters, high-pass filters, and band-pass filters with masks and frequency intuition.

This helps understand why certain filters remove noise better than others.
 caemra settings:
 # 📸 Camera Settings Explained (Beginner Guide)

Understanding camera settings becomes easy when you learn the **Exposure Triangle** — the three main controls that affect how your photo looks.

---

## 🔺 Exposure Triangle

### 1. Shutter Speed
**Controls:** Motion + Light duration

- Fast (e.g., 1/1000s)
  - Freezes motion
  - Lets in less light
- Slow (e.g., 1/10s, 1s)
  - Creates motion blur
  - Lets in more light

---

### 2. Aperture (f-number)
**Controls:** Light + Background blur (Depth of Field)

- Low f-number (f/1.8)
  - Wide opening
  - More light
  - Blurry background (bokeh)
- High f-number (f/16)
  - Small opening
  - Less light
  - Sharp background

---

### 3. ISO
**Controls:** Sensor sensitivity (brightness)

- Low ISO (100)
  - Clean image (less noise)
  - Needs more light
- High ISO (1600+)
  - Brighter image
  - More noise/grain

---

## ⚖️ How They Work Together

| Setting        | Affects Light | Affects Look                |
|----------------|--------------|-----------------------------|
| Shutter Speed  | Yes          | Motion blur / sharpness     |
| Aperture       | Yes          | Background blur             |
| ISO            | Yes          | Noise / grain               |

---

## 🎨 Additional Important Settings

### White Balance
**Controls:** Color temperature

- Auto → usually fine
- Adjust if colors look off:
  - Tungsten → cooler (blue)
  - Shade → warmer

---

### Focus (AF / MF)
**Controls:** What is sharp

- AF (Auto Focus) → camera decides
- MF (Manual Focus) → you decide

---

### Focal Length (Lens Zoom)
**Controls:** Field of view

- Wide (18mm) → more scene, slight distortion
- Standard (50mm) → natural look
- Telephoto (200mm) → zoomed in, background compression

---

## 📷 Example Settings

### Portrait (Blurred Background)
- Aperture: f/1.8 – f/2.8
- Shutter Speed: ~1/100s
- ISO: 100–400

---

### Landscape (Everything Sharp)
- Aperture: f/8 – f/11
- Shutter Speed: Adjust as needed
- ISO: 100

---

### Action Photography
- Shutter Speed: ~1/1000s
- Aperture: f/2.8 – f/5.6
- ISO: Adjust for brightness

---

### Night Photography
- Shutter Speed: Slow (use tripod)
- Aperture: Wide (f/1.8)
- ISO: 800–3200

---

## 🧠 Quick Summary

- **Shutter Speed** → Controls time (motion blur)
- **Aperture** → Controls opening (depth of field)
- **ISO** → Controls sensitivity (brightness & noise)

---


# 📸 Global Shutter vs Rolling Shutter (Camera Sensors Explained)

Understanding shutter types helps you avoid weird distortions in photos and videos.

---

## 🟢 Global Shutter

**How it works:**
- The entire sensor captures the image **at the same time**

### ✅ Result:
- No motion distortion
- Moving objects appear natural

### 📷 Examples:
- Spinning fan → blades look normal  
- Fast-moving car → no bending or skew  

### 👍 Pros:
- No distortion
- Accurate motion capture
- Ideal for:
  - Sports photography
  - Industrial vision systems
  - Scientific imaging

### 👎 Cons:
- More expensive
- Sometimes lower dynamic range

---

## 🔵 Rolling Shutter (Common in Most Cameras)

**How it works:**
- The sensor captures the image **line by line (top to bottom)**

---

### ⚠️ Effects of Rolling Shutter

#### 1. Jello Effect
- Video looks wobbly when camera shakes

#### 2. Skew / Bending
- Fast-moving objects appear slanted

#### 3. Partial Exposure
- Flash may only illuminate part of the image

---

### 📷 Examples:
- Spinning fan → appears curved  
- Moving car → looks tilted or stretched  

---

## ⚖️ Comparison Table

| Feature            | Global Shutter        | Rolling Shutter        |
|--------------------|----------------------|------------------------|
| Capture Method     | All at once          | Line-by-line           |
| Motion Distortion  | ❌ None              | ⚠️ Yes                 |
| Cost               | 💸 Expensive         | 💰 Cheaper             |
| Common Usage       | High-end cameras     | Phones, DSLRs, mirrorless |

---

## 🧠 Easy Analogy

- **Global Shutter** → Like a flash (everything captured instantly)  
- **Rolling Shutter** → Like scanning a document (top to bottom)

---

## 📱 Real-World Usage

Most modern devices use **rolling shutter**, including:
- Smartphones  
- Mirrorless cameras  
- DSLRs  

---

## 🎯 When It Matters

Rolling shutter issues appear when:
- Shooting fast-moving subjects  
- Recording video with camera movement  
- Capturing spinning objects (fans, propellers)  

---

## 🔥 Tips to Reduce Rolling Shutter Effects

- Use a **faster shutter speed**
- Avoid rapid camera movement (panning)
- Keep your camera steady
- Use stabilization (tripod/gimbal)

---

## 🧠 Summary

- **Global Shutter** → Best for accuracy, no distortion  
- **Rolling Shutter** → More common, but can cause motion artifacts  

---
