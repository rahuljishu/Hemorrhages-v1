# app.py
import streamlit as st
import cv2
import numpy as np
from skimage.filters import frangi
from skimage.morphology import remove_small_objects, disk
from skimage import exposure, morphology, filters
import matplotlib.pyplot as plt

st.set_option('deprecation.showPyplotGlobalUse', False)

def vessel_leakage_detection(image):
    # 1. Enhance vessels
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    vessel_enhanced = frangi(gray)
    # 2. Binarize and clean
    _, vessel_binary = cv2.threshold((vessel_enhanced * 255).astype(np.uint8), 20, 255, cv2.THRESH_BINARY)
    vessel_binary = remove_small_objects(vessel_binary.astype(bool), min_size=100).astype(np.uint8) * 255
    # 3. Mask red hemorrhages
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, (0, 120, 70), (10, 255, 255))
    m2 = cv2.inRange(hsv, (170, 120, 70), (180, 255, 255))
    red_areas = cv2.bitwise_or(m1, m2)
    # 4. Find overlap = leaking vessels
    leaking = cv2.bitwise_and(vessel_binary, red_areas)
    # 5. Highlight on original
    highlighted = image.copy()
    highlighted[leaking == 255] = [0, 0, 255]
    return image, red_areas, highlighted

def hemorrhage_detection(image):
    # 1. Standardize size & extract green channel
    resized = cv2.resize(image, (512, 512))
    green = resized[:, :, 1]
    # 2. CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_green = clahe.apply(green)
    # 3. Complement, opening, subtraction
    comp = 255 - clahe_green
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened = cv2.morphologyEx(comp, cv2.MORPH_OPEN, kernel)
    subtracted = cv2.subtract(comp, opened)
    # 4. Median & further subtraction
    med = cv2.medianBlur(subtracted, 5)
    final_sub = cv2.subtract(med, opened)
    # 5. Intensity adjust & complement
    adjusted = exposure.rescale_intensity(final_sub, in_range=(50, 200))
    final_comp = 255 - adjusted
    # 6. Local thresholding + closing
    thresh = filters.threshold_local(final_comp, block_size=51, offset=10)
    binary = final_comp > thresh
    closed = morphology.binary_closing(binary, footprint=disk(3))
    # 7. Convert boolean to uint8 for display
    bin_img = (binary.astype(np.uint8) * 255)
    closed_img = (closed.astype(np.uint8) * 255)
    return resized, clahe_green, comp, adjusted, bin_img, closed_img

def main():
    st.title("Fundus Image: Vessel Leakage & Hemorrhage Detection")
    st.write("Upload a fundus image and see (1) leaking vessels highlighted in red, and (2) step‑wise hemorrhage detection.")

    uploaded = st.file_uploader("Choose a fundus image...", type=["png","jpg","jpeg"])
    if not uploaded:
        st.info("🔍 Please upload an image file.")
        return

    # Read image from uploaded file
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        st.error("⚠️ Could not decode image. Try a different file.")
        return

    # Run both pipelines
    orig1, red_areas, leak = vessel_leakage_detection(img)
    (resized, clahe_g, comp, adj, bin_img, closed_img) = hemorrhage_detection(img)

    st.subheader("1) Vessel Leakage Detection")
    c1, c2, c3 = st.columns(3)
    c1.image(cv2.cvtColor(orig1, cv2.COLOR_BGR2RGB), caption="Original", use_column_width=True)
    c2.image(red_areas, caption="Red Lesion Mask", use_column_width=True)
    c3.image(cv2.cvtColor(leak, cv2.COLOR_BGR2RGB), caption="Leaking Vessels Highlighted", use_column_width=True)

    st.subheader("2) Hemorrhage Detection Steps")
    steps = [
        ("Resized Original", resized),
        ("CLAHE on Green", clahe_g),
        ("Complemented", comp),
        ("Intensity Adjusted", adj),
        ("Local Threshold (Binary)", bin_img),
        ("Closed (Final)", closed_img)
    ]
    for title, im in steps:
        st.image(im if im.ndim==2 else cv2.cvtColor(im, cv2.COLOR_BGR2RGB),
                 caption=title, use_column_width=True)

if __name__ == "__main__":
    main()
