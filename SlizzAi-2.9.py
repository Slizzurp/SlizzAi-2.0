#!/usr/bin/env python3
"""
SlizzAi v2.9
============

AI-powered image enhancement, adjustment, chatbot, and deep learning image regeneration tool.
Sleek, minimalist chat UI with 10 tabs × 20 image tools, all controlled by chat or buttons.

Author: SlizzAi Team
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageEnhance, ImageFilter, ImageOps, ImageDraw
import numpy as np
import cv2
import threading

# Optional: Deep learning imports (comment out if not using AI features)
import torch
from torchvision import transforms
from torchvision.models import segmentation
# from torchvision.utils import save_image

# ========== Theme ==========
DARK_BG = "#23272a"
LIGHT_BG = "#f7f7fa"
MID_BG = "#36393f"
ACCENT = "#7289da"
WHITE = "#ffffff"
TRANSPARENT = "#23272a"
BORDER = "#e0e0e0"
HEADER = "#fafafa"

FONT = ("Segoe UI", 11)
HEADER_FONT = ("Segoe UI", 13, "bold")
BTN_FONT = ("Segoe UI", 10, "bold")

WINDOW_W = 820
WINDOW_H = 540

CANVAS_W = 320
CANVAS_H = 320

# ========== Helper Functions ==========

def pil_to_cv(img):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def cv_to_pil(img):
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def center_crop(img, size):
    w, h = img.size
    new_w, new_h = size
    left = (w - new_w) // 2
    top = (h - new_h) // 2
    return img.crop((left, top, left + new_w, top + new_h))

def add_drop_shadow(pil_img, offset=10, blur=10, shadow_color=(0,0,0,128)):
    base = pil_img.convert("RGBA")
    shadow = Image.new("RGBA", (base.width + 2*offset, base.height + 2*offset), (0,0,0,0))
    shadow_draw = ImageDraw.Draw(shadow)
    shadow_draw.rectangle([offset, offset, offset+base.width, offset+base.height], fill=shadow_color)
    shadow = shadow.filter(ImageFilter.GaussianBlur(blur))
    shadow.paste(base, (offset, offset), base)
    return shadow

# ========== AI & Image Operations ==========

def ai_upscale(img):
    # Placeholder: Use OpenCV for simple upscaling, replace with real SRGAN for production
    arr = pil_to_cv(img)
    arr = cv2.resize(arr, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    return cv_to_pil(arr)

def ai_inpaint(img):
    # Placeholder: Add random inpainting effect
    arr = pil_to_cv(img)
    mask = np.zeros(arr.shape[:2], np.uint8)
    cv2.rectangle(mask, (arr.shape[1]//4, arr.shape[0]//4), (arr.shape[1]*3//4, arr.shape[0]*3//4), 255, -1)
    arr = cv2.inpaint(arr, mask, 3, cv2.INPAINT_TELEA)
    return cv_to_pil(arr)

def ai_style_transfer(img):
    # Placeholder: Use PIL filters for style, replace with torch Hub model for production
    return img.filter(ImageFilter.CONTOUR)

# Load DeepLabV3 model once for efficiency
_deeplabv3_model = None
def ai_segmentation(img):
    global _deeplabv3_model
    # Placeholder: Use torchvision DeepLabV3 for segmentation
    preprocess = transforms.Compose([
        transforms.Resize((224,224)), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    input_tensor = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        if _deeplabv3_model is None:
            _deeplabv3_model = segmentation.deeplabv3_resnet50(pretrained=True).eval()
        output = _deeplabv3_model(input_tensor)["out"][0]
    mask = output.argmax(0).byte().cpu().numpy()
    mask_img = Image.fromarray(np.uint8(mask*255)).resize(img.size)
    return Image.composite(img, Image.new("RGB", img.size, (128,128,128)), mask_img)

def ai_generate(prompt, size=(256,256)):
    # Placeholder: Generate a blank image with the prompt text
    img = Image.new("RGB", size, (200, 200, 200))
    draw = ImageDraw.Draw(img)
    draw.text((10, size[1]//2), prompt, fill=(50,50,50))
    return img

# ========== UI Classes ==========

class ImageCanvas(tk.Canvas):
    def __init__(self, master, **kwargs):
        super().__init__(master, width=CANVAS_W, height=CANVAS_H, bg=LIGHT_BG, highlightthickness=0, **kwargs)
        self.orig_img = None
        self.display_img = None
        self.zoom = 1.0
        self.angle = 0
        self.blur = 0
        self.shadow = 0
        self.brightness = 1.0
        self.bind("<Configure>", self._redraw)
    def set_image(self, pil_img):
        self.orig_img = pil_img.copy()
        self.update_image()
    def update_image(self):
        if self.orig_img is None: return
        img = self.orig_img.copy()
        # Apply adjustments
        if self.brightness != 1.0:
            img = ImageEnhance.Brightness(img).enhance(self.brightness)
        if self.blur > 0:
            img = img.filter(ImageFilter.GaussianBlur(self.blur))
        if self.angle != 0:
            img = img.rotate(self.angle, expand=True)
        if self.zoom != 1.0:
            w, h = img.size
            img = img.resize((int(w*self.zoom), int(h*self.zoom)), Image.LANCZOS)
        if self.shadow > 0:
            img = add_drop_shadow(img, offset=self.shadow, blur=self.shadow)
        # Resize to fit canvas
        img.thumbnail((self.winfo_width(), self.winfo_height()))
        self.display_img = ImageTk.PhotoImage(img)
        self.delete("all")
        self.create_image(self.winfo_width()//2, self.winfo_height()//2, image=self.display_img)
    def _redraw(self, _=None):
        self.update_image()

class ChatPanel(tk.Frame):
    def __init__(self, master, app, **kwargs):
        super().__init__(master, bg=DARK_BG, **kwargs)
        self.app = app
        self.pack_propagate(0)
        self.chat_log = tk.Text(self, bg=MID_BG, fg=WHITE, font=FONT, bd=0, relief=tk.FLAT, wrap=tk.WORD, height=18)
        self.chat_log.pack(fill=tk.BOTH, expand=True, padx=10, pady=(10,5))
        self.chat_log.config(state=tk.DISABLED)
        self.chat_entry = tk.Entry(self, bg=LIGHT_BG, fg=DARK_BG, font=FONT, relief=tk.FLAT)
        self.chat_entry.pack(fill=tk.X, padx=10, pady=(0,10))
        self.chat_entry.bind("<Return>", self.on_send)
        self.insert_bot("Welcome to SlizzAi v2.9! Upload an image and chat with me to enhance or transform it.")
    def insert_user(self, text):
        self.chat_log.config(state=tk.NORMAL)
        self.chat_log.insert(tk.END, f"You: {text}\n")
        self.chat_log.config(state=tk.DISABLED)
        self.chat_log.see(tk.END)
    def insert_bot(self, text):
        self.chat_log.config(state=tk.NORMAL)
        self.chat_log.insert(tk.END, f"SlizzAi: {text}\n")
        self.chat_log.config(state=tk.DISABLED)
        self.chat_log.see(tk.END)
    def on_send(self, _=None):
        text = self.chat_entry.get().strip()
        if not text: return
        self.insert_user(text)
        self.chat_entry.delete(0, tk.END)
        threading.Thread(target=self.handle_command, args=(text,)).start()
    def handle_command(self, text):
        # Parse command and apply effect
        response = self.app.handle_chat_command(text)
        self.insert_bot(response)

class SidePanel(tk.Frame):
    def __init__(self, master, app, **kwargs):
        super().__init__(master, bg=TRANSPARENT, **kwargs)
        self.app = app
        self.tabs = []
        self.tab_frames = []
        self.tab_control = ttk.Notebook(self)
        self.tab_control.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)
        self.populate_tabs()
    def populate_tabs(self):
        tab_names = [
            "Enhance", "Adjust", "AI", "Art", "Effects",
            "Filters", "Restore", "Generate", "Segment", "Fun"
        ]
        tab_options = [
            # Each tab gets 20 creative or technical operations
            [
                ("Sharpen", lambda img: img.filter(ImageFilter.SHARPEN)),
                ("Auto Contrast", lambda img: ImageOps.autocontrast(img)),
                ("Denoise", lambda img: img.filter(ImageFilter.MedianFilter(3))),
                ("HDR", lambda img: img.filter(ImageFilter.DETAIL)),
                ("Super-Res", ai_upscale),
                ("Color Boost", lambda img: ImageEnhance.Color(img).enhance(1.5)),
                ("Brightness+", lambda img: ImageEnhance.Brightness(img).enhance(1.2)),
                ("Brightness-", lambda img: ImageEnhance.Brightness(img).enhance(0.8)),
                ("Contrast+", lambda img: ImageEnhance.Contrast(img).enhance(1.4)),
                ("Contrast-", lambda img: ImageEnhance.Contrast(img).enhance(0.7)),
                ("Gamma Up", lambda img: img.point(lambda p: min(255, int(p**1.1)))),
                ("Gamma Down", lambda img: img.point(lambda p: max(0, int(p**0.9)))),
                ("Edge Enhance", lambda img: img.filter(ImageFilter.EDGE_ENHANCE)),
                ("Smooth", lambda img: img.filter(ImageFilter.SMOOTH)),
                ("Posterize", lambda img: ImageOps.posterize(img, 4)),
                ("Solarize", lambda img: ImageOps.solarize(img, threshold=128)),
                ("Invert", lambda img: ImageOps.invert(img)),
                ("Sepia", lambda img: ImageOps.colorize(ImageOps.grayscale(img), "#704214", "#C0C080")),
                ("Vivid", lambda img: ImageEnhance.Color(img).enhance(2.0)),
                ("Dehaze", lambda img: img.filter(ImageFilter.ModeFilter(5))),
            ],
            [
                ("Crop Center", lambda img: center_crop(img, (img.width//2, img.height//2))),
                ("Resize 50%", lambda img: img.resize((img.width//2, img.height//2))),
                ("Resize 200%", lambda img: img.resize((img.width*2, img.height*2))),
                ("Rotate 90°", lambda img: img.rotate(90, expand=True)),
                ("Rotate 180°", lambda img: img.rotate(180, expand=True)),
                ("Flip H", lambda img: img.transpose(Image.FLIP_LEFT_RIGHT)),
                ("Flip V", lambda img: img.transpose(Image.FLIP_TOP_BOTTOM)),
                ("Add Border", lambda img: ImageOps.expand(img, border=10, fill=ACCENT)),
                ("Add Frame", lambda img: ImageOps.expand(img, border=20, fill=WHITE)),
                ("Round Corners", lambda img: img.filter(ImageFilter.SMOOTH_MORE)),
                ("Pad 10%", lambda img: ImageOps.expand(img, border=int(0.1*img.width), fill=LIGHT_BG)),
                ("Square Crop", lambda img: center_crop(img, (min(img.size), min(img.size)))),
                ("Circle Crop", lambda img: img), # Placeholder
                ("Tile", lambda img: img), # Placeholder
                ("Mirror", lambda img: img.transpose(Image.FLIP_LEFT_RIGHT)),
                ("Perspective", lambda img: img), # Placeholder
                ("Skew", lambda img: img), # Placeholder
                ("Shear", lambda img: img), # Placeholder
                ("Stretch", lambda img: img), # Placeholder
                ("Trim", lambda img: img.crop(img.getbbox())),
            ],
            [
                ("AI Upscale", ai_upscale),
                ("AI Denoise", lambda img: img.filter(ImageFilter.MedianFilter(5))),
                ("AI Inpaint", ai_inpaint),
                ("AI Style Transfer", ai_style_transfer),
                ("AI Segmentation", ai_segmentation),
                ("AI Generate", lambda _: ai_generate("A cat in the style of Van Gogh")),
                ("AI Colorize", lambda img: img.convert("L").convert("RGB")),
                ("AI Remove BG", lambda img: img), # Placeholder
                ("AI Face Enhance", lambda img: img), # Placeholder
                ("AI Deblur", lambda img: img.filter(ImageFilter.SMOOTH)),
                ("AI Super-Res", ai_upscale),
                ("AI Cartoon", lambda img: img.filter(ImageFilter.CONTOUR)),
                ("AI Line Art", lambda img: img.filter(ImageFilter.FIND_EDGES)),
                ("AI HDR", lambda img: img.filter(ImageFilter.DETAIL)),
                ("AI Night2Day", lambda img: img), # Placeholder
                ("AI Day2Night", lambda img: img), # Placeholder
                ("AI Old2Young", lambda img: img), # Placeholder
                ("AI Gender Swap", lambda img: img), # Placeholder
                ("AI Age Up", lambda img: img), # Placeholder
                ("AI Age Down", lambda img: img), # Placeholder
            ],
            [
                ("Oil Paint", lambda img: img.filter(ImageFilter.SMOOTH)),
                ("Watercolor", lambda img: img.filter(ImageFilter.BLUR)),
                ("Sketch", lambda img: img.filter(ImageFilter.CONTOUR)),
                ("Charcoal", lambda img: img.filter(ImageFilter.EDGE_ENHANCE)),
                ("Cartoon", lambda img: img.filter(ImageFilter.CONTOUR)),
                ("Pop Art", lambda img: img.filter(ImageFilter.EMBOSS)),
                ("Pixelate", lambda img: img.resize((32,32)).resize(img.size)),
                ("Halftone", lambda img: img), # Placeholder
                ("Mosaic", lambda img: img), # Placeholder
                ("Stipple", lambda img: img), # Placeholder
                ("Pastel", lambda img: img), # Placeholder
                ("Impressionist", lambda img: img), # Placeholder
                ("Surreal", lambda img: img), # Placeholder
                ("Abstract", lambda img: img), # Placeholder
                ("Color Splash", lambda img: img), # Placeholder
                ("Glitch", lambda img: img), # Placeholder
                ("Noise", lambda img: img), # Placeholder
                ("Vaporwave", lambda img: img), # Placeholder
                ("Cyberpunk", lambda img: img), # Placeholder
                ("Graffiti", lambda img: img), # Placeholder
            ],
            [
                ("Blur", lambda img: img.filter(ImageFilter.BLUR)),
                ("Gaussian Blur", lambda img: img.filter(ImageFilter.GaussianBlur(3))),
                ("Box Blur", lambda img: img.filter(ImageFilter.BoxBlur(2))),
                ("Motion Blur", lambda img: img), # Placeholder
                ("Radial Blur", lambda img: img), # Placeholder
                ("Glow", lambda img: img), # Placeholder
                ("Shadow", lambda img: add_drop_shadow(img, offset=15, blur=10)),
                ("Emboss", lambda img: img.filter(ImageFilter.EMBOSS)),
                ("Find Edges", lambda img: img.filter(ImageFilter.FIND_EDGES)),
                ("Edge Enhance", lambda img: img.filter(ImageFilter.EDGE_ENHANCE)),
                ("Contour", lambda img: img.filter(ImageFilter.CONTOUR)),
                ("Detail", lambda img: img.filter(ImageFilter.DETAIL)),
                ("Smooth", lambda img: img.filter(ImageFilter.SMOOTH)),
                ("Sharpen", lambda img: img.filter(ImageFilter.SHARPEN)),
                ("Max Filter", lambda img: img.filter(ImageFilter.MaxFilter(3))),
                ("Min Filter", lambda img: img.filter(ImageFilter.MinFilter(3))),
                ("Median Filter", lambda img: img.filter(ImageFilter.MedianFilter(5))),
                ("Mode Filter", lambda img: img.filter(ImageFilter.ModeFilter(5))),
                ("Rank Filter", lambda img: img), # Placeholder
                ("Unsharp Mask", lambda img: img.filter(ImageFilter.UnsharpMask(radius=2, percent=150))),
            ],
            [
                ("Red Channel", lambda img: img.split()[0]),
                ("Green Channel", lambda img: img.split()[1]),
                ("Blue Channel", lambda img: img.split()[2]),
                ("Gray", lambda img: img.convert("L")),
                ("Invert", lambda img: ImageOps.invert(img)),
                ("Posterize", lambda img: ImageOps.posterize(img, 2)),
                ("Solarize", lambda img: ImageOps.solarize(img, threshold=100)),
                ("Equalize", lambda img: ImageOps.equalize(img)),
                ("Autocontrast", lambda img: ImageOps.autocontrast(img)),
                ("Colorize", lambda img: ImageOps.colorize(img.convert("L"), "#222", "#ff0")),
                ("Threshold", lambda img: img.convert("L").point(lambda p: 255 if p > 128 else 0)),
                ("Dither", lambda img: img.convert("1")),
                ("Quantize", lambda img: img.quantize(colors=8)),
                ("Palette Swap", lambda img: img), # Placeholder
                ("Channel Swap", lambda img: img), # Placeholder
                ("Desaturate", lambda img: ImageEnhance.Color(img).enhance(0)),
                ("Color Balance", lambda img: img), # Placeholder
                ("Tint", lambda img: img), # Placeholder
                ("Hue Shift", lambda img: img), # Placeholder
                ("Saturation", lambda img: img), # Placeholder
            ],
            [
                ("Restore Old", lambda img: img.filter(ImageFilter.DETAIL)),
                ("Remove Scratches", lambda img: img.filter(ImageFilter.SMOOTH)),
                ("De-yellow", lambda img: img), # Placeholder
                ("De-fade", lambda img: img), # Placeholder
                ("De-blur", lambda img: img.filter(ImageFilter.SHARPEN)),
                ("Fill Gaps", lambda img: img), # Placeholder
                ("Recolor", lambda img: img), # Placeholder
                ("Sharpen Faces", lambda img: img), # Placeholder
                ("Reconstruct", lambda img: img), # Placeholder
                ("Recompose", lambda img: img), # Placeholder
                ("Auto Restore", lambda img: img), # Placeholder
                ("Remove Noise", lambda img: img.filter(ImageFilter.MedianFilter(3))),
                ("Remove Spots", lambda img: img), # Placeholder
                ("Balance Tone", lambda img: img), # Placeholder
                ("Contrast Fix", lambda img: img), # Placeholder
                ("Color Fix", lambda img: img), # Placeholder
                ("Edge Fix", lambda img: img), # Placeholder
                ("Smooth Faces", lambda img: img), # Placeholder
                ("Brighten", lambda img: ImageEnhance.Brightness(img).enhance(1.2)),
                ("Darken", lambda img: ImageEnhance.Brightness(img).enhance(0.8)),
            ],
            [
                ("Text to Image", lambda _: ai_generate("A sunset over mountains")),
                ("Expand Canvas", lambda img: ImageOps.expand(img, border=50, fill=WHITE)),
                ("Content Fill", lambda img: img), # Placeholder
                ("Remove Object", lambda img: img), # Placeholder
                ("Add Object", lambda img: img), # Placeholder
                ("Change BG", lambda img: img), # Placeholder
                ("Clone Stamp", lambda img: img), # Placeholder
                ("Pattern Fill", lambda img: img), # Placeholder
                ("Texture", lambda img: img), # Placeholder
                ("Sticker", lambda img: img), # Placeholder
                ("Frame", lambda img: ImageOps.expand(img, border=20, fill=ACCENT)),
                ("Add Text", lambda img: img), # Placeholder
                ("Watermark", lambda img: img), # Placeholder
                ("QR Code", lambda img: img), # Placeholder
                ("Barcode", lambda img: img), # Placeholder
                ("Add Emoji", lambda img: img), # Placeholder
                ("Add Icon", lambda img: img), # Placeholder
                ("Add Signature", lambda img: img), # Placeholder
                ("Add Date", lambda img: img), # Placeholder
                ("Add Location", lambda img: img), # Placeholder
            ],
            [
                ("Segment Person", ai_segmentation),
                ("Segment Sky", lambda img: img), # Placeholder
                ("Segment BG", lambda img: img), # Placeholder
                ("Segment Foreground", lambda img: img), # Placeholder
                ("Segment Object", lambda img: img), # Placeholder
                ("Segment Animal", lambda img: img), # Placeholder
                ("Segment Food", lambda img: img), # Placeholder
                ("Segment Plant", lambda img: img), # Placeholder
                ("Segment Car", lambda img: img), # Placeholder
                ("Segment Road", lambda img: img), # Placeholder
                ("Segment Water", lambda img: img), # Placeholder
                ("Segment Building", lambda img: img), # Placeholder
                ("Segment Face", lambda img: img), # Placeholder
                ("Segment Hand", lambda img: img), # Placeholder
                ("Segment Eye", lambda img: img), # Placeholder
                ("Segment Mouth", lambda img: img), # Placeholder
                ("Segment Nose", lambda img: img), # Placeholder
                ("Segment Ear", lambda img: img), # Placeholder
                ("Segment Hair", lambda img: img), # Placeholder
                ("Segment Clothes", lambda img: img), # Placeholder
            ],
            [
                ("Stickerify", lambda img: img), # Placeholder
                ("Meme", lambda img: img), # Placeholder
                ("Comic", lambda img: img), # Placeholder
                ("Speech Bubble", lambda img: img), # Placeholder
                ("Add Glasses", lambda img: img), # Placeholder
                ("Add Hat", lambda img: img), # Placeholder
                ("Add Beard", lambda img: img), # Placeholder
                ("Add Mask", lambda img: img), # Placeholder
                ("Add Crown", lambda img: img), # Placeholder
                ("Add Wings", lambda img: img), # Placeholder
                ("Add Animal Ears", lambda img: img), # Placeholder
                ("Add Rainbow", lambda img: img), # Placeholder
                ("Add Sparkles", lambda img: img), # Placeholder
                ("Add Fire", lambda img: img), # Placeholder
                ("Add Lightning", lambda img: img), # Placeholder
                ("Add Hearts", lambda img: img), # Placeholder
                ("Add Stars", lambda img: img), # Placeholder
                ("Add Sunglasses", lambda img: img), # Placeholder
                ("Add Tie", lambda img: img), # Placeholder
                ("Add Bow", lambda img: img), # Placeholder
            ],
        ]
        for i, tab_name in enumerate(tab_names):
            frame = tk.Frame(self.tab_control, bg=LIGHT_BG)
            for j, (label, func) in enumerate(tab_options[i]):
                btn = tk.Button(frame, text=label, font=BTN_FONT, bg=WHITE, fg=DARK_BG, bd=0, relief=tk.FLAT,
                                activebackground=ACCENT, activeforeground=WHITE,
                                command=lambda f=func: self.app.apply_effect(f))
                btn.grid(row=j//2, column=j%2, padx=6, pady=3, sticky="ew")
            self.tab_control.add(frame, text=tab_name)
            self.tab_frames.append(frame)

class ControlPanel(tk.Frame):
    def __init__(self, master, app, canvas, **kwargs):
        super().__init__(master, bg=TRANSPARENT, **kwargs)
        self.app = app
        self.canvas = canvas
        self.upload_btn = tk.Button(self, text="Upload", font=BTN_FONT, bg=ACCENT, fg=WHITE, bd=0, relief=tk.FLAT, command=self.upload_image)
        self.save_btn = tk.Button(self, text="Save", font=BTN_FONT, bg=ACCENT, fg=WHITE, bd=0, relief=tk.FLAT, command=self.save_image)
        self.upload_btn.pack(side=tk.LEFT, padx=8, pady=6)
        self.save_btn.pack(side=tk.LEFT, padx=8, pady=6)
        # Sliders for canvas
        self.zoom_slider = tk.Scale(self, from_=0.5, to=2.0, resolution=0.01, orient=tk.HORIZONTAL, label="Zoom", bg=LIGHT_BG, fg=DARK_BG, length=120, command=self.on_zoom)
        self.zoom_slider.set(1.0)
        self.zoom_slider.pack(side=tk.LEFT, padx=4)
        self.angle_slider = tk.Scale(self, from_=-180, to=180, resolution=1, orient=tk.HORIZONTAL, label="Rotate", bg=LIGHT_BG, fg=DARK_BG, length=120, command=self.on_angle)
        self.angle_slider.set(0)
        self.angle_slider.pack(side=tk.LEFT, padx=4)
        self.blur_slider = tk.Scale(self, from_=0, to=10, resolution=0.1, orient=tk.HORIZONTAL, label="Blur", bg=LIGHT_BG, fg=DARK_BG, length=120, command=self.on_blur)
        self.blur_slider.set(0)
        self.blur_slider.pack(side=tk.LEFT, padx=4)
        self.shadow_slider = tk.Scale(self, from_=0, to=30, resolution=1, orient=tk.HORIZONTAL, label="Drop Shadow", bg=LIGHT_BG, fg=DARK_BG, length=120, command=self.on_shadow)
        self.shadow_slider.set(0)
        self.shadow_slider.pack(side=tk.LEFT, padx=4)
        self.brightness_slider = tk.Scale(self, from_=0.2, to=2.0, resolution=0.01, orient=tk.HORIZONTAL, label="Brightness", bg=LIGHT_BG, fg=DARK_BG, length=120, command=self.on_brightness)
        self.brightness_slider.set(1.0)
        self.brightness_slider.pack(side=tk.LEFT, padx=4)
    def upload_image(self):
        file_path = filedialog.askopenfilename(title="Select Image", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")])
        if file_path:
            img = Image.open(file_path).convert("RGB")
            self.app.set_image(img)
    def save_image(self):
        if self.app.canvas.orig_img is None:
            messagebox.showwarning("No Image", "Load and edit an image before saving.")
            return
        file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg;*.jpeg")])
        if file_path:
            # Save the current displayed image with all adjustments
            img = self.app.canvas.orig_img.copy()
            # Apply adjustments before saving
            if self.app.canvas.brightness != 1.0:
                img = ImageEnhance.Brightness(img).enhance(self.app.canvas.brightness)
            if self.app.canvas.blur > 0:
                img = img.filter(ImageFilter.GaussianBlur(self.app.canvas.blur))
            if self.app.canvas.angle != 0:
                img = img.rotate(self.app.canvas.angle, expand=True)
            if self.app.canvas.zoom != 1.0:
                w, h = img.size
                img = img.resize((int(w*self.app.canvas.zoom), int(h*self.app.canvas.zoom)), Image.LANCZOS)
            if self.app.canvas.shadow > 0:
                img = add_drop_shadow(img, offset=self.app.canvas.shadow, blur=self.app.canvas.shadow)
            img.save(file_path)
            messagebox.showinfo("Saved", f"Image saved to {file_path}")
    def on_zoom(self, val):
        self.canvas.zoom = float(val)
        self.canvas.update_image()
    def on_angle(self, val):
        self.canvas.angle = int(val)
        self.canvas.update_image()
    def on_blur(self, val):
        self.canvas.blur = float(val)
        self.canvas.update_image()
    def on_shadow(self, val):
        self.canvas.shadow = int(val)
        self.canvas.update_image()
    def on_brightness(self, val):
        self.canvas.brightness = float(val)
        self.canvas.update_image()

# ========== Main Application ==========

class SlizzAiApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SlizzAi v2.9")
        self.geometry(f"{WINDOW_W}x{WINDOW_H}")
        self.resizable(False, False)
        self.configure(bg=DARK_BG)
        self.orig_img = None
        # Layout: Left chat, right controls
        self.left_frame = tk.Frame(self, bg=DARK_BG, width=WINDOW_W//2, height=WINDOW_H)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)
        self.right_frame = tk.Frame(self, bg=TRANSPARENT, width=WINDOW_W//2, height=WINDOW_H)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False)
        # Chat
        self.chat_panel = ChatPanel(self.left_frame, self)
        self.chat_panel.pack(fill=tk.BOTH, expand=True)
        # Canvas + controls
        self.canvas = ImageCanvas(self.right_frame)
        self.canvas.pack(padx=10, pady=10)
        self.ctrl_panel = ControlPanel(self.right_frame, self, self.canvas)
        self.ctrl_panel.pack(fill=tk.X, padx=10, pady=(0,8))
        # Tabs
        self.side_panel = SidePanel(self.right_frame, self)
        self.side_panel.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)
    def set_image(self, pil_img):
        self.orig_img = pil_img.copy()
        self.canvas.set_image(pil_img)
    def apply_effect(self, func):
        if self.orig_img is None:
            messagebox.showwarning("No Image", "Please upload an image first.")
            return
        try:
            result = func(self.orig_img)
            self.set_image(result)
            self.chat_panel.insert_bot("Effect applied.")
        except Exception as e:
            self.chat_panel.insert_bot(f"Error: {e}")
    def handle_chat_command(self, text):
        # Very basic NLP: check for keywords
        if "upload" in text:
            self.ctrl_panel.upload_image()
            return "Image uploaded."
        if "save" in text:
            self.ctrl_panel.save_image()
            return "Image saved."
        if "zoom" in text:
            self.ctrl_panel.zoom_slider.set(1.5)
            return "Zoom set to 1.5x."
        if "rotate" in text:
            self.ctrl_panel.angle_slider.set(90)
            return "Image rotated 90°."
        if "blur" in text:
            self.ctrl_panel.blur_slider.set(5)
            return "Blur applied."
        if "shadow" in text:
            self.ctrl_panel.shadow_slider.set(15)
            return "Drop shadow applied."
        if "bright" in text:
            self.ctrl_panel.brightness_slider.set(1.5)
            return "Brightness increased."
        # Try to match tab/option
        for tab in self.side_panel.tab_frames:
            for child in tab.winfo_children():
                if isinstance(child, tk.Button) and child["text"].lower() in text.lower():
                    child.invoke()
                    return f"{child['text']} applied."
        return "Sorry, I didn't understand. Try asking for an image effect or adjustment."
if __name__ == "__main__":
    SlizzAiApp().mainloop()
# SlizzAi v2.9 - AI-powered image enhancement and chatbot tool
