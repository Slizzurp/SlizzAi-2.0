#!/usr/bin/env python3
"""
SlizzAi v2.8 - Sleek Dark Edition
=================================
AI-powered image enhancement and chatbot tool with a modern dark gray interface.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from PIL import Image, ImageTk
import cv2
import numpy as np

# =======================
# Constants and Theme
# =======================
APP_TITLE = "SlizzAi 2.8"
WINDOW_GEOMETRY = "900x650"
DARK_BG = "#23272a"         # Main background
DARKER_BG = "#181a1b"       # Panels, entries
LIGHT_DARK = "#2c2f33"      # Widget backgrounds
ACCENT = "#7289da"          # Accent color (buttons, highlights)
FG_COLOR = "#f8f8f2"        # Foreground/text

# =======================
# Image Processing
# =======================
def apply_neural_denoising(image):
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    denoised = cv2.fastNlMeansDenoisingColored(image_cv, None, 10, 10, 7, 21)
    return Image.fromarray(cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB))

def apply_ray_tracing_optimization(image):
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    optimized = cv2.filter2D(image_cv, -1, kernel)
    return Image.fromarray(cv2.cvtColor(optimized, cv2.COLOR_BGR2RGB))

def apply_neural_style_transfer(image):
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    stylized = cv2.stylization(image_cv, sigma_s=60, sigma_r=0.07)
    return Image.fromarray(cv2.cvtColor(stylized, cv2.COLOR_BGR2RGB))

def apply_quantum_compression(image):
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
    _, enc = cv2.imencode('.jpg', image_cv, encode_param)
    comp = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return Image.fromarray(cv2.cvtColor(comp, cv2.COLOR_BGR2RGB))

def apply_hierarchical_texture(image):
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    texture = cv2.detailEnhance(image_cv, sigma_s=10, sigma_r=0.15)
    return Image.fromarray(cv2.cvtColor(texture, cv2.COLOR_BGR2RGB))

def apply_signature_branding(image):
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    watermark = np.zeros_like(image_cv)
    cv2.putText(watermark, "SlizzAi", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (114,137,218), 3)
    branded = cv2.addWeighted(image_cv, 0.8, watermark, 0.2, 0)
    return Image.fromarray(cv2.cvtColor(branded, cv2.COLOR_BGR2RGB))

def apply_anime_cyber_style(image):
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    edges = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,9,9)
    color = cv2.bilateralFilter(image_cv,9,300,300)
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return Image.fromarray(cv2.cvtColor(cartoon, cv2.COLOR_BGR2RGB))

def apply_narrative_scene(image):
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    rows, cols = image_cv.shape[:2]
    kernel_x = cv2.getGaussianKernel(cols, cols/3)
    kernel_y = cv2.getGaussianKernel(rows, rows/3)
    kernel = kernel_y * kernel_x.T
    vignette = np.clip(image_cv * kernel[:,:,np.newaxis], 0, 255).astype(np.uint8)
    return Image.fromarray(cv2.cvtColor(vignette, cv2.COLOR_BGR2RGB))

def apply_atmospheric_simulation(image):
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    haze = cv2.addWeighted(image_cv, 0.7, np.full_like(image_cv, 100), 0.3, 0)
    return Image.fromarray(cv2.cvtColor(haze, cv2.COLOR_BGR2RGB))

def apply_hair_cloth_dynamics(image):
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    blurred = cv2.GaussianBlur(image_cv, (15, 15), 0)
    return Image.fromarray(cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB))

def apply_motion_blur(image):
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    kernel_size = 15
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
    kernel /= kernel_size
    blurred = cv2.filter2D(image_cv, -1, kernel)
    return Image.fromarray(cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB))

def apply_3d_environment(image):
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    blurred = cv2.GaussianBlur(image_cv, (15, 15), 0)
    dof = cv2.addWeighted(image_cv, 0.5, blurred, 0.5, 0)
    return Image.fromarray(cv2.cvtColor(dof, cv2.COLOR_BGR2RGB))

def apply_fractal_zoom(image):
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    rows, cols = image_cv.shape[:2]
    center = (cols//2, rows//2)
    zoomed = cv2.warpAffine(image_cv, cv2.getRotationMatrix2D(center, 0, 1.5), (cols, rows))
    return Image.fromarray(cv2.cvtColor(zoomed, cv2.COLOR_BGR2RGB))

def apply_temporal_filtering(image):
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    filtered = cv2.medianBlur(image_cv, 5)
    return Image.fromarray(cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB))

def apply_material_processing(image):
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(image_cv, -1, kernel)
    return Image.fromarray(cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB))

def apply_hdr_enhancement(image):
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(image_cv, cv2.COLOR_BGR2Lab)
    l,a,b = cv2.split(lab)
    l = cv2.equalizeHist(l)
    lab = cv2.merge((l,a,b))
    hdr = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)
    return Image.fromarray(cv2.cvtColor(hdr, cv2.COLOR_BGR2RGB))

EFFECTS = [
    ("Neural Denoising", apply_neural_denoising),
    ("Ray Tracing Optimization", apply_ray_tracing_optimization),
    ("Neural Style Transfer", apply_neural_style_transfer),
    ("Quantum Compression", apply_quantum_compression),
    ("Hierarchical Texture", apply_hierarchical_texture),
    ("Signature Branding", apply_signature_branding),
    ("Anime/Cyber Style", apply_anime_cyber_style),
    ("Narrative Scene", apply_narrative_scene),
    ("Atmospheric Simulation", apply_atmospheric_simulation),
    ("Hair/Cloth Dynamics", apply_hair_cloth_dynamics),
    ("Motion Blur", apply_motion_blur),
    ("3D Environment", apply_3d_environment),
    ("Fractal Zoom", apply_fractal_zoom),
    ("Temporal Filtering", apply_temporal_filtering),
    ("Material Processing", apply_material_processing),
    ("HDR Enhancement", apply_hdr_enhancement)
]

# =======================
# Persona
# =======================
class SlizzAiPersona:
    def __init__(self, persona_style="SlizzAi"):
        self.persona_style = persona_style
    def respond(self, user_input):
        lower = user_input.lower()
        if "hello" in lower or "hi" in lower:
            return f"{self.persona_style} greets you warmly!"
        elif "help" in lower:
            return f"{self.persona_style} can process images and answer questions."
        elif "image" in lower:
            return f"{self.persona_style} is ready to enhance your images!"
        elif "goodbye" in lower or "bye" in lower:
            return f"{self.persona_style} bids you farewell!"
        return f"{self.persona_style} is pondering your question..."

# =======================
# Main Tabs
# =======================
class ChatbotTab(ttk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.persona = SlizzAiPersona()
        self.configure(style='Dark.TFrame')
        self.create_widgets()
    def create_widgets(self):
        lbl = ttk.Label(self, text="SlizzAi Chatbot", style='Accent.TLabel')
        lbl.pack(pady=10)
        self.chat_log = tk.Text(self, wrap=tk.WORD, state='disabled', height=15, bg=DARKER_BG, fg=FG_COLOR, insertbackground=ACCENT, relief=tk.FLAT)
        self.chat_log.pack(fill=tk.BOTH, padx=8, pady=5, expand=True)
        self.chat_entry = ttk.Entry(self, style='Dark.TEntry')
        self.chat_entry.pack(fill=tk.X, padx=8, pady=8)
        self.chat_entry.bind("<Return>", self.process_chat)
        self.image_label = ttk.Label(self, text="No image loaded. Click to load.", style='Dark.TLabel')
        self.image_label.pack(pady=8)
        self.image_label.bind("<Button-1>", self.load_image)
        self.log_chat("Welcome to SlizzAi! How can I assist you today?")
    def log_chat(self, message):
        self.chat_log.config(state='normal')
        self.chat_log.insert(tk.END, message + "\n")
        self.chat_log.config(state='disabled')
        self.chat_log.see(tk.END)
    def load_image(self, event=None):
        file_path = filedialog.askopenfilename(title="Select Image", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")])
        if file_path:
            try:
                self.controller.loaded_image = Image.open(file_path).convert("RGB")
                self.controller.processed_image = self.controller.loaded_image.copy()
                self.display_image(self.controller.loaded_image)
                self.log_chat(f"Image loaded: {file_path}")
            except Exception as e:
                messagebox.showerror("Image Load Error", f"Failed to load image: {e}")
    def display_image(self, img):
        if hasattr(self, "img_panel"):
            self.img_panel.destroy()
        img_disp = img.copy()
        img_disp.thumbnail((400, 400))
        img_tk = ImageTk.PhotoImage(img_disp)
        self.img_panel = ttk.Label(self, image=img_tk, style='Dark.TLabel')
        self.img_panel.image = img_tk
        self.img_panel.pack(pady=10)
        self.img_panel.bind("<Button-1>", self.load_image)
    def improved_chatbot_response(self, user_input):
        lower = user_input.lower()
        # Image processing commands
        for name, func in EFFECTS:
            if name.lower().replace("/", " ").replace("&", "and") in lower:
                if self.controller.processed_image:
                    new_img = func(self.controller.processed_image)
                    self.controller.processed_image = new_img
                    self.display_image(new_img)
                    return f"{name} applied."
                return "No image loaded to apply effect."
        if "flip image" in lower:
            if self.controller.processed_image:
                flip_direction = simpledialog.askstring("Flip Image", "Enter 'h' for horizontal or 'v' for vertical:")
                if flip_direction and flip_direction.lower() == 'h':
                    flipped = self.controller.processed_image.transpose(Image.FLIP_LEFT_RIGHT)
                    self.controller.processed_image = flipped
                    self.display_image(flipped)
                    return "Image flipped horizontally."
                elif flip_direction and flip_direction.lower() == 'v':
                    flipped = self.controller.processed_image.transpose(Image.FLIP_TOP_BOTTOM)
                    self.controller.processed_image = flipped
                    self.display_image(flipped)
                    return "Image flipped vertically."
                else:
                    return "Invalid direction. Enter 'h' or 'v'."
            else:
                return "No image loaded to flip."
        # General responses
        return self.persona.respond(user_input)
    def process_chat(self, event=None):
        user_input = self.chat_entry.get().strip()
        if user_input:
            self.log_chat(f"You: {user_input}")
            response = self.improved_chatbot_response(user_input)
            self.log_chat(f"SlizzAi: {response}")
            self.chat_entry.delete(0, tk.END)

class EffectsTab(ttk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.configure(style='Dark.TFrame')
        self.create_widgets()
    def create_widgets(self):
        lbl = ttk.Label(self, text="Image Effects", style='Accent.TLabel')
        lbl.pack(pady=10)
        btn_frame = ttk.Frame(self, style='Dark.TFrame')
        btn_frame.pack(fill=tk.X, padx=10, pady=10)
        for name, func in EFFECTS:
            btn = ttk.Button(btn_frame, text=name, style='Accent.TButton', command=lambda f=func, n=name: self.apply_effect(f, n))
            btn.pack(side=tk.LEFT, padx=5, pady=3)
    def apply_effect(self, func, name):
        if self.controller.processed_image:
            new_img = func(self.controller.processed_image)
            self.controller.processed_image = new_img
            self.controller.chatbot_tab.display_image(new_img)
            messagebox.showinfo("Effect Applied", f"{name} applied.")
        else:
            messagebox.showwarning("No Image", "No image loaded to apply effect.")

class ModulesTab(ttk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.configure(style='Dark.TFrame')
        self.create_widgets()
    def create_widgets(self):
        lbl = ttk.Label(self, text="Modules (Simulated)", style='Accent.TLabel')
        lbl.pack(pady=10)
        info = ttk.Label(self, text="Dynamic module import is simulated in this version.", style='Dark.TLabel')
        info.pack(pady=10)

class PositionWindowTab(ttk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.configure(style='Dark.TFrame')
        self.create_widgets()
    def create_widgets(self):
        lbl = ttk.Label(self, text="Window Position", style='Accent.TLabel')
        lbl.pack(pady=10)
        btn_frame = ttk.Frame(self, style='Dark.TFrame')
        btn_frame.pack(fill=tk.X, padx=10, pady=10)
        positions = [
            ("Top Left", "+0+0"),
            ("Top Right", lambda: f"+{self.controller.winfo_screenwidth()-900}+0"),
            ("Bottom Left", lambda: f"+0+{self.controller.winfo_screenheight()-650}"),
            ("Bottom Right", lambda: f"+{self.controller.winfo_screenwidth()-900}+{self.controller.winfo_screenheight()-650}")
        ]
        for name, pos in positions:
            btn = ttk.Button(btn_frame, text=name, style='Accent.TButton', command=lambda p=pos: self.set_position(p))
            btn.pack(side=tk.LEFT, padx=5)
    def set_position(self, pos):
        if callable(pos):
            pos = pos()
        self.controller.geometry(pos)

# =======================
# Main Application
# =======================
class MainApplication(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry(WINDOW_GEOMETRY)
        self.configure(bg=DARK_BG)
        self.loaded_image = None
        self.processed_image = None
        self.setup_theme()
        self.create_tabs()
    def setup_theme(self):
        style = ttk.Style(self)
        style.theme_use('clam')
        style.configure('Dark.TFrame', background=DARK_BG)
        style.configure('Dark.TLabel', background=DARK_BG, foreground=FG_COLOR)
        style.configure('Accent.TLabel', background=DARK_BG, foreground=ACCENT, font=('Segoe UI', 12, 'bold'))
        style.configure('Accent.TButton', background=ACCENT, foreground=FG_COLOR, borderwidth=0, focusthickness=2, focuscolor=ACCENT)
        style.map('Accent.TButton', background=[('active', LIGHT_DARK)])
        style.configure('Dark.TEntry', fieldbackground=DARKER_BG, foreground=FG_COLOR, insertcolor=ACCENT)
    def create_tabs(self):
        tab_control = ttk.Notebook(self)
        tab_control.pack(expand=1, fill='both')
        self.chatbot_tab = ChatbotTab(tab_control, self)
        self.effects_tab = EffectsTab(tab_control, self)
        self.modules_tab = ModulesTab(tab_control, self)
        self.position_window_tab = PositionWindowTab(tab_control, self)
        tab_control.add(self.chatbot_tab, text="Chatbot")
        tab_control.add(self.effects_tab, text="Effects")
        tab_control.add(self.modules_tab, text="Modules")
        tab_control.add(self.position_window_tab, text="Window")
        self.chatbot_tab.display_image = self.chatbot_tab.display_image  # For EffectsTab to access

if __name__ == "__main__":
    app = MainApplication()
    app.mainloop()
