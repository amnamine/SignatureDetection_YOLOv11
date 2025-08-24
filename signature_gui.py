import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from ultralytics import YOLO

# Load YOLO model
model = YOLO("signature.pt")  # Make sure best.pt is in the working directory

# Initialize Tkinter
root = tk.Tk()
root.title("Pill Detection")
root.geometry("800x600")

# Variables
img_label = None
img_path = None
display_image = None

def load_image():
    global img_path, display_image, img_label
    img_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if img_path:
        image = Image.open(img_path)
        image = image.resize((600, 400))
        display_image = ImageTk.PhotoImage(image)
        if img_label is None:
            img_label = tk.Label(root, image=display_image)
            img_label.pack(pady=20)
        else:
            img_label.configure(image=display_image)
            img_label.image = display_image

def predict():
    global img_path, display_image, img_label
    if not img_path:
        messagebox.showerror("Error", "Please load an image first!")
        return

    # Run YOLO prediction and save results with built-in boxes
    results = model.predict(img_path, save=True, imgsz=640)  # Saves in './runs/detect/predict'
    pred_img_path = results[0].plot()  # plot() returns annotated image as numpy array

    # Convert to PIL for Tkinter
    img_pil = Image.fromarray(pred_img_path)
    img_pil = img_pil.resize((600, 400))
    display_image = ImageTk.PhotoImage(img_pil)
    img_label.configure(image=display_image)
    img_label.image = display_image

def reset():
    global img_label, img_path, display_image
    if img_label:
        img_label.configure(image="")
        img_label.image = None
    img_path = None
    display_image = None

# Buttons
btn_load = tk.Button(root, text="Load Image", command=load_image, width=15, bg="lightblue")
btn_load.pack(side=tk.LEFT, padx=20, pady=10)

btn_predict = tk.Button(root, text="Predict", command=predict, width=15, bg="lightgreen")
btn_predict.pack(side=tk.LEFT, padx=20, pady=10)

btn_reset = tk.Button(root, text="Reset", command=reset, width=15, bg="lightcoral")
btn_reset.pack(side=tk.LEFT, padx=20, pady=10)

root.mainloop()
