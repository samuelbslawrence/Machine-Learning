import tkinter as tk
from tkinter import ttk
import numpy as np
from PIL import Image, ImageDraw
import tensorflow as tf
import os
import cv2

class DigitPredictor:
    def __init__(self):
        # - PATH SETUP
        script_dir = os.path.abspath(os.path.dirname(__file__))
        while os.path.basename(script_dir) != "Machine-Learning":
            parent = os.path.dirname(script_dir)
            if parent == script_dir:
                raise FileNotFoundError(
                    "Could not locate 'Machine-Learning' directory in path tree."
                )
            script_dir = parent

        # Define output directory and model path
        task_dir = os.path.join(script_dir, "Task 5")
        model_path = os.path.join(task_dir, "Task5_Model.h5")

        self.canvas_size = 280
        self.model_input_size = 28
        self.model_path = model_path
        self.model = tf.keras.models.load_model(self.model_path)

        self.root = tk.Tk()
        self.root.title("MNIST Digit Predictor")
        self.root.geometry("400x500")

        self.canvas = tk.Canvas(
            self.root, width=self.canvas_size, height=self.canvas_size, bg="black"
        )
        self.canvas.pack(pady=10)

        self.image = Image.new("L", (self.canvas_size, self.canvas_size), "black")
        self.draw = ImageDraw.Draw(self.image)

        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset_position)
        self.old_x = None
        self.old_y = None

        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=10)

        self.predict_button = tk.Button(
            button_frame,
            text="Predict",
            command=self.predict_digit,
            bg="green",
            fg="white",
            font=("Arial", 12),
        )
        self.predict_button.pack(side=tk.LEFT, padx=5)

        self.clear_button = tk.Button(
            button_frame,
            text="Clear",
            command=self.clear_canvas,
            bg="red",
            fg="white",
            font=("Arial", 12),
        )
        self.clear_button.pack(side=tk.LEFT, padx=5)

        self.result_label = tk.Label(
            self.root, text="Draw a digit and click Predict", font=("Arial", 16)
        )
        self.result_label.pack(pady=10)

        self.confidence_frame = tk.Frame(self.root)
        self.confidence_frame.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)

        self.confidence_bars = {}
        for i in range(10):
            frame = tk.Frame(self.confidence_frame)
            frame.pack(fill=tk.X, pady=2)

            label = tk.Label(frame, text=str(i), width=3)
            label.pack(side=tk.LEFT)

            bar = ttk.Progressbar(frame, length=200, mode="determinate")
            bar.pack(side=tk.LEFT, padx=5)

            conf_label = tk.Label(frame, text="0%", width=5)
            conf_label.pack(side=tk.LEFT)

            self.confidence_bars[i] = (bar, conf_label)

    def reset_position(self, event):
        self.old_x = None
        self.old_y = None

    def paint(self, event):
        if self.old_x and self.old_y:
            self.canvas.create_line(
                self.old_x,
                self.old_y,
                event.x,
                event.y,
                width=20,
                fill="white",
                capstyle=tk.ROUND,
                smooth=tk.TRUE,
            )
            self.draw.line(
                [self.old_x, self.old_y, event.x, event.y], fill="white", width=20
            )

        self.old_x = event.x
        self.old_y = event.y

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), "black")
        self.draw = ImageDraw.Draw(self.image)
        self.old_x = None
        self.old_y = None
        self.result_label.config(text="Draw a digit and click Predict")

        for i in range(10):
            bar, label = self.confidence_bars[i]
            bar["value"] = 0
            label.config(text="0%")

    def preprocess_image(self):
        # Resize to 28x28
        img_resized = self.image.resize((28, 28), Image.Resampling.LANCZOS)

        # Convert to numpy and apply enhanced preprocessing
        img_array = np.array(img_resized).astype("uint8")

        img_array = cv2.GaussianBlur(img_array, (3, 3), 0)
        img_array = cv2.adaptiveThreshold(
            img_array, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        if np.mean(img_array) > 127:
            img_array = 255 - img_array

        img_array = img_array.astype("float32") / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)

        return img_array

    def predict_digit(self):
        processed_img = self.preprocess_image()
        predictions = self.model.predict(processed_img, verbose=0)
        predicted_digit = np.argmax(predictions[0])
        confidence = predictions[0][predicted_digit] * 100

        self.result_label.config(
            text=f"Predicted: {predicted_digit} ({confidence:.1f}% confident)"
        )

        for i in range(10):
            bar, label = self.confidence_bars[i]
            conf = predictions[0][i] * 100
            bar["value"] = conf
            label.config(text=f"{conf:.0f}%")

            if i == predicted_digit:
                label.config(fg="green", font=("Arial", 10, "bold"))
            else:
                label.config(fg="black", font=("Arial", 10, "normal"))

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    print("Starting MNIST Digit Predictor")
    app = DigitPredictor()
    app.run()