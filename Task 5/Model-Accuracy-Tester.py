import tkinter as tk
from tkinter import ttk
import numpy as np
from PIL import Image, ImageDraw
import tensorflow as tf
import os

class DigitPredictor:
    def __init__(self):
        # Load the MNIST model
        self.model_path = 'C:/Users/spenc/Desktop/New folder (2)/Machine-Learning/Task 5/mnist_classifier.h5'
        self.model = tf.keras.models.load_model(self.model_path)
        print("MNIST model loaded successfully!")
        
        # Canvas settings
        self.canvas_size = 280
        self.model_input_size = 28
        
        # Create the main window
        self.root = tk.Tk()
        self.root.title("MNIST Digit Predictor")
        self.root.geometry("400x500")
        
        # Create the drawing canvas
        self.canvas = tk.Canvas(self.root, width=self.canvas_size, height=self.canvas_size, bg='black')
        self.canvas.pack(pady=10)
        
        # Create image for drawing
        self.image = Image.new('L', (self.canvas_size, self.canvas_size), 'black')
        self.draw = ImageDraw.Draw(self.image)
        
        # Bind mouse events
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset_position)
        self.old_x = None
        self.old_y = None
        
        # Create buttons
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=10)
        
        self.predict_button = tk.Button(button_frame, text="Predict", command=self.predict_digit, 
                                      bg='green', fg='white', font=('Arial', 12))
        self.predict_button.pack(side=tk.LEFT, padx=5)
        
        self.clear_button = tk.Button(button_frame, text="Clear", command=self.clear_canvas,
                                    bg='red', fg='white', font=('Arial', 12))
        self.clear_button.pack(side=tk.LEFT, padx=5)
        
        # Create result label
        self.result_label = tk.Label(self.root, text="Draw a digit and click Predict", 
                                   font=('Arial', 16))
        self.result_label.pack(pady=10)
        
        # Create confidence bars
        self.confidence_frame = tk.Frame(self.root)
        self.confidence_frame.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)
        
        self.confidence_bars = {}
        for i in range(10):
            frame = tk.Frame(self.confidence_frame)
            frame.pack(fill=tk.X, pady=2)
            
            label = tk.Label(frame, text=str(i), width=3)
            label.pack(side=tk.LEFT)
            
            bar = ttk.Progressbar(frame, length=200, mode='determinate')
            bar.pack(side=tk.LEFT, padx=5)
            
            conf_label = tk.Label(frame, text="0%", width=5)
            conf_label.pack(side=tk.LEFT)
            
            self.confidence_bars[i] = (bar, conf_label)
    
    def reset_position(self, event):
        self.old_x = None
        self.old_y = None
    
    def paint(self, event):
        if self.old_x and self.old_y:
            # Draw on canvas with thicker line
            self.canvas.create_line(self.old_x, self.old_y, event.x, event.y,
                                  width=20, fill='white', capstyle=tk.ROUND, smooth=tk.TRUE)
            # Draw on PIL image
            self.draw.line([self.old_x, self.old_y, event.x, event.y], 
                          fill='white', width=20)
        
        self.old_x = event.x
        self.old_y = event.y
    
    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new('L', (self.canvas_size, self.canvas_size), 'black')
        self.draw = ImageDraw.Draw(self.image)
        self.old_x = None
        self.old_y = None
        self.result_label.config(text="Draw a digit and click Predict")
        
        # Reset confidence bars
        for i in range(10):
            bar, label = self.confidence_bars[i]
            bar['value'] = 0
            label.config(text="0%")
    
    def preprocess_image(self):
        # Resize to model input size
        img_resized = self.image.resize((self.model_input_size, self.model_input_size), 
                                       Image.Resampling.LANCZOS)
        
        # Convert to numpy array and normalize
        img_array = np.array(img_resized).astype('float32') / 255.0
        
        # Add batch and channel dimensions
        img_array = img_array.reshape(1, self.model_input_size, self.model_input_size, 1)
        
        return img_array
    
    def predict_digit(self):
        # Preprocess the image
        processed_img = self.preprocess_image()
        
        # Make prediction
        predictions = self.model.predict(processed_img, verbose=0)
        predicted_digit = np.argmax(predictions[0])
        confidence = predictions[0][predicted_digit] * 100
        
        # Update result label
        self.result_label.config(text=f"Predicted: {predicted_digit} ({confidence:.1f}% confident)")
        
        # Update confidence bars
        for i in range(10):
            bar, label = self.confidence_bars[i]
            conf = predictions[0][i] * 100
            bar['value'] = conf
            label.config(text=f"{conf:.0f}%")
            
            # Highlight the predicted digit
            if i == predicted_digit:
                label.config(fg='green', font=('Arial', 10, 'bold'))
            else:
                label.config(fg='black', font=('Arial', 10, 'normal'))
    
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    print("Starting MNIST Digit Predictor...")
    app = DigitPredictor()
    app.run()