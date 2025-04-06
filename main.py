import tkinter as tk
from tkinter import messagebox
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
import os

# ✅ Updated realistic dataset
data = {
    'Hours_Studied': [1, 2, 3, 4, 5, 6, 1.5, 2.5, 3.5, 4.5, 6.5, 1, 5.5, 3, 2],
    'Attendance':    [45, 50, 60, 65, 85, 95, 40, 55, 70, 80, 92, 35, 88, 60, 50],
    'Previous_Score':[40, 50, 60, 65, 75, 90, 35, 45, 70, 75, 94, 30, 85, 62, 52],
    'Result':        [0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0]
}
df = pd.DataFrame(data)

# Prepare data
X = df[['Hours_Studied', 'Attendance', 'Previous_Score']]
y = df['Result']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Accuracy
accuracy = model.score(X_test, y_test)

# GUI setup
def predict_result():
    try:
        hours = float(entry_hours.get())
        attendance = float(entry_attendance.get())
        score = float(entry_score.get())
        input_data = np.array([[hours, attendance, score]])
        prediction = model.predict(input_data)
        result_text = "Pass ✅" if prediction[0] == 1 else "Fail ❌"
        messagebox.showinfo("Prediction Result", f"The student will: {result_text}")
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numbers!")

def reset_fields():
    entry_hours.delete(0, tk.END)
    entry_attendance.delete(0, tk.END)
    entry_score.delete(0, tk.END)

# Create main window
root = tk.Tk()
root.title("Student Performance Predictor")
root.state('zoomed')  # Fullscreen
root.configure(bg="#f0f0f0")

# Main frame
frame = tk.Frame(root, bg="#f0f0f0")
frame.pack(pady=20)

# Add student icon image (ensure student.png is in the same directory)
icon_path = os.path.join(os.path.dirname(__file__), 'student.png')
if os.path.exists(icon_path):
    student_img = Image.open(icon_path).resize((80, 80))
    student_photo = ImageTk.PhotoImage(student_img)
    icon_label = tk.Label(frame, image=student_photo, bg="#f0f0f0")
    icon_label.grid(row=0, column=0, columnspan=2, pady=(10, 0))

# Title with emojis
title_label = tk.Label(frame, text="🎓 Student Performance Predictor 🎯", font=("Arial", 26, "bold"), bg="#f0f0f0", fg="black")
title_label.grid(row=1, column=0, columnspan=2, pady=(5, 20))

# Inputs with emojis
tk.Label(frame, text="⏱️ Hours Studied:", font=("Arial", 16), bg="#f0f0f0").grid(row=2, column=0, sticky='e', pady=10)
entry_hours = tk.Entry(frame, font=("Arial", 16), width=10)
entry_hours.grid(row=2, column=1, pady=10, sticky='w')

tk.Label(frame, text="🗓️ Attendance (%):", font=("Arial", 16), bg="#f0f0f0").grid(row=3, column=0, sticky='e', pady=10)
entry_attendance = tk.Entry(frame, font=("Arial", 16), width=10)
entry_attendance.grid(row=3, column=1, pady=10, sticky='w')

tk.Label(frame, text="📋 Previous Score:", font=("Arial", 16), bg="#f0f0f0").grid(row=4, column=0, sticky='e', pady=10)
entry_score = tk.Entry(frame, font=("Arial", 16), width=10)
entry_score.grid(row=4, column=1, pady=10, sticky='w')

# Buttons
btn_frame = tk.Frame(frame, bg="#f0f0f0")
btn_frame.grid(row=5, column=0, columnspan=2, pady=20)

predict_btn = tk.Button(btn_frame, text="🔍 Predict Result", command=predict_result, font=("Arial", 14), bg="green", fg="white", padx=20)
predict_btn.pack(side='left', padx=10)

reset_btn = tk.Button(btn_frame, text="🔄 Reset", command=reset_fields, font=("Arial", 14), bg="orange", fg="white", padx=20)
reset_btn.pack(side='left', padx=10)

# ✅ Corrected Accuracy Graph
fig = plt.Figure(figsize=(4, 3), dpi=100)
ax = fig.add_subplot(111)
ax.bar(['Accuracy'], [accuracy], color='skyblue')
ax.set_ylim([0, 1])
ax.set_title("Model Accuracy")
ax.set_ylabel("Accuracy")
ax.text(0, accuracy / 2, f"{accuracy * 100:.2f}%", ha='center', fontsize=14, weight='bold')

canvas = FigureCanvasTkAgg(fig, master=frame)
canvas.get_tk_widget().grid(row=2, column=2, rowspan=4, padx=40)

root.mainloop()