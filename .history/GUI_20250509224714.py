# gui.py

import tkinter as tk
from tkinter import messagebox
import threading
import processor  # تأكد إن الملف processor.py في نفس المجلد

def start_detection():
    try:
        threading.Thread(target=processor.run_detection).start()
    except Exception as e:
        messagebox.showerror("خطأ", f"حدث خطأ أثناء التشغيل: {e}")

# إنشاء واجهة المستخدم
root = tk.Tk()
root.title("Face & Object Detection")
root.geometry("300x150")

btn = tk.Button(root, text="ابدأ التعرف", font=("Arial", 14), command=start_detection)
btn.pack(pady=40)

root.mainloop()

