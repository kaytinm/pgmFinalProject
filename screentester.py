import tkmacosx as tkm
import tkinter as tk
from tkinter import *
root = Tk()
label = Label(root, text="I am a label widget")
button = Button(root, text="I am a button")
label.pack()
button.pack()
root.mainloop()