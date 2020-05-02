from tkinter import Tk
from tkinter import *
from tkinter import ttk, Label, Entry, Button, E, W, messagebox

def open_msg_box():
    messagebox.showwarning(
        "Event Triggered",
        "Button Clicked"
    )

root = Tk()

# widthxheight+xoffset+yoffset
root.geometry("400x400+300+300")

root.resizable(width=False, height=False)

frame = Frame(root)

style = ttk.Style()

# http://wiki.tcl.tk/37701

style.configure("TButton",
foreground="midnight blue",
font="Times 20 bold italic",
padding=20)

print(ttk.Style().theme_names())

print(style.lookup("TButton", "font"))
print(style.lookup("TButton", "foreground"))
print(style.lookup("TButton", "padding"))

theButton= ttk.Button(frame,
text="Important Button",
command=open_msg_box)

theButton["state"] = 'disabled'
theButton["state"] = 'normal'

theButton.pack()

frame.pack()

# Keep the window open unitl the user hits the 'close' button
root.mainloop()