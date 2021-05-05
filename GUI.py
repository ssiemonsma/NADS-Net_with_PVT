import tkinter
from tkinter import filedialog
import cv2
import PIL.Image, PIL.ImageTk
import time
import numpy as np

class GUI:
    def __init__(self, window, window_title, video_source):
        size = 384
        self.window = window
        self.window.title(window_title)
        self.video_source = video_source
        self.vid = MyVideoCapture(self.video_source)

        self.canvas = tkinter.Canvas(window, width = 384, height = 216)
        self.canvas.grid(row=0, column=1)

        self.btn_Process=tkinter.Button(window, text="Process", width=10, command=self.value_changer)
        self.btn_Process.grid(row=2, column=0)

        self.btn_norm=tkinter.Button(window, text="Normal", width=10, command=self.normal)
        self.btn_norm.grid(row=3, column=0)

        self.open_button = tkinter.Button(window, text="Open", width=10, command=self.open)
        self.open_button.grid(row=2, column=2)

        self.exit = tkinter.Button(window, text="Exit", width=10, command=self.exit)
        self.exit.grid(row=3, column=2)

        self.fast = tkinter.Button(window, text=">>", width=20, command=self.forward)
        self.fast.grid(row=1, column=1, sticky = "NE")

        self.slow = tkinter.Button(window, text="<<", width=20, command=self.slow)
        self.slow.grid(row=1, column=1, sticky = "NW")

        self.norm = tkinter.Button(window, text=">", width=5, command=self.norm)
        self.norm.grid(row=1, column=1)

        self.box_label = tkinter.Label(window, text = "Frames Per Second: ", width=20)
        self.box_label.grid(row=2, column=1)

        self.fps = 30
        self.box = tkinter.Text(window, width = 3, height = 1)
        self.box.grid(row=3, column=1)
        self.box.insert('insert', str(self.fps))


        self.delay = 30
        self.jump = 0
        self.path = ""
        self.value = 0
        self.update()
        self.window.mainloop()

    def forward (self):
        self.delay = 1

    def slow (self):
        self.delay = 200

    def norm (self):
        self.delay = 30

    def path (self):
        self.path = "inputs/test1.mp4"
        return

    def value_changer(self):
        self.value = 1

    def exit(self):
        self.window.destroy()

    def open(self):
        inputFileName = filedialog.askopenfilename()
        self.window.destroy()
        GUI(tkinter.Tk(), "Tkinter and OpenCV", inputFileName)

    def No_process(self, frame):
        size = 384
        result = cv2.resize(frame,None,fx=size/frame.shape[1], fy=size/frame.shape[1], interpolation = cv2.INTER_CUBIC)
        return result, 30

    def normal(self):
        self.value = 0

    def update(self):
        ret, frame = self.vid.get_frame()
        if ret:
            if self.value == 0:
                out_frame = self.No_process(frame)
            elif self.value == 1:
                out_frame = self.process(frame)
            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(out_frame))
            self.canvas.create_image(0, 0, image = self.photo, anchor = tkinter.NW)
        self.window.after(self.delay, self.update)

class MyVideoCapture:
    def __init__(self, video_source=0):
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)
        else:
            return (ret, None)

    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()
