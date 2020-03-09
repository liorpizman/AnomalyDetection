import tkinter as tk


class AnimatedGif(tk.Label):
    """
    Class which shows an animated gif using tkinter
    """

    def __init__(self, root, file_path, frames_delay=0.03):

        tk.Label.__init__(self, root)
        self.root = root
        self.frames_delay = frames_delay
        self.file_path = file_path
        self.stop = False
        self.count_frames = 0

    def stop(self):
        self.stop = True

    def reset_frames_count(self):
        self.count_frames = 0

    def display_frame(self):
        self.animation = tk.PhotoImage(file=self.file_path, format='gif -index {}'.format(self.count_frames))
        self.configure(image=self.animation)
        self.count_frames += 1

    # Looping through the frames
    def start(self):
        try:
            self.display_frame()
        except tk.TclError:
            self.reset_frames_count()
        if not self.stop:
            self.root.after(int(self.frames_delay * 1000), self.start)
