#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ZetCode Tkinter e-book

This script produces a long-running task of calculating
a large Pi number, while keeping the GUI responsive.
This is an example written for Windows.

Author: Jan Bodnar
Last modified: January 2016
Website: www.zetcode.com
"""

from tkinter import (Tk, BOTH, Text, E, W, S, N, END,
                     NORMAL, DISABLED, StringVar)
from tkinter.ttk import Frame, Label, Button, Progressbar, Entry
from tkinter import scrolledtext

from multiprocessing import Process, Manager, Queue
from queue import Empty
from decimal import Decimal, getcontext

DELAY1 = 80
DELAY2 = 20

# Queue must be global
q = Queue()


class Example(Frame):

    def __init__(self, parent):
        Frame.__init__(self, parent, name="frame")

        self.parent = parent
        self.initUI()

    def initUI(self):

        self.parent.title("Pi computation")
        self.pack(fill=BOTH, expand=True)

        self.grid_columnconfigure(4, weight=1)
        self.grid_rowconfigure(3, weight=1)

        lbl1 = Label(self, text="Digits:")
        lbl1.grid(row=0, column=0, sticky=E, padx=10, pady=10)

        self.ent1 = Entry(self, width=10)
        self.ent1.insert(END, "4000")
        self.ent1.grid(row=0, column=1, sticky=W)

        lbl2 = Label(self, text="Accuracy:")
        lbl2.grid(row=0, column=2, sticky=E, padx=10, pady=10)

        self.ent2 = Entry(self, width=10)
        self.ent2.insert(END, "100")
        self.ent2.grid(row=0, column=3, sticky=W)

        self.startBtn = Button(self, text="Start",
                               command=self.onStart)
        self.startBtn.grid(row=1, column=0, padx=10, pady=5, sticky=W)

        self.pbar = Progressbar(self, mode='indeterminate')
        self.pbar.grid(row=1, column=1, columnspan=3, sticky=W + E)

        self.txt = scrolledtext.ScrolledText(self)
        self.txt.grid(row=2, column=0, rowspan=4, padx=10, pady=5,
                      columnspan=5, sticky=E + W + S + N)

    def onStart(self):

        self.startBtn.config(state=DISABLED)
        self.txt.delete("1.0", END)

        digits = int(self.ent1.get())
        accuracy = int(self.ent2.get())

        self.p1 = Process(target=generatePi, args=(q, digits, accuracy))
        self.p1.start()
        self.pbar.start(DELAY2)
        self.after(DELAY1, self.onGetValue)

    def onGetValue(self):

        if (self.p1.is_alive()):

            self.after(DELAY1, self.onGetValue)
            return
        else:

            try:

                self.txt.insert('end', q.get(0))
                self.txt.insert('end', "\n")
                self.pbar.stop()
                self.startBtn.config(state=NORMAL)

            except Empty:
                print("queue is empty")


# Generate function must be a top-level module funtion
def generatePi(q, digs, acc):
    getcontext().prec = digs

    pi = Decimal(0)
    k = 0
    n = acc

    while k < n:
        pi += (Decimal(1) / (16 ** k)) * ((Decimal(4) / (8 * k + 1)) - \
                                          (Decimal(2) / (8 * k + 4)) - (Decimal(1) / (8 * k + 5)) - \
                                          (Decimal(1) / (8 * k + 6)))
        k += 1

    q.put(pi)


def main():
    root = Tk()
    root.geometry("400x350+300+300")
    app = Example(root)
    root.mainloop()


if __name__ == '__main__':
    main()