import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from functions import *

from pandas import read_csv
from tensorflow import keras
class App:
    def __init__(self, root):
        root.title("COVID Prediction")
        width = 600
        height = 500

        screenwidth = root.winfo_screenwidth()
        screenheight = root.winfo_screenheight()

        alignstr = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2)
        root.geometry(alignstr)
        root.resizable(width=False, height=False)

        self.button = tk.Button(root, bg="#efefef", fg="#000000", justify="center", text="Generisi!")
        self.button.place(x=350, y=270, width=89, height=33)
        self.button["command"] = self.generate

        self.entrybox = tk.Entry(root, fg="#333333", justify="center")
        self.entrybox.place(x=330, y=180, width=136, height=49)

        self.label = tk.Label(root, fg="#333333", justify="center", text="Unesi broj dana")
        self.label.place(x=340, y=140, width=120, height=25)

        cols = ('Day', 'Hospitalized')
        self.listBox = ttk.Treeview(root, columns=cols, show='headings', height=20)
        verscrlbar = ttk.Scrollbar(root, orient="vertical", command=self.listBox.yview)
        self.listBox.configure(xscrollcommand=verscrlbar.set)

        self.listBox.column("Day", width=80, anchor='center')
        self.listBox.column("Hospitalized", width=200, anchor='center')

        for col in cols:
            self.listBox.heading(col, text=col)
            self.listBox.grid(row=1, column=0, columnspan=2)
            self.listBox.place(x=10, y=20)

    def generate(self):
        day = self.entrybox.get()

        # ---------Svaki put novi listbox
        self.listBox.destroy()
        cols = ('Day', 'Hospitalized')
        self.listBox = ttk.Treeview(root, columns=cols, show='headings', height=20)
        verscrlbar = ttk.Scrollbar(root, orient="vertical", command=self.listBox.yview)
        self.listBox.configure(xscrollcommand=verscrlbar.set)

        self.listBox.column("Day", width=80, anchor='center')
        self.listBox.column("Hospitalized", width=200, anchor='center')

        for col in cols:
            self.listBox.heading(col, text=col)
            self.listBox.grid(row=1, column=0, columnspan=2)
            self.listBox.place(x=10, y=20)
        # -------------------------------------------------
        try:
            day = int(day)
            #prediction = predict(day) #model vrsi novu predikciju za svaki unos
            prediction = predict_from_model(day)#model predvidja iz postojeceg modela
            print(type(prediction))
        except:
            messagebox.showerror(title="Greska!",
                                 message="Unesite ispravan broj dana za koji zelite da generiste predikciju.")
            prediction = 0

        count = 1
        for i in prediction:
            if count <= day:
                self.listBox.insert("", "end", values=(count, int(i[0].item())))
                count +=1

if __name__ == "__main__":

    root = tk.Tk()
    app = App(root)
    root.mainloop()
