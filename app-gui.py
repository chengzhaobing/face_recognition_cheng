from Detector import main_app
from create_classifier import train_classifer
from create_dataset import start_capture
import tkinter as tk
from tkinter import font as tkfont
from tkinter import messagebox, PhotoImage

"""
designer ：  cheng
title ： 基于opencv的监控人脸识别系统的实现
time  ： 2022.10.1

"""
names = set()


class MainUI(tk.Tk):

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        global names
        with open("nameslist.txt", "r") as f:
            x = f.read()
            z = x.rstrip().split(" ")
            for i in z:
                names.add(i)
        self.title_font = tkfont.Font(family='Helvetica', size=16, weight="bold")
        self.title("监狱身份认证系统")
        self.resizable(False, False)
        self.geometry("820x320")
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.active_name = None
        container = tk.Frame(self)
        container.grid(sticky="nsew")
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)
        self.frames = {}
        for F in (StartPage, PageOne, PageTwo, PageThree, PageFour):
            page_name = F.__name__
            frame = F(parent=container, controller=self)
            self.frames[page_name] = frame
            frame.grid(row=0, column=0, sticky="nsew")
        self.show_frame("StartPage")

    def show_frame(self, page_name):
        frame = self.frames[page_name]
        frame.tkraise()

    def on_closing(self):

        if messagebox.askokcancel("Quit", "Are you sure?"):
            global names
            f = open("nameslist.txt", 'a+')
            for i in names:
                f.write(i + " ")
            self.destroy()


class StartPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        render = PhotoImage(file='homepage.png')
        img = tk.Label(self, image=render)
        img.image = render
        img.grid(row=0, column=1, rowspan=4, sticky="nsew")
        label = tk.Label(self, text="        欢迎来到监狱身份认证识别系统！        ", font=self.controller.title_font, fg="#263942")
        label.grid(row=0, sticky="ew")
        button1 = tk.Button(self, text="  添加用户  ", fg="#ffffff", bg="#263942",
                            command=lambda: self.controller.show_frame("PageOne"))
        button2 = tk.Button(self, text="  身份认证  ", fg="#ffffff", bg="#263942",
                            command=lambda: self.controller.show_frame("PageTwo"))
        button3 = tk.Button(self, text="Quit", fg="#263942", bg="#ffffff", command=self.on_closing)
        button1.grid(row=1, column=0, ipady=3, ipadx=7)
        button2.grid(row=2, column=0, ipady=3, ipadx=2)
        button3.grid(row=3, column=0, ipady=3, ipadx=32)

    def on_closing(self):
        if messagebox.askokcancel("提示", "您确定要退出系统吗?"):
            global names
            with open("nameslist.txt", 'w') as f:
                for i in names:
                    f.write(i + " ")
            self.controller.destroy()


class PageOne(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        render2 = PhotoImage(file='detection.png')
        img = tk.Label(self, image=render2)
        img.image = render2
        img.grid(row=0, column=2, rowspan=4, sticky="nsew")
        tk.Label(self, text="    请输入您的姓名：    ", fg="#263942", font='Helvetica 12 bold').grid(row=0, column=0, pady=60,
                                                                                           padx=10)
        self.user_name = tk.Entry(self, borderwidth=3, bg="lightgrey", font='Helvetica 11')
        # show =''显示输入字符形式，为了监狱安全性，这里加密 因为这里是监狱，加密姓名为了保证每个用户识别对应的人脸模型，保证数据信息可靠度，如果不加密将可以识别所有人脸模型
        self.user_name.grid(row=0, column=1, pady=10, padx=10)
        self.buttoncanc = tk.Button(self, text="Cancel", bg="#ffffff", fg="#263942",
                                    command=lambda: controller.show_frame("StartPage"))
        self.buttonext = tk.Button(self, text="Next", fg="#ffffff", bg="#263942", command=self.start_training)
        self.buttoncanc.grid(row=1, column=0, pady=10, ipadx=5, ipady=4)
        self.buttonext.grid(row=1, column=1, pady=10, ipadx=10, ipady=4)

    def start_training(self):
        global names
        if self.user_name.get() == "None":
            messagebox.showerror("错误", "姓名ID不能为'None'")
            return
        elif self.user_name.get() in names:
            messagebox.showerror("警告", "认证id已存在!")
            return
        elif len(self.user_name.get()) == 0:
            messagebox.showerror("警告", "监狱认证id不能为空!")
            return
        name = self.user_name.get()
        names.add(name)
        self.controller.active_name = name
        self.controller.frames["PageTwo"].refresh_names()
        self.controller.show_frame("PageThree")


class PageTwo(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        global names
        self.controller = controller
        render = PhotoImage(file='homepagepic.png')
        img = tk.Label(self, image=render)
        img.image = render
        img.grid(row=0, column=4, rowspan=10, sticky="nsew")
        tk.Label(self, text="请选择您的姓名：", fg="#263942", font='Helvetica 12 bold').grid(row=4, column=1, padx=10,
                                                                                        pady=10)
        self.buttoncanc = tk.Button(self, text="Cancel", command=lambda: controller.show_frame("StartPage"),
                                    bg="#ffffff", fg="#263942")
        self.menuvar = tk.StringVar(self)
        self.dropdown = tk.OptionMenu(self, self.menuvar, *names)
        self.dropdown.config(bg="lightgrey")
        self.dropdown["menu"].config(bg="lightgrey")
        self.buttonext = tk.Button(self, text="Next", command=self.nextfoo, fg="#ffffff", bg="#263942")
        self.dropdown.grid(row=4, column=2, ipadx=4, padx=10, pady=10)
        self.buttoncanc.grid(row=8, ipadx=2, ipady=4, column=1, pady=10)
        self.buttonext.grid(row=8, ipadx=32, ipady=4, column=2, pady=10)

    def nextfoo(self):
        if self.menuvar.get() == "None":
            messagebox.showerror("ERROR", "Name cannot be 'None'")
            return
        self.controller.active_name = self.menuvar.get()
        self.controller.show_frame("PageFour")

    def refresh_names(self):
        global names
        self.menuvar.set('')
        self.dropdown['menu'].delete(0, 'end')
        for name in names:
            self.dropdown['menu'].add_command(label=name, command=tk._setit(self.menuvar, name))


class PageThree(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        self.numimglabel = tk.Label(self, text=" 捕获的人脸数据图像 = 0", font='Helvetica 12 bold', fg="#263942")
        self.numimglabel.grid(row=0, column=0, columnspan=2, sticky="ew", pady=10)
        self.capturebutton = tk.Button(self, text="Capture Data Set", fg="#ffffff", bg="#263942", command=self.capimg)
        self.trainbutton = tk.Button(self, text="Train The Model", fg="#ffffff", bg="#263942", command=self.trainmodel)
        self.capturebutton.grid(row=1, column=0, ipadx=5, ipady=4, padx=10, pady=20)
        self.trainbutton.grid(row=1, column=1, ipadx=5, ipady=4, padx=10, pady=20)

    def capimg(self):
        self.numimglabel.config(text=str("Captured Images = 0 "))
        messagebox.showinfo("注意！", "本监狱将批量采集您的人脸数据集！")
        x = start_capture(self.controller.active_name)
        self.controller.num_of_images = x
        self.numimglabel.config(text=str(" 捕获的人脸数据图像 = " + str(x)))

    def trainmodel(self):
        if self.controller.num_of_images < 300:
            messagebox.showerror("错误", "数据不足, 至少录入300张数据照片！")
            return
        train_classifer(self.controller.active_name)
        messagebox.showinfo("成功", " 模型已训练完成! ")
        self.controller.show_frame("PageFour")


class PageFour(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        render = PhotoImage(file='recognition.png')
        img = tk.Label(self, image=render)
        img.image = render
        img.grid(row=0, column=2, rowspan=10, sticky="nsew")
        label = tk.Label(self, text="Face Recognition", font='Helvetica 16 bold')
        label.grid(row=0, column=1, sticky="ew")
        button1 = tk.Button(self, text="    Face Recognition    ", command=self.openwebcam, fg="#ffffff", bg="#263942")
        button2 = tk.Button(self, text="Go to Home Page", command=lambda: self.controller.show_frame("StartPage"),
                            bg="#ffffff", fg="#263942")

        button1.grid(row=3, column=0, sticky="ew", ipadx=5, ipady=4, padx=10, pady=10)
        button2.grid(row=3, column=1, sticky="ew", ipadx=5, ipady=4, padx=10, pady=10)

    def openwebcam(self):
        main_app(self.controller.active_name)


app = MainUI()
app.iconphoto(False, tk.PhotoImage(file='icon.ico'))
app.mainloop()
