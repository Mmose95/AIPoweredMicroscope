from tkinter import Tk, filedialog

def select_training_folder():
    root = Tk()
    root.withdraw()  # Hide the root window
    folder_path = filedialog.askdirectory(title="Select Training Data Folder")
    root.destroy()
    return folder_path