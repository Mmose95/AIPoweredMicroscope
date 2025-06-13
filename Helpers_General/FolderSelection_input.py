from tkinter import Tk, filedialog

def select_training_folder(disp_text):
    print(disp_text)
    root = Tk()
    root.withdraw()  # Hide the root window
    folder_path = filedialog.askdirectory(title="Select Training Data Folder")
    root.destroy()
    return folder_path

def select_encoder_file():
    print("Please select the encoder checkpoint file (.pt or .pth) — window might be hiding in the background")
    root = Tk()
    root.withdraw()  # Hide the root window

    encoder_file = filedialog.askopenfilename(
        title="Select pretrained encoder file",
        filetypes=[("PyTorch checkpoint", "*.pt *.pth"), ("All files", "*.*")]
    )

    root.destroy()
    return encoder_file
