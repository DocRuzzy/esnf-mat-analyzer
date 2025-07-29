import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
from pathlib import Path
from esnf_mat_analyzer.main import create_config, setup_dependencies

class MainWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Nanofiber Analyzer")
        self.geometry("1200x800")

        self.selected_files = []
        self.current_image = None
        self.rect = None
        self.start_x = None
        self.start_y = None

        self.create_widgets()

    def create_widgets(self):
        self.main_frame = ttk.Frame(self)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left panel for file selection and controls
        self.left_panel = ttk.Frame(self.main_frame)
        self.left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=5)

        # File selection
        self.file_frame = ttk.LabelFrame(self.left_panel, text="File Selection")
        self.file_frame.pack(fill=tk.X, pady=5)

        self.select_button = ttk.Button(
            self.file_frame, text="Select Files", command=self.select_files
        )
        self.select_button.pack(side=tk.TOP, padx=5, pady=5)

        self.file_listbox = tk.Listbox(self.file_frame, selectmode=tk.SINGLE)
        self.file_listbox.pack(side=tk.TOP, fill=tk.X, expand=True, padx=5, pady=5)
        self.file_listbox.bind("<<ListboxSelect>>", self.on_file_select)

        # Analysis controls
        self.analysis_frame = ttk.LabelFrame(self.left_panel, text="Analysis")
        self.analysis_frame.pack(fill=tk.X, pady=5)

        self.analyze_button = ttk.Button(
            self.analysis_frame, text="Analyze", command=self.analyze
        )
        self.analyze_button.pack(padx=5, pady=5)

        # Right panel for image display
        self.image_frame = ttk.LabelFrame(self.main_frame, text="Image Preview")
        self.image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        self.canvas = tk.Canvas(self.image_frame, bg="gray")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)

    def select_files(self):
        files = filedialog.askopenfilenames(
            title="Select Image Files",
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.gif *.tiff")],
        )
        if files:
            self.selected_files = files
            self.update_file_list()

    def update_file_list(self):
        self.file_listbox.delete(0, tk.END)
        for file in self.selected_files:
            self.file_listbox.insert(tk.END, file)

    def on_file_select(self, event):
        selection = event.widget.curselection()
        if selection:
            index = selection[0]
            filepath = self.selected_files[index]
            self.load_image(filepath)

    def load_image(self, filepath):
        try:
            image = Image.open(filepath)
            self.current_image = image
            self.display_image(image)
        except Exception as e:
            print(f"Error loading image: {e}")

    def display_image(self, image):
        self.canvas.delete("all")
        self.tk_image = ImageTk.PhotoImage(image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
        self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))

    def on_button_press(self, event):
        self.start_x = event.x
        self.start_y = event.y
        if self.rect:
            self.canvas.delete(self.rect)
        self.rect = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline="red")

    def on_mouse_drag(self, event):
        cur_x, cur_y = (event.x, event.y)
        self.canvas.coords(self.rect, self.start_x, self.start_y, cur_x, cur_y)

    def on_button_release(self, event):
        pass

    def analyze(self):
        if not self.current_image or not self.rect:
            print("No image or ROI selected")
            return

        # Get ROI from canvas
        x1, y1, x2, y2 = self.canvas.coords(self.rect)
        roi = (int(x1), int(y1), int(x2), int(y2))

        # Get selected file
        selection = self.file_listbox.curselection()
        if not selection:
            print("No file selected")
            return
        index = selection[0]
        filepath = self.selected_files[index]

        # Create config and analyzer
        config = create_config()
        analyzer = setup_dependencies(config)

        # Run analysis
        try:
            result = analyzer.process_image(Path(filepath), roi=roi)
            self.display_results(result)
        except Exception as e:
            print(f"Error during analysis: {e}")

    def display_results(self, result):
        # Create a new window to display results
        results_window = tk.Toplevel(self)
        results_window.title("Analysis Results")
        results_window.geometry("600x400")

        text = tk.Text(results_window, wrap=tk.WORD)
        text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Format and insert results
        formatted_result = f"Image: {result.image_path}\n"
        formatted_result += f"Processing Time: {result.processing_time:.2f}s\n"
        if result.spatial_scale_pixels_per_mm:
            formatted_result += f"Spatial Scale: {result.spatial_scale_pixels_per_mm:.2f} pixels/mm\n"
        formatted_result += "\nMetrics:\n"
        for name, value in result.metrics.items():
            formatted_result += f"  {name}: {value:.4f}\n"
        text.insert(tk.END, formatted_result)
        text.config(state=tk.DISABLED)


if __name__ == "__main__":
    app = MainWindow()
    app.mainloop()
