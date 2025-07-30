import tkinter as tk
from tkinter import ttk, filedialog, simpledialog
from PIL import Image, ImageTk
from pathlib import Path
import math
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from esnf_mat_analyzer.main import create_config, setup_dependencies
from esnf_mat_analyzer.visualization.visualization import Visualizer
from esnf_mat_analyzer.core.data_types import VisualizationConfig

class MainWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Nanofiber Analyzer")
        self.geometry("1200x800")

        self.selected_files = []
        self.current_image = None
        self.displayed_image = None
        self.tk_image = None
        self.rect = None
        self.oval = None
        self.start_x = None
        self.start_y = None
        self.drawing_mode = "roi"
        self.spatial_scale = None
        self.last_analysis_result = None
        
        # Image display properties
        self.scale_factor = 1.0
        self.min_scale = 0.1
        self.max_scale = 5.0
        self.canvas_width = 800
        self.canvas_height = 600
        
        # Pan properties
        self.pan_start_x = 0
        self.pan_start_y = 0
        self.is_panning = False

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

        self.set_scale_button = ttk.Button(
            self.analysis_frame, text="Set Scale Manually", command=self.set_scale_manually
        )
        self.set_scale_button.pack(padx=5, pady=5)

        # Image controls
        self.image_controls_frame = ttk.LabelFrame(self.left_panel, text="Image Controls")
        self.image_controls_frame.pack(fill=tk.X, pady=5)

        # Zoom controls
        zoom_frame = ttk.Frame(self.image_controls_frame)
        zoom_frame.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Label(zoom_frame, text="Zoom:").pack(side=tk.LEFT)
        self.zoom_in_button = ttk.Button(zoom_frame, text="+", width=3, command=self.zoom_in)
        self.zoom_in_button.pack(side=tk.LEFT, padx=2)
        
        self.zoom_out_button = ttk.Button(zoom_frame, text="-", width=3, command=self.zoom_out)
        self.zoom_out_button.pack(side=tk.LEFT, padx=2)
        
        self.fit_button = ttk.Button(zoom_frame, text="Fit", command=self.fit_to_window)
        self.fit_button.pack(side=tk.LEFT, padx=2)
        
        self.reset_button = ttk.Button(zoom_frame, text="100%", command=self.reset_zoom)
        self.reset_button.pack(side=tk.LEFT, padx=2)

        # Scale display
        self.scale_label = ttk.Label(self.image_controls_frame, text="Scale: 100%")
        self.scale_label.pack(pady=2)

        self.insert_scale_bar_button = ttk.Button(
            self.image_controls_frame, text="Insert Scale Bar", command=self.insert_scale_bar
        )
        self.insert_scale_bar_button.pack(pady=5)

        self.show_heatmap_button = ttk.Button(
            self.analysis_frame, text="Show Heatmap", command=self.show_heatmap
        )
        self.show_heatmap_button.pack(padx=5, pady=5)

        self.export_image_button = ttk.Button(
            self.analysis_frame, text="Export Image", command=self.export_image
        )
        self.export_image_button.pack(padx=5, pady=5)

        # Right panel for image display
        self.image_frame = ttk.LabelFrame(self.main_frame, text="Image Preview")
        self.image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        # Create scrollable canvas
        self.canvas_frame = ttk.Frame(self.image_frame)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(self.canvas_frame, bg="gray")
        
        # Add scrollbars
        self.v_scrollbar = ttk.Scrollbar(self.canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.h_scrollbar = ttk.Scrollbar(self.canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=self.v_scrollbar.set, xscrollcommand=self.h_scrollbar.set)
        
        # Pack scrollbars and canvas
        self.v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Bind events
        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)
        self.canvas.bind("<ButtonPress-3>", self.start_pan)  # Right click to pan
        self.canvas.bind("<B3-Motion>", self.do_pan)
        self.canvas.bind("<ButtonRelease-3>", self.end_pan)
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)  # Mouse wheel zoom
        self.canvas.bind("<Configure>", self.on_canvas_configure)

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
            self.fit_to_window()
        except Exception as e:
            print(f"Error loading image: {e}")

    def fit_to_window(self):
        if not self.current_image:
            return
            
        # Get canvas dimensions
        self.canvas.update_idletasks()
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            # Canvas not ready yet, try again later
            self.canvas.after(100, self.fit_to_window)
            return
        
        # Calculate scale to fit image in canvas
        img_width, img_height = self.current_image.size
        scale_x = (canvas_width - 20) / img_width  # Leave some margin
        scale_y = (canvas_height - 20) / img_height
        
        self.scale_factor = min(scale_x, scale_y, 1.0)  # Don't scale up initially
        self.scale_factor = max(self.scale_factor, self.min_scale)
        
        self.update_image_display()

    def reset_zoom(self):
        if not self.current_image:
            return
        self.scale_factor = 1.0
        self.update_image_display()

    def zoom_in(self):
        if not self.current_image:
            return
        self.scale_factor = min(self.scale_factor * 1.2, self.max_scale)
        self.update_image_display()

    def zoom_out(self):
        if not self.current_image:
            return
        self.scale_factor = max(self.scale_factor / 1.2, self.min_scale)
        self.update_image_display()

    def on_mousewheel(self, event):
        if not self.current_image:
            return
            
        # Zoom in/out with mouse wheel
        if event.delta > 0:
            self.zoom_in()
        else:
            self.zoom_out()

    def on_canvas_configure(self, event):
        # Update scroll region when canvas is resized
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def update_image_display(self):
        if not self.current_image:
            return
            
        # Clear canvas
        self.canvas.delete("all")
        
        # Calculate new image size
        img_width, img_height = self.current_image.size
        new_width = int(img_width * self.scale_factor)
        new_height = int(img_height * self.scale_factor)
        
        # Resize image
        if new_width > 0 and new_height > 0:
            self.displayed_image = self.current_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            self.tk_image = ImageTk.PhotoImage(self.displayed_image)
            
            # Display image
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image, tags="image")
            
            # Update scroll region
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))
            
            # Update scale label
            self.scale_label.config(text=f"Scale: {self.scale_factor*100:.0f}%")

        if hasattr(self, 'scale_bar_length_mm'):
            self.draw_scale_bar()

    def start_pan(self, event):
        self.is_panning = True
        self.pan_start_x = event.x
        self.pan_start_y = event.y

    def do_pan(self, event):
        if not self.is_panning:
            return
            
        # Calculate how much to pan
        dx = event.x - self.pan_start_x
        dy = event.y - self.pan_start_y
        
        # Pan the canvas
        self.canvas.scan_dragto(event.x, event.y, gain=1)
        
        self.pan_start_x = event.x
        self.pan_start_y = event.y

    def end_pan(self, event):
        self.is_panning = False

    def on_button_press(self, event):
        if self.is_panning:
            return
            
        # Convert canvas coordinates to actual coordinates
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        
        self.start_x = canvas_x
        self.start_y = canvas_y
        
        if self.drawing_mode == "roi":
            if self.rect:
                self.canvas.delete(self.rect)
            self.rect = self.canvas.create_rectangle(
                self.start_x, self.start_y, self.start_x, self.start_y,
                outline="red", width=2, tags="roi"
            )
        elif self.drawing_mode == "scale":
            if self.oval:
                self.canvas.delete(self.oval)
            self.oval = self.canvas.create_oval(
                self.start_x, self.start_y, self.start_x, self.start_y,
                outline="blue", width=2, tags="scale"
            )

    def on_mouse_drag(self, event):
        if self.is_panning or not self.start_x or not self.start_y:
            return
            
        # Convert canvas coordinates to actual coordinates
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        
        if self.drawing_mode == "roi":
            self.canvas.coords(self.rect, self.start_x, self.start_y, canvas_x, canvas_y)
        elif self.drawing_mode == "scale":
            self.canvas.coords(self.oval, self.start_x, self.start_y, canvas_x, canvas_y)

    def on_button_release(self, event):
        if self.is_panning:
            return

        if self.drawing_mode == "scale":
            self.calculate_scale_from_oval()

        self.drawing_mode = "roi" # Reset drawing mode

    def insert_scale_bar(self):
        if not self.spatial_scale:
            print("Please set the scale first.")
            return

        length_mm = simpledialog.askfloat("Scale Bar Length", "Enter the length of the scale bar in mm:")
        if not length_mm or length_mm <= 0:
            return

        self.scale_bar_length_mm = length_mm
        self.draw_scale_bar()

    def draw_scale_bar(self):
        if hasattr(self, 'scale_bar_items'):
            for item in self.scale_bar_items:
                self.canvas.delete(item)

        length_pixels = self.scale_bar_length_mm * self.spatial_scale * self.scale_factor

        # Position the scale bar at the bottom-left corner
        x = 20
        y = self.canvas.winfo_height() - 20

        self.scale_bar_items = []
        line = self.canvas.create_line(x, y, x + length_pixels, y, fill="white", width=3)
        text = self.canvas.create_text(x + length_pixels / 2, y - 10, text=f"{self.scale_bar_length_mm} mm", fill="white")
        self.scale_bar_items.extend([line, text])

        # Make the scale bar draggable
        self.canvas.tag_bind(line, "<ButtonPress-1>", self.on_scale_bar_press)
        self.canvas.tag_bind(line, "<B1-Motion>", self.on_scale_bar_drag)
        self.canvas.tag_bind(text, "<ButtonPress-1>", self.on_scale_bar_press)
        self.canvas.tag_bind(text, "<B1-Motion>", self.on_scale_bar_drag)

    def on_scale_bar_press(self, event):
        self.scale_bar_start_x = event.x
        self.scale_bar_start_y = event.y

    def on_scale_bar_drag(self, event):
        dx = event.x - self.scale_bar_start_x
        dy = event.y - self.scale_bar_start_y
        for item in self.scale_bar_items:
            self.canvas.move(item, dx, dy)
        self.scale_bar_start_x = event.x
        self.scale_bar_start_y = event.y

    def set_scale_manually(self):
        self.drawing_mode = "scale"
        print("Set drawing mode to 'scale'. Draw a circle to define the scale.")

    def calculate_scale_from_oval(self):
        if not self.oval:
            return

        x1, y1, x2, y2 = self.canvas.coords(self.oval)
        diameter_pixels = math.sqrt((x2 - x1)**2 + (y2 - y1)**2) / self.scale_factor

        known_distance = simpledialog.askfloat("Known Distance", "Enter the known distance in mm:")

        if known_distance and known_distance > 0:
            self.spatial_scale = diameter_pixels / known_distance
            print(f"Spatial scale set to: {self.spatial_scale:.2f} pixels/mm")
        else:
            print("Invalid distance entered. Scale not set.")

    def analyze(self):
        if not self.current_image or not self.rect:
            print("No image or ROI selected")
            return

        # Get ROI from canvas (in scaled coordinates)
        x1, y1, x2, y2 = self.canvas.coords(self.rect)
        
        # Convert scaled coordinates back to original image coordinates
        orig_x1 = int(x1 / self.scale_factor)
        orig_y1 = int(y1 / self.scale_factor)
        orig_x2 = int(x2 / self.scale_factor)
        orig_y2 = int(y2 / self.scale_factor)
        
        # Ensure coordinates are within image bounds
        img_width, img_height = self.current_image.size
        orig_x1 = max(0, min(orig_x1, img_width))
        orig_y1 = max(0, min(orig_y1, img_height))
        orig_x2 = max(0, min(orig_x2, img_width))
        orig_y2 = max(0, min(orig_y2, img_height))
        
        roi = (orig_x1, orig_y1, orig_x2, orig_y2)
        
        print(f"ROI in original coordinates: {roi}")

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
            self.last_analysis_result = analyzer.process_image(Path(filepath), roi=roi, spatial_scale_pixels_per_mm=self.spatial_scale)
            self.display_results(self.last_analysis_result)
        except Exception as e:
            print(f"Error during analysis: {e}")

    def show_heatmap(self):
        if not self.last_analysis_result:
            print("Please run an analysis first.")
            return

        # Create a visualizer
        vis_config = VisualizationConfig()
        visualizer = Visualizer(vis_config)

        # Create the heatmap figure
        fig = visualizer.create_thickness_heatmap(
            self.last_analysis_result.thickness_map,
            self.last_analysis_result.mask,
            self.last_analysis_result.saturation_mask
        )

        # Display the figure in a new window
        heatmap_window = tk.Toplevel(self)
        heatmap_window.title("Thickness Heatmap")
        canvas = FigureCanvasTkAgg(fig, master=heatmap_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def export_image(self):
        if not self.current_image:
            print("No image to export.")
            return

        filepath = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("All Files", "*.*")],
        )
        if not filepath:
            return

        # Create an image from the canvas
        self.canvas.postscript(file="canvas.ps", colormode="color")
        img = Image.open("canvas.ps")

        # If a heatmap exists, blend it with the image
        if self.last_analysis_result:
            vis_config = VisualizationConfig()
            visualizer = Visualizer(vis_config)
            fig = visualizer.create_thickness_heatmap(
                self.last_analysis_result.thickness_map,
                self.last_analysis_result.mask,
                self.last_analysis_result.saturation_mask
            )
            fig.canvas.draw()
            heatmap = Image.frombytes(
                "RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
            )
            heatmap = heatmap.resize(img.size)
            img = Image.blend(img, heatmap, alpha=0.5)

        img.save(filepath)
        print(f"Image saved to {filepath}")

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
