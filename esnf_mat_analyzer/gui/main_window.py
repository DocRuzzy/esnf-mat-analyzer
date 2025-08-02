
"""
Main GUI window for ESNF Mat Analyzer.

Author: ESNF Mat Analyzer Team
License: GNU General Public License v3.0 or later (GPLv3)
"""

import tkinter as tk
from tkinter import ttk, filedialog, simpledialog, messagebox
from PIL import Image, ImageTk
from pathlib import Path
import math
import numpy as np
import cv2
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from esnf_mat_analyzer.main import setup_dependencies
from esnf_mat_analyzer.visualization.visualization import Visualizer
from esnf_mat_analyzer.core.data_types import VisualizationConfig, BackgroundCorrectionMethod, ThicknessModelType
from esnf_mat_analyzer.config.config_manager import get_default_config

class MainWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Nanofiber Analyzer")
        self.geometry("1200x800")

        self.selected_files = []
        self.current_image_path = None
        self.current_image = None
        self.current_image_path = None
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

        # Background leveling checkbox
        self.background_leveling_var = tk.BooleanVar(value=True)
        self.background_leveling_checkbox = ttk.Checkbutton(
            self.analysis_frame, 
            text="Enable Background Leveling", 
            variable=self.background_leveling_var
        )
        self.background_leveling_checkbox.pack(padx=5, pady=2, anchor=tk.W)


        # Heatmap auto-range checkbox
        self.auto_range_var = tk.BooleanVar(value=True)
        self.auto_range_checkbox = ttk.Checkbutton(
            self.analysis_frame, 
            text="Auto-adjust Heatmap Range", 
            variable=self.auto_range_var
        )
        self.auto_range_checkbox.pack(padx=5, pady=2, anchor=tk.W)

        # Max coefficients for multiscale uniformity
        max_coeff_frame = ttk.Frame(self.analysis_frame)
        max_coeff_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(max_coeff_frame, text="Max Coefficients (memory limit):").pack(side=tk.LEFT)
        self.max_coeff_var = tk.IntVar(value=100000)
        self.max_coeff_spinbox = ttk.Spinbox(max_coeff_frame, from_=1000, to=1000000, increment=1000, textvariable=self.max_coeff_var, width=10)
        self.max_coeff_spinbox.pack(side=tk.LEFT, padx=2)

        # Background correction method selection
        self.bg_method_frame = ttk.LabelFrame(self.analysis_frame, text="Background Correction")
        self.bg_method_frame.pack(fill=tk.X, padx=5, pady=5)

        self.bg_method_var = tk.StringVar(value="none")
        bg_methods = [
            ("No Correction (Preserve Gradients)", "none"),
            ("Gentle Normalization", "gentle"),
            ("BASIC (Robust)", "basic"),
            ("Rolling Ball", "rolling_ball"),
            ("Percentile BG", "restore"),
            ("Homomorphic", "homomorphic"),
            ("ROI-Aware (Gentle)", "roi_aware")
        ]

        for text, value in bg_methods:
            ttk.Radiobutton(
                self.bg_method_frame,
                text=text,
                variable=self.bg_method_var,
                value=value,
                command=self.on_bg_method_change
            ).pack(anchor=tk.W, padx=5, pady=1)

        # Preview button for background methods
        self.preview_bg_button = ttk.Button(
            self.bg_method_frame,
            text="Preview Methods",
            command=self.preview_background_methods
        )
        self.preview_bg_button.pack(padx=5, pady=5)

        # Thickness model selection
        self.thickness_model_frame = ttk.LabelFrame(self.analysis_frame, text="Thickness Model")
        self.thickness_model_frame.pack(fill=tk.X, padx=5, pady=5)

        self.thickness_model_var = tk.StringVar(value="beer_lambert")
        thickness_models = [
            ("Beer-Lambert (Reflection Physics)", "beer_lambert"),
            ("Linear", "linear"),
            ("Logarithmic", "logarithmic"),
            ("Exponential", "exponential")
        ]

        for text, value in thickness_models:
            ttk.Radiobutton(
                self.thickness_model_frame,
                text=text,
                variable=self.thickness_model_var,
                value=value
            ).pack(anchor=tk.W, padx=5, pady=1)

        # Add model comparison button
        self.compare_models_button = ttk.Button(
            self.thickness_model_frame,
            text="Compare Models",
            command=self.compare_thickness_models
        )
        self.compare_models_button.pack(padx=5, pady=5)

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

        # Add scale visualization button
        self.visualize_scales_button = ttk.Button(
            self.analysis_frame, text="Visualize Wavelet Scales", command=self.visualize_wavelet_scales
        )
        self.visualize_scales_button.pack(padx=5, pady=5)

        # Add scale effectiveness test button
        self.test_scales_button = ttk.Button(
            self.analysis_frame, text="Test Scale Effectiveness", command=self.analyze_scale_effectiveness
        )
        self.test_scales_button.pack(padx=5, pady=5)

        # Add fiber size estimation button
        self.fiber_size_button = ttk.Button(
            self.analysis_frame, text="Estimate Fiber Size & Scales", command=self.estimate_fiber_size_pixels
        )
        self.fiber_size_button.pack(padx=5, pady=5)

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
            self.current_image_path = filepath  # Store the file path
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
        
        # Convert corner coordinates to (x, y, width, height) format
        x = orig_x1
        y = orig_y1
        width = orig_x2 - orig_x1
        height = orig_y2 - orig_y1
        
        roi = (x, y, width, height)
        
        print(f"ROI in original coordinates: {roi}")

        # Get selected file
        selection = self.file_listbox.curselection()
        if not selection:
            print("No file selected")
            return
        index = selection[0]
        filepath = self.selected_files[index]

        # Create config and analyzer
        config = get_default_config()
        # Update background leveling setting based on checkbox
        config.processing.leveling.enabled = self.background_leveling_var.get()
        # Update max_coefficients from GUI
        config.uniformity.max_coefficients = self.max_coeff_var.get()
        
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

        try:
            print("Creating heatmap visualization...")
            
            # Create a visualizer with custom configuration
            vis_config = VisualizationConfig()
            vis_config.auto_range_heatmap = self.auto_range_var.get()
            visualizer = Visualizer(vis_config)

            # Create the heatmap figure
            print("Generating thickness heatmap...")
            fig = visualizer.create_thickness_heatmap(
                self.last_analysis_result.thickness_map,
                self.last_analysis_result.mask,
                self.last_analysis_result.saturation_mask
            )
            print("Heatmap generated successfully.")

            # Display the figure in a new window
            heatmap_window = tk.Toplevel(self)
            heatmap_window.title("Thickness Heatmap")
            canvas = FigureCanvasTkAgg(fig, master=heatmap_window)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            print("Heatmap window opened successfully.")
            
        except Exception as e:
            print(f"Error creating heatmap: {e}")
            import traceback
            traceback.print_exc()

    def visualize_wavelet_scales(self):
        """Visualize what each wavelet scale is detecting."""
        if not self.last_analysis_result:
            print("Please run analysis first.")
            return
        
        try:
            import pywt
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            
            # Get ROI from the analysis result
            roi_image = self.last_analysis_result.thickness_map
            
            # Perform wavelet decomposition
            coeffs = pywt.wavedec2(roi_image, 'db4', level=2)
            
            # Create visualization window
            viz_window = tk.Toplevel(self)
            viz_window.title("Wavelet Scale Analysis")
            viz_window.geometry("1400x900")
            
            # Create figure with subplots
            fig = Figure(figsize=(16, 10))
            
            # Plot original with better contrast
            ax1 = fig.add_subplot(231)
            # Use a smaller percentile range to preserve gradients
            vmin_orig = np.percentile(roi_image, 5)
            vmax_orig = np.percentile(roi_image, 95)
            im1 = ax1.imshow(roi_image, cmap='gray', vmin=vmin_orig, vmax=vmax_orig)
            ax1.set_title('Original ROI\n(Input to Analysis)')
            ax1.axis('off')
            fig.colorbar(im1, ax=ax1, fraction=0.046)
            
            # Plot approximation (Scale 0) with preserved gradients
            ax2 = fig.add_subplot(232)
            vmin_approx = np.percentile(coeffs[0], 5)
            vmax_approx = np.percentile(coeffs[0], 95)
            im2 = ax2.imshow(coeffs[0], cmap='gray', vmin=vmin_approx, vmax=vmax_approx)
            ax2.set_title(f'Scale 0: Large Features\n({coeffs[0].shape[0]}×{coeffs[0].shape[1]} pixels)\nOverall thickness patterns')
            ax2.axis('off')
            fig.colorbar(im2, ax=ax2, fraction=0.046)
            
            # Plot detail coefficients for each level
            detail_titles = ['Horizontal Details', 'Vertical Details', 'Diagonal Details']
            colors = ['hot', 'plasma', 'inferno']
            scale_descriptions = [
                'Individual fiber features\n(4-8 pixel structures)',
                'Small fiber bundles\n(8-16 pixel structures)'
            ]
            
            plot_idx = 3
            for level in range(min(len(coeffs)-1, 2)):  # Limit to 2 levels
                for i, (detail, title, cmap) in enumerate(zip(coeffs[level+1], detail_titles, colors)):
                    if plot_idx <= 6:  # Only plot if we have space
                        ax = fig.add_subplot(2, 3, plot_idx)
                        # Use adaptive range for detail coefficients to show subtle features
                        detail_abs = np.abs(detail)
                        if detail_abs.max() > 0:
                            vmax_detail = np.percentile(detail_abs, 85)  # Conservative range
                            im = ax.imshow(detail_abs, cmap=cmap, vmin=0, vmax=vmax_detail)
                        else:
                            im = ax.imshow(detail_abs, cmap=cmap, vmin=0)
                        
                        # Create descriptive title
                        desc = scale_descriptions[level] if level < len(scale_descriptions) else f'Scale {level+1} details'
                        ax.set_title(f'Scale {level+1}: {title}\n({detail.shape[0]}×{detail.shape[1]} pixels)\n{desc}')
                        ax.axis('off')
                        fig.colorbar(im, ax=ax, fraction=0.046)
                        plot_idx += 1
            
            # Add comprehensive statistics as text
            fig.suptitle('Wavelet Decomposition Analysis - Understanding Your Nanofiber Mat at Different Scales', 
                        fontsize=14, fontweight='bold')
            
            # Embed in window
            canvas = FigureCanvasTkAgg(fig, viz_window)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Add detailed statistics frame
            stats_frame = tk.Frame(viz_window)
            stats_frame.pack(fill=tk.X, padx=10, pady=5)
            
            # Calculate detailed statistics
            approx_std = np.std(coeffs[0])
            detail_stds = []
            for level in range(len(coeffs)-1):
                level_std = np.mean([np.std(d) for d in coeffs[level+1]])
                detail_stds.append(level_std)
            
            # Estimate feature sizes based on image dimensions and scale
            img_height, img_width = roi_image.shape
            pixel_sizes = []
            for level in range(len(coeffs)):
                if level == 0:
                    pixel_sizes.append("2-4 pixels")
                else:
                    min_size = 2 ** (level + 1)
                    max_size = 2 ** (level + 2)
                    pixel_sizes.append(f"{min_size}-{max_size} pixels")
            
            stats_text = f"""
WAVELET SCALE ANALYSIS RESULTS:

Image Information:
• Original ROI size: {img_width} × {img_height} pixels
• Total decomposition levels: {len(coeffs)-1}

Scale Breakdown & Feature Detection:
• Scale 0 (Approximation): {pixel_sizes[0]} features - Standard deviation: {approx_std:.2f}
  → Captures overall thickness gradients and large-scale uniformity patterns
  → High variation here indicates non-uniform thickness across the mat
"""
            
            for i, (std_val, pixel_size) in enumerate(zip(detail_stds, pixel_sizes[1:])):
                feature_type = ""
                if i == 0:
                    feature_type = "Individual nanofibers and fine surface texture"
                elif i == 1:
                    feature_type = "Small fiber bundles and processing artifacts"
                else:
                    feature_type = f"Larger structural features (level {i+1})"
                
                stats_text += f"""
• Scale {i+1} (Details): {pixel_size} features - Average standard deviation: {std_val:.2f}
  → {feature_type}
  → {'High' if std_val > 10 else 'Moderate' if std_val > 5 else 'Low'} variation detected at this scale
"""
            
            stats_text += f"""

Interpretation Guide:
• Higher standard deviation = More variation = Lower uniformity at that scale
• Scale 0 should dominate for uniform mats (smooth thickness variation)
• Scale 1-2 capture fiber-level features (most important for nanofiber analysis)
• If all scales show low variation, background correction may be too aggressive

Physical Scale Estimates (approximate):
• If your nanofibers are ~500nm diameter and spatial scale is known
• Scale 1 features ({pixel_sizes[1] if len(pixel_sizes) > 1 else 'N/A'}) should capture individual fiber variations
• Scale 2 features ({pixel_sizes[2] if len(pixel_sizes) > 2 else 'N/A'}) should capture small bundle formations
"""
            
            stats_label = tk.Label(stats_frame, text=stats_text, justify=tk.LEFT, 
                                 font=('Courier', 9), bg='lightgray', relief='sunken')
            stats_label.pack(fill=tk.X, padx=5, pady=5)
            
        except ImportError:
            tk.messagebox.showerror("Error", "PyWavelets not installed. Please install with: pip install PyWavelets")
        except Exception as e:
            print(f"Error creating scale visualization: {e}")
            import traceback
            traceback.print_exc()
            tk.messagebox.showerror("Error", f"Failed to create scale visualization: {str(e)}")

    def analyze_scale_effectiveness(self):
        """Test if wavelet scales are detecting meaningful features."""
        if not self.last_analysis_result:
            tk.messagebox.showwarning("Warning", "Please run analysis first.")
            return
        
        try:
            # Create test images
            mat_image = self.last_analysis_result.thickness_map
            height, width = mat_image.shape
            
            # 1. Completely uniform image
            uniform_test = np.ones_like(mat_image) * 128
            
            # 2. Random noise image  
            noise_test = np.random.random(mat_image.shape) * 255
            
            # 3. Synthetic fiber pattern (for comparison)
            x, y = np.meshgrid(np.arange(width), np.arange(height))
            synthetic_fibers = 128 + 30 * np.sin(x * 0.1) * np.cos(y * 0.1) + np.random.normal(0, 5, mat_image.shape)
            synthetic_fibers = np.clip(synthetic_fibers, 0, 255).astype(np.uint8)
            
            from esnf_mat_analyzer.analysis.multiscale_uniformity import MultiScaleUniformityAnalyzer
            analyzer = MultiScaleUniformityAnalyzer(levels=2, max_coefficients=5000)
            
            # Create a simple mask for testing
            test_mask = np.ones_like(mat_image, dtype=bool)
            
            # Test all images
            uniform_scores = analyzer.analyze_multiscale_uniformity(uniform_test, test_mask)
            noise_scores = analyzer.analyze_multiscale_uniformity(noise_test, test_mask)
            synthetic_scores = analyzer.analyze_multiscale_uniformity(synthetic_fibers, test_mask)
            mat_scores = analyzer.analyze_multiscale_uniformity(mat_image, self.last_analysis_result.mask)
            
            # Create results window
            results_window = tk.Toplevel(self)
            results_window.title("Scale Effectiveness Test Results")
            results_window.geometry("900x700")
            
            # Create text widget for results
            text_widget = tk.Text(results_window, wrap=tk.WORD, font=('Courier', 10))
            text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Format results
            results_text = "WAVELET SCALE EFFECTIVENESS TEST\n"
            results_text += "=" * 50 + "\n\n"
            results_text += "This test evaluates whether the wavelet scales are properly detecting\n"
            results_text += "different types of features in your nanofiber mat images.\n\n"
            
            results_text += "Expected Results:\n"
            results_text += "• Uniform image → Scores near 1.0 (perfect uniformity)\n"
            results_text += "• Random noise → Scores near 0.0 (no uniformity)\n"
            results_text += "• Synthetic fibers → Intermediate scores (structured but not uniform)\n"
            results_text += "• Your mat → Should fall between synthetic and uniform\n\n"
            
            results_text += "ACTUAL RESULTS:\n"
            results_text += "-" * 30 + "\n\n"
            
            def format_scores(scores, name):
                text = f"{name}:\n"
                for scale, score in scores.items():
                    if isinstance(score, (int, float)) and not np.isnan(score):
                        rating = self._get_uniformity_rating(score)
                        text += f"  {scale}: {score:.4f} ({rating})\n"
                    else:
                        text += f"  {scale}: {score}\n"
                return text + "\n"
            
            results_text += format_scores(uniform_scores, "Uniform Test Image")
            results_text += format_scores(noise_scores, "Random Noise Image")
            results_text += format_scores(synthetic_scores, "Synthetic Fiber Pattern")
            results_text += format_scores(mat_scores, "Your Nanofiber Mat")
            
            # Analysis and recommendations
            results_text += "ANALYSIS & RECOMMENDATIONS:\n"
            results_text += "-" * 35 + "\n\n"
            
            # Check if scales are working properly
            uniform_avg = np.mean([v for v in uniform_scores.values() if isinstance(v, (int, float)) and not np.isnan(v)])
            noise_avg = np.mean([v for v in noise_scores.values() if isinstance(v, (int, float)) and not np.isnan(v)])
            mat_avg = np.mean([v for v in mat_scores.values() if isinstance(v, (int, float)) and not np.isnan(v)])
            
            if uniform_avg > 0.8 and noise_avg < 0.3:
                results_text += "✓ SCALES WORKING CORRECTLY:\n"
                results_text += "  Scales can distinguish between uniform and random patterns.\n\n"
            else:
                results_text += "⚠ POTENTIAL SCALE ISSUES:\n"
                results_text += "  Scales may not be properly detecting uniformity differences.\n"
                results_text += "  Consider adjusting wavelet parameters or background correction.\n\n"
            
            if mat_avg > 0.9:
                results_text += "⚠ MAT SCORES TOO HIGH:\n"
                results_text += "  Your mat shows nearly perfect uniformity (suspiciously high).\n"
                results_text += "  This suggests background correction may be too aggressive,\n"
                results_text += "  removing real mat features along with background variations.\n"
                results_text += "  Try using 'Percentile BG' method instead.\n\n"
            elif mat_avg < 0.1:
                results_text += "⚠ MAT SCORES TOO LOW:\n"
                results_text += "  Your mat shows very poor uniformity.\n"
                results_text += "  This could indicate real mat quality issues,\n"
                results_text += "  or insufficient background correction.\n\n"
            else:
                results_text += "✓ MAT SCORES REASONABLE:\n"
                results_text += f"  Average uniformity: {mat_avg:.3f}\n"
                results_text += "  This suggests the analysis is detecting real mat features.\n\n"
            
            # Scale-specific recommendations
            results_text += "SCALE-SPECIFIC INSIGHTS:\n"
            for scale_name, score in mat_scores.items():
                if isinstance(score, (int, float)) and not np.isnan(score):
                    if 'scale_0' in scale_name:
                        results_text += f"• Large-scale uniformity: {score:.3f}\n"
                        results_text += "  (Overall thickness consistency across the mat)\n"
                    elif 'scale_1' in scale_name:
                        results_text += f"• Fiber-level uniformity: {score:.3f}\n"
                        results_text += "  (Individual fiber and surface texture consistency)\n"
                    elif 'scale_2' in scale_name:
                        results_text += f"• Bundle-level uniformity: {score:.3f}\n"
                        results_text += "  (Small fiber bundle and defect consistency)\n"
            
            text_widget.insert(tk.END, results_text)
            text_widget.config(state=tk.DISABLED)
            
            # Add close button
            close_button = ttk.Button(results_window, text="Close", command=results_window.destroy)
            close_button.pack(pady=10)
            
        except Exception as e:
            print(f"Error in scale effectiveness test: {e}")
            import traceback
            traceback.print_exc()
            tk.messagebox.showerror("Error", f"Scale effectiveness test failed: {str(e)}")

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
            vis_config.auto_range_heatmap = self.auto_range_var.get()
            visualizer = Visualizer(vis_config)
            fig = visualizer.create_thickness_heatmap(
                self.last_analysis_result.thickness_map,
                self.last_analysis_result.mask,
                self.last_analysis_result.saturation_mask
            )
            fig.canvas.draw()
            
            # Convert matplotlib figure to image
            fig_img = Image.frombytes(
                "RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
            )
            # Here you could blend img and fig_img if needed
            img = fig_img
            
        img.save(filepath)
        print(f"Image saved to {filepath}")

    def on_bg_method_change(self):
        """Called when background correction method is changed."""
        # Optional: Could show a brief description or update UI
        pass

    def preview_background_methods(self):
        """Show a comparison of different background correction methods."""
        if not self.current_image or not self.current_image_path:
            tk.messagebox.showwarning("Warning", "Please select an image first.")
            return

        # Create preview window
        preview_window = tk.Toplevel(self)
        preview_window.title("Background Correction Methods Comparison")
        preview_window.geometry("1400x800")

        # Create notebook for tabs
        notebook = ttk.Notebook(preview_window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Load and preprocess the image
        try:
            import cv2
            raw_image = cv2.imread(str(self.current_image_path))
            raw_image_rgb = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
            gray_image = cv2.cvtColor(raw_image_rgb, cv2.COLOR_RGB2GRAY)

            # Define methods to test with descriptions and links
            methods = [
                ("No Correction", "none", self._apply_no_correction, 
                 "No background correction applied. Preserves all original gradients and fiber details.",
                 ""),
                ("Gentle Normalization", "gentle", self._apply_gentle_correction, 
                 "Very light correction that preserves fine gradients while reducing gross illumination variations.",
                 ""),
                ("BASIC (Robust)", "basic", self._apply_basic_correction,
                 "Robust background correction using polynomial fitting and statistical outlier removal.",
                 "https://scikit-image.org/docs/stable/auto_examples/color_exposure/plot_local_equalize.html"),
                ("Rolling Ball", "rolling_ball", self._apply_rolling_ball_correction,
                 "ImageJ-style rolling ball algorithm that estimates background by rolling a sphere under the image surface.",
                 "https://imagej.net/plugins/rolling-ball-background-subtraction"),
                ("Percentile BG", "restore", self._apply_restore_correction,
                 "Automatic negative control region identification. Uses the darkest regions (lowest percentile) as background reference for subtraction.",
                 "https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_thresholding.html"),
                ("Homomorphic", "homomorphic", self._apply_homomorphic_correction,
                 "Frequency domain filtering that separates illumination from reflectance components.",
                 "https://en.wikipedia.org/wiki/Homomorphic_filtering")
            ]

            for method_name, method_id, method_func, description, learn_more_url in methods:
                # Create tab for this method
                tab_frame = ttk.Frame(notebook)
                notebook.add(tab_frame, text=method_name)

                # Create a frame for the description and link at the top
                info_frame = ttk.Frame(tab_frame)
                info_frame.pack(fill=tk.X, padx=10, pady=5)
                
                # Add description
                desc_label = ttk.Label(info_frame, text=description, wraplength=400, justify=tk.LEFT)
                desc_label.pack(anchor=tk.W)
                
                # Add clickable link
                link_label = ttk.Label(info_frame, text="📖 Learn more about this method", 
                                     foreground="blue", cursor="hand2")
                link_label.pack(anchor=tk.W, pady=(2, 0))
                
                # Make the link clickable
                def open_link(url=learn_more_url):
                    import webbrowser
                    webbrowser.open(url)
                
                link_label.bind("<Button-1>", lambda e, url=learn_more_url: open_link(url))
                
                # Add hover effect
                def on_enter(e, label=link_label):
                    label.configure(foreground="darkblue")
                
                def on_leave(e, label=link_label):
                    label.configure(foreground="blue")
                
                link_label.bind("<Enter>", on_enter)
                link_label.bind("<Leave>", on_leave)

                # Apply the background correction method
                try:
                    corrected_image = method_func(gray_image)
                    
                    # Create side-by-side comparison
                    fig = Figure(figsize=(12, 5))
                    
                    # Original image
                    ax1 = fig.add_subplot(121)
                    ax1.imshow(gray_image, cmap='gray')
                    ax1.set_title('Original Image')
                    ax1.axis('off')
                    
                    # Corrected image
                    ax2 = fig.add_subplot(122)
                    ax2.imshow(corrected_image, cmap='viridis')
                    ax2.set_title(f'{method_name} Corrected')
                    ax2.axis('off')
                    
                    # Add statistics
                    stats_text = self._get_correction_stats(gray_image, corrected_image)
                    fig.suptitle(f'{method_name}\n{stats_text}', fontsize=10)
                    
                    # Embed plot in tab
                    canvas = FigureCanvasTkAgg(fig, tab_frame)
                    canvas.draw()
                    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                    
                    # Add selection button
                    select_button = ttk.Button(
                        tab_frame,
                        text=f"Use {method_name}",
                        command=lambda m=method_id: self._select_bg_method(m, preview_window)
                    )
                    select_button.pack(pady=10)

                except Exception as e:
                    # Show error in tab
                    error_label = ttk.Label(tab_frame, text=f"Error: {str(e)}")
                    error_label.pack(expand=True)

        except Exception as e:
            tk.messagebox.showerror("Error", f"Failed to create preview: {str(e)}")

    def _apply_no_correction(self, image):
        """Apply no correction - return original image."""
        return image

    def _apply_gentle_correction(self, image):
        """Apply very gentle normalization."""
        h, w = image.shape
        kernel_size = max(101, min(w//3, h//3))
        if kernel_size % 2 == 0:
            kernel_size += 1
            
        # Very gentle background estimation
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size//4, kernel_size//4))
        background = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        
        correction_strength = 0.15
        image_float = image.astype(np.float32)
        bg_float = background.astype(np.float32)
        
        mean_image = np.mean(image_float)
        mean_bg = np.mean(bg_float)
        
        if abs(mean_bg - mean_image) > 10:
            correction = (bg_float - mean_bg) * correction_strength
            corrected = image_float - correction
        else:
            corrected = image_float
        
        # Preserve original dynamic range
        orig_min, orig_max = float(image.min()), float(image.max())
        corrected_min, corrected_max = float(corrected.min()), float(corrected.max())
        
        if corrected_max > corrected_min:
            corrected = ((corrected - corrected_min) / (corrected_max - corrected_min)) * (orig_max - orig_min) + orig_min
        
        return np.clip(corrected, 0, 255).astype(np.uint8)

    def _apply_default_correction(self, image):
        """Apply default normalization correction."""
        kernel_size = 25
        background = cv2.medianBlur(image, kernel_size)
        image_float = image.astype(np.float32)
        background_float = background.astype(np.float32)
        epsilon = 1.0
        corrected = (image_float / (background_float + epsilon)) * 128.0
        return np.clip(corrected, 0, 255).astype(np.uint8)

    def _apply_basic_correction(self, image):
        """Apply BASIC correction."""
        from esnf_mat_analyzer.processing.advanced_background import AdvancedBackgroundProcessor
        processor = AdvancedBackgroundProcessor()
        return processor.basic_correction(image, 1)

    def _apply_rolling_ball_correction(self, image):
        """Apply rolling ball correction."""
        from esnf_mat_analyzer.processing.advanced_background import AdvancedBackgroundProcessor
        processor = AdvancedBackgroundProcessor()
        return processor.rolling_ball_3d(image, 50)

    def _apply_restore_correction(self, image):
        """Apply Percentile BG correction."""
        from esnf_mat_analyzer.processing.advanced_background import AdvancedBackgroundProcessor
        processor = AdvancedBackgroundProcessor()
        return processor.restore_method(image, 5.0)

    def _apply_homomorphic_correction(self, image):
        """Apply homomorphic correction."""
        from esnf_mat_analyzer.processing.advanced_background import AdvancedBackgroundProcessor
        processor = AdvancedBackgroundProcessor()
        return processor.homomorphic_filter(image, 30, 0.5, 2.0)

    def _get_correction_stats(self, original, corrected):
        """Get statistical comparison of correction methods."""
        orig_stats = f"Original: {original.min()}-{original.max()}, μ={original.mean():.1f}"
        corr_stats = f"Corrected: {corrected.min()}-{corrected.max()}, μ={corrected.mean():.1f}"
        sat_rate = 100 * np.sum(corrected >= 240) / corrected.size
        return f"{orig_stats}\n{corr_stats}\nSaturation: {sat_rate:.1f}%"

    def _select_bg_method(self, method_id, preview_window):
        """Select a background correction method and close preview."""
        self.bg_method_var.set(method_id)
        preview_window.destroy()
        tk.messagebox.showinfo("Selection", f"Background correction method updated to: {method_id}")

    def compare_thickness_models(self):
        """Show comparison of different thickness models."""
        if not self.current_image:
            tk.messagebox.showwarning("Warning", "Please select an image first.")
            return

        try:
            # Quick analysis with current image
            from esnf_mat_analyzer.processing.physics_based_thickness import MultiModelThicknessEstimator
            
            # Convert PIL to numpy
            img_array = np.array(self.current_image.convert('L'))
            
            estimator = MultiModelThicknessEstimator()
            results = estimator.validate_model_performance(img_array)
            
            # Create results window
            results_window = tk.Toplevel(self)
            results_window.title("Thickness Model Comparison")
            results_window.geometry("800x600")
            
            # Create text widget with results
            text_widget = tk.Text(results_window, wrap=tk.WORD, font=('Courier', 10))
            text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Format results
            comparison_text = "THICKNESS MODEL COMPARISON\n" + "="*50 + "\n\n"
            
            for model_name, stats in results.items():
                comparison_text += f"{model_name.upper()} MODEL:\n"
                comparison_text += f"  Mean Thickness: {stats['mean_thickness']:.2f}\n"
                comparison_text += f"  Std Deviation: {stats['std_thickness']:.2f}\n"
                comparison_text += f"  Dynamic Range: {stats['dynamic_range']:.2f}\n"
                comparison_text += f"  Signal/Noise: {stats['signal_to_noise']:.2f}\n"
                comparison_text += f"  Coeff. of Variation: {stats['coefficient_of_variation']:.3f}\n\n"
            
            comparison_text += "\nRECOMMENDATION:\n"
            comparison_text += "Beer-Lambert model is recommended for physics-based analysis\n"
            comparison_text += "with improved accuracy for fiber mat thickness estimation.\n"
            
            text_widget.insert(tk.END, comparison_text)
            text_widget.config(state=tk.DISABLED)
            
        except Exception as e:
            tk.messagebox.showerror("Error", f"Model comparison failed: {str(e)}")

    def get_roi_coordinates(self):
        """Get ROI coordinates from canvas rectangle in (x, y, width, height) format."""
        if not self.rect:
            return None
            
        # Get ROI from canvas (in scaled coordinates)
        x1, y1, x2, y2 = self.canvas.coords(self.rect)
        
        # Convert scaled coordinates back to original image coordinates
        orig_x1 = int(x1 / self.scale_factor)
        orig_y1 = int(y1 / self.scale_factor)
        orig_x2 = int(x2 / self.scale_factor)
        orig_y2 = int(y2 / self.scale_factor)
        
        # Ensure coordinates are within image bounds
        if self.current_image:
            img_width, img_height = self.current_image.size
            orig_x1 = max(0, min(orig_x1, img_width))
            orig_y1 = max(0, min(orig_y1, img_height))
            orig_x2 = max(0, min(orig_x2, img_width))
            orig_y2 = max(0, min(orig_y2, img_height))
        
        # Convert corner coordinates to (x, y, width, height) format
        x = orig_x1
        y = orig_y1
        width = orig_x2 - orig_x1
        height = orig_y2 - orig_y1
        
        return (x, y, width, height)

    def analyze(self):
        """Modified analyze method to use selected background correction method and thickness model."""
        if not self.current_image:
            print("No image selected.")
            return

        # Get ROI coordinates
        roi = self.get_roi_coordinates()
        if not roi:
            print("No ROI selected.")
            return

        try:
            # Create config with selected methods
            config = get_default_config()
            
            # Background correction method
            method_mapping = {
                "none": BackgroundCorrectionMethod.NONE,
                "basic": BackgroundCorrectionMethod.BASIC,
                "rolling_ball": BackgroundCorrectionMethod.ROLLING_BALL,
                "restore": BackgroundCorrectionMethod.RESTORE,
                "homomorphic": BackgroundCorrectionMethod.HOMOMORPHIC
            }
            
            selected_bg_method = self.bg_method_var.get()
            config.processing.background_correction_method = method_mapping.get(
                selected_bg_method, BackgroundCorrectionMethod.NONE
            )
            
            # Thickness model selection
            thickness_mapping = {
                "beer_lambert": ThicknessModelType.BEER_LAMBERT,
                "linear": ThicknessModelType.LINEAR,
                "logarithmic": ThicknessModelType.LOGARITHMIC,
                "exponential": ThicknessModelType.EXPONENTIAL
            }
            
            selected_thickness_model = self.thickness_model_var.get()
            config.thickness.model_type = thickness_mapping.get(
                selected_thickness_model, ThicknessModelType.BEER_LAMBERT
            )
            
            # Apply other settings
            config.processing.leveling.enabled = self.background_leveling_var.get()

            # Get selected file path
            selection = self.file_listbox.curselection()
            if not selection:
                print("No file selected")
                return
            index = selection[0]
            filepath = self.selected_files[index]

            # Create analyzer with updated config
            analyzer = setup_dependencies(config)

            # Run analysis
            print(f"Analyzing with background: {selected_bg_method}, thickness: {selected_thickness_model}")
            print(f"ROI (x, y, width, height): {roi}")
            
            self.last_analysis_result = analyzer.process_image(
                Path(filepath), roi=roi, spatial_scale_pixels_per_mm=self.spatial_scale
            )
            print("Analysis complete.")

            # Display results
            self.display_results(self.last_analysis_result)

        except Exception as e:
            print(f"Analysis failed: {e}")
            import traceback
            traceback.print_exc()

    def export_image(self):
        """Export the current image with optional heatmap overlay."""
        if not self.current_image:
            print("No image to export.")
            return

        filepath = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("All Files", "*.*")],
        )
        if not filepath:
            return

        # Create an image from the current display
        try:
            # Simple export of the current image for now
            pil_image = Image.fromarray(cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB))
            pil_image.save(filepath)
            print(f"Image saved to {filepath}")
        except Exception as e:
            print(f"Export failed: {e}")

    def display_results(self, result):
        # Create a new window to display results
        results_window = tk.Toplevel(self)
        results_window.title("Analysis Results")
        results_window.geometry("800x600")

        # Create notebook for tabbed results
        notebook = ttk.Notebook(results_window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Basic Results Tab
        basic_frame = ttk.Frame(notebook)
        notebook.add(basic_frame, text="Basic Metrics")
        
        basic_text = tk.Text(basic_frame, wrap=tk.WORD)
        basic_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Format basic results
        basic_result = f"Image: {result.image_path}\n"
        basic_result += f"Processing Time: {result.processing_time:.2f}s\n"
        if result.spatial_scale_pixels_per_mm:
            basic_result += f"Spatial Scale: {result.spatial_scale_pixels_per_mm:.2f} pixels/mm\n"
        
        basic_result += "\nTraditional Uniformity Metrics:\n"
        traditional_metrics = {k: v for k, v in result.metrics.items() 
                             if not k.startswith(('anisotropy_', 'texture_', 'psd_', 'overall_mat_uniformity'))}
        for name, value in traditional_metrics.items():
            if isinstance(value, float) and not math.isnan(value):
                basic_result += f"  {name}: {value:.4f}\n"
            else:
                basic_result += f"  {name}: {value}\n"
        
        basic_text.insert(tk.END, basic_result)
        basic_text.config(state=tk.DISABLED)

        # Mat-Scale Analysis Tab
        mat_frame = ttk.Frame(notebook)
        notebook.add(mat_frame, text="Mat-Scale Analysis")
        
        mat_text = tk.Text(mat_frame, wrap=tk.WORD)
        mat_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Format mat-scale results
        mat_result = "Mat-Scale Uniformity Analysis\n"
        mat_result += "=" * 40 + "\n\n"
        
        # Overall mat uniformity score
        if 'overall_mat_uniformity' in result.metrics:
            score = result.metrics['overall_mat_uniformity']
            if isinstance(score, float) and not math.isnan(score):
                mat_result += f"Overall Mat Uniformity Score: {score:.4f}\n"
                mat_result += f"Uniformity Rating: {self._get_uniformity_rating(score)}\n\n"
        
        # Anisotropy metrics
        anisotropy_metrics = {k: v for k, v in result.metrics.items() if k.startswith('anisotropy_')}
        if anisotropy_metrics:
            mat_result += "FFT-Based Anisotropy Analysis:\n"
            mat_result += "-" * 30 + "\n"
            for name, value in anisotropy_metrics.items():
                clean_name = name.replace('anisotropy_', '').replace('_', ' ').title()
                if isinstance(value, float) and not math.isnan(value):
                    mat_result += f"  {clean_name}: {value:.4f}\n"
                else:
                    mat_result += f"  {clean_name}: {value}\n"
            mat_result += "\n"
        
        # Texture metrics
        texture_metrics = {k: v for k, v in result.metrics.items() if k.startswith('texture_')}
        if texture_metrics:
            mat_result += "GLCM Texture Analysis:\n"
            mat_result += "-" * 25 + "\n"
            for name, value in texture_metrics.items():
                clean_name = name.replace('texture_', '').replace('_', ' ').title()
                if isinstance(value, float) and not math.isnan(value):
                    mat_result += f"  {clean_name}: {value:.4f}\n"
                else:
                    mat_result += f"  {clean_name}: {value}\n"
            mat_result += "\n"
        
        # Power spectral density metrics
        psd_metrics = {k: v for k, v in result.metrics.items() if k.startswith('psd_')}
        if psd_metrics:
            mat_result += "Power Spectral Density Analysis:\n"
            mat_result += "-" * 35 + "\n"
            for name, value in psd_metrics.items():
                clean_name = name.replace('psd_', '').replace('_', ' ').title()
                if isinstance(value, float) and not math.isnan(value):
                    mat_result += f"  {clean_name}: {value:.4f}\n"
                else:
                    mat_result += f"  {clean_name}: {value}\n"
        
        # Multiscale uniformity metrics
        scale_metrics = {k: v for k, v in result.metrics.items() if k.startswith('scale_') and k.endswith('_uniformity')}
        if scale_metrics:
            mat_result += "Multiscale Uniformity Analysis:\n"
            mat_result += "-" * 35 + "\n"
            for name, value in sorted(scale_metrics.items()):
                scale_level = name.replace('scale_', '').replace('_uniformity', '')
                if isinstance(value, float) and not math.isnan(value):
                    mat_result += f"  Scale Level {scale_level}: {value:.4f}\n"
                else:
                    mat_result += f"  Scale Level {scale_level}: {value}\n"
            
            # Calculate overall multiscale score
            valid_values = [v for v in scale_metrics.values() if isinstance(v, float) and not math.isnan(v)]
            if valid_values:
                overall_multiscale = np.mean(valid_values)
                mat_result += f"\n  Overall Multiscale Uniformity: {overall_multiscale:.4f}\n"
                mat_result += f"  Multiscale Rating: {self._get_uniformity_rating(overall_multiscale)}\n"
            mat_result += "\n"
        
        # Add interpretation guide
        mat_result += "\n" + "=" * 40 + "\n"
        mat_result += "Interpretation Guide:\n"
        mat_result += "- Anisotropy Index: Lower values indicate more isotropic (uniform) structure\n"
        mat_result += "- Homogeneity: Higher values indicate more uniform texture\n"
        mat_result += "- Energy: Higher values indicate more ordered structure\n"
        mat_result += "- Correlation: Measures linear dependencies in texture\n"
        mat_result += "- Contrast: Lower values indicate smoother texture\n"
        mat_result += "- PSD Uniformity: Higher values indicate more uniform frequency distribution\n"
        mat_result += "- Multiscale Uniformity: Higher values indicate better uniformity at different length scales\n"
        
        mat_text.insert(tk.END, mat_result)
        mat_text.config(state=tk.DISABLED)

    def _get_uniformity_rating(self, score):
        """Convert uniformity score to a descriptive rating."""
        if score >= 0.8:
            return "Excellent"
        elif score >= 0.6:
            return "Good"
        elif score >= 0.4:
            return "Fair"
        elif score >= 0.2:
            return "Poor"
        else:
            return "Very Poor"

    def estimate_fiber_size_pixels(self):
        """Estimate typical fiber diameter in pixels and show scale information."""
        if not self.spatial_scale:
            tk.messagebox.showwarning("Warning", "Please set spatial scale first.")
            return
        
        # Create dialog for fiber size estimation
        estimation_window = tk.Toplevel(self)
        estimation_window.title("Fiber Size & Scale Information")
        estimation_window.geometry("600x500")
        
        # Main frame
        main_frame = ttk.Frame(estimation_window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        title_label = ttk.Label(main_frame, text="Nanofiber Size Estimation & Scale Analysis", 
                               font=('Arial', 12, 'bold'))
        title_label.pack(pady=(0, 10))
        
        # Input frame for fiber diameter
        input_frame = ttk.LabelFrame(main_frame, text="Expected Fiber Properties")
        input_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Fiber diameter input
        diameter_frame = ttk.Frame(input_frame)
        diameter_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(diameter_frame, text="Typical fiber diameter (nm):").pack(side=tk.LEFT)
        diameter_var = tk.StringVar(value="500")
        diameter_entry = ttk.Entry(diameter_frame, textvariable=diameter_var, width=10)
        diameter_entry.pack(side=tk.LEFT, padx=(5, 0))
        
        # Results text area
        results_text = tk.Text(main_frame, wrap=tk.WORD, font=('Courier', 10), height=20)
        results_text.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        def calculate_estimates():
            try:
                fiber_diameter_nm = float(diameter_var.get())
                fiber_diameter_um = fiber_diameter_nm / 1000  # Convert to micrometers
                fiber_diameter_mm = fiber_diameter_um / 1000  # Convert to mm
                fiber_diameter_pixels = fiber_diameter_mm * self.spatial_scale
                
                # Calculate scale relevance
                scale_info = f"""
FIBER SIZE & WAVELET SCALE ANALYSIS
{'='*50}

Current Spatial Scale: {self.spatial_scale:.2f} pixels/mm
Expected Fiber Diameter: {fiber_diameter_nm:.0f} nm

CALCULATED FIBER SIZE:
• {fiber_diameter_pixels:.2f} pixels diameter
• {1/self.spatial_scale*1000:.2f} μm per pixel resolution

WAVELET SCALE RELEVANCE:
• Scale 0 (2-4 pixel features): {'✓ RELEVANT' if fiber_diameter_pixels >= 2 else '✗ TOO SMALL'} for fiber detection
  → Captures features {2/self.spatial_scale*1000:.0f}-{4/self.spatial_scale*1000:.0f} nm
  
• Scale 1 (4-8 pixel features): {'✓ OPTIMAL' if 4 <= fiber_diameter_pixels <= 8 else '✓ RELEVANT' if fiber_diameter_pixels >= 4 else '✗ TOO SMALL'} for individual fibers
  → Captures features {4/self.spatial_scale*1000:.0f}-{8/self.spatial_scale*1000:.0f} nm
  
• Scale 2 (8-16 pixel features): {'✓ RELEVANT' if fiber_diameter_pixels <= 16 else '✗ TOO LARGE'} for fiber bundles
  → Captures features {8/self.spatial_scale*1000:.0f}-{16/self.spatial_scale*1000:.0f} nm

RECOMMENDATIONS:
"""
                
                if fiber_diameter_pixels < 2:
                    scale_info += """
⚠ FIBERS TOO SMALL FOR CURRENT SCALES:
  • Your fibers are smaller than the smallest wavelet scale
  • Consider using higher magnification images
  • Scale 1 might not capture individual fiber variations effectively
"""
                elif 4 <= fiber_diameter_pixels <= 8:
                    scale_info += """
✓ OPTIMAL SCALE SETUP:
  • Scale 1 should effectively capture individual fiber variations
  • Scale 2 will capture small bundle formations
  • Your wavelet analysis should work well for fiber uniformity
"""
                elif fiber_diameter_pixels > 16:
                    scale_info += """
⚠ FIBERS LARGE RELATIVE TO SCALES:
  • Consider increasing wavelet decomposition levels
  • Current scales might miss fiber-level details
  • Scale 0 might be most relevant for your fiber size
"""
                else:
                    scale_info += """
✓ REASONABLE SCALE SETUP:
  • Scales should capture fiber features reasonably well
  • Some scales may be more relevant than others
"""
                
                # Add interpretation guide
                scale_info += f"""

EXPECTED UNIFORMITY BEHAVIOR:
• If your mat has uniform fiber distribution:
  → Scale 1 should show high uniformity (>0.6)
  → Scale 2 should show moderate uniformity
  → Scale 0 should show overall thickness uniformity

• If uniformity scores are all very high (>0.9):
  → Background correction may be too aggressive
  → Try 'Percentile BG' method instead

• If uniformity scores are all very low (<0.3):
  → May indicate real quality issues
  → Or insufficient background correction

PIXEL-TO-PHYSICAL CONVERSION:
• 1 pixel = {1/self.spatial_scale*1000:.1f} μm = {1/self.spatial_scale*1000000:.0f} nm
• Scale 1 features = {4/self.spatial_scale*1000:.0f}-{8/self.spatial_scale*1000:.0f} nm (individual fibers)
• Scale 2 features = {8/self.spatial_scale*1000:.0f}-{16/self.spatial_scale*1000:.0f} nm (small bundles)
"""
                
                results_text.delete(1.0, tk.END)
                results_text.insert(tk.END, scale_info)
                
            except ValueError:
                tk.messagebox.showerror("Error", "Please enter a valid fiber diameter.")
        
        # Calculate button
        calc_button = ttk.Button(input_frame, text="Calculate", command=calculate_estimates)
        calc_button.pack(pady=5)
        
        # Initial calculation
        calculate_estimates()
        
        # Close button
        close_button = ttk.Button(main_frame, text="Close", command=estimation_window.destroy)
        close_button.pack(pady=(10, 0))


def main():
    """Entry point for the GUI application."""
    import sys
    from pathlib import Path
    
    # Add project root to path if needed
    project_root = Path(__file__).parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    try:
        app = MainWindow()
        app.mainloop()
    except Exception as e:
        print(f"Error starting ESNF Mat Analyzer GUI: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
