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
from esnf_mat_analyzer.config.config_manager import get_default_config, save_config, load_config
import yaml


class MainWindow(tk.Tk):
    def __init__(self, user_config_path=None):
        super().__init__()
        self.title("Nanofiber Analyzer")
        self.geometry("1100x720")

        # Path where user-specific GUI tunables are saved
        self.user_config_path = user_config_path

        self.selected_files = []
        self.current_image = None
        self.displayed_image = None
        # Main container frame for left controls and right image preview
        self.main_frame = ttk.Frame(self)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # --- Create a scrollable left panel ---
        left_panel_container = ttk.Frame(self.main_frame)
        left_panel_container.pack(side=tk.LEFT, fill=tk.Y, padx=5)

        left_canvas = tk.Canvas(left_panel_container, borderwidth=0, highlightthickness=0)
        left_scrollbar = ttk.Scrollbar(left_panel_container, orient="vertical", command=left_canvas.yview)
        
        # This frame will contain all the widgets and be scrolled by the canvas
        self.left_panel = ttk.Frame(left_canvas)
        
        left_canvas.configure(yscrollcommand=left_scrollbar.set)

        left_scrollbar.pack(side="right", fill="y")
        left_canvas.pack(side="left", fill="both", expand=True)
        
        canvas_window = left_canvas.create_window((0, 0), window=self.left_panel, anchor="nw")

        def on_frame_configure(event):
            # Update scroll region to encompass the inner frame
            left_canvas.configure(scrollregion=left_canvas.bbox("all"))

        def on_canvas_configure(event):
            # Resize the inner frame to match the canvas width
            left_canvas.itemconfig(canvas_window, width=event.width)

        self.left_panel.bind("<Configure>", on_frame_configure)
        left_canvas.bind("<Configure>", on_canvas_configure)
        # --- End of scrollable left panel setup ---

        # File selection
        self.file_frame = ttk.LabelFrame(self.left_panel, text="File Selection")
        self.file_frame.pack(fill=tk.X, pady=5, padx=5)

        self.select_button = ttk.Button(
            self.file_frame, text="Select Files", command=self.select_files
        )
        self.select_button.pack(side=tk.TOP, padx=5, pady=5)

        # Help button (opens consolidated help dialog)
        self.help_button = ttk.Button(self.file_frame, text="Help", command=self.show_help)
        self.help_button.pack(side=tk.TOP, padx=5, pady=(0,6))

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

        # Background correction method selection
        self.bg_method_frame = ttk.LabelFrame(self.analysis_frame, text="Background Correction")
        self.bg_method_frame.pack(fill=tk.X, padx=5, pady=5)

        self.bg_method_var = tk.StringVar(value="none")
        bg_methods = [
            ("Default (None)", "none"),
            ("Polynomial Surface (Robust)", "basic"),
            ("Large Kernel Blur", "rolling_ball"),
            ("Complete Workflow", "restore"),
            ("Complete Workflow (Alt)", "homomorphic")
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

        # Recommend method button (uses composite quality metric)
        self.recommend_button = ttk.Button(
            self.bg_method_frame,
            text="Recommend Method",
            command=self.recommend_method_ui
        )
        self.recommend_button.pack(padx=5, pady=2)

        # --- Method Parameters (advanced) ---
        # Expose common tunable parameters per background method
        self.method_params_frame = ttk.LabelFrame(self.analysis_frame, text="Method Parameters (advanced)")
        self.method_params_frame.pack(fill=tk.X, padx=5, pady=5)

        # Initialize parameter variables from default config
        # Load defaults, then override with any persisted user config
        default_cfg = get_default_config()
        dp = default_cfg.processing
        if self.user_config_path:
            try:
                loaded = load_config(Path(self.user_config_path))
                # Copy known processing fields into dp
                dp.blur_kernel_size = loaded.processing.blur_kernel_size
                dp.basic_correction_n_components = loaded.processing.basic_correction_n_components
                dp.rolling_ball_radius = loaded.processing.rolling_ball_radius
                dp.restore_percentile = loaded.processing.restore_percentile
                dp.homomorphic_cutoff = loaded.processing.homomorphic_cutoff
                dp.homomorphic_g_low = loaded.processing.homomorphic_g_low
                dp.homomorphic_g_high = loaded.processing.homomorphic_g_high
            except Exception:
                # Ignore load errors and continue with defaults
                pass

        # Recommendation metric weights (dyn, sig, sat) — allow persistence via user config
        # Default weights mirror those used in recommendation.score_correction
        rec_dyn = 0.4
        rec_sig = 0.4
        rec_sat = 0.2
        if self.user_config_path:
            try:
                loaded = load_config(Path(self.user_config_path))
                # Expect a top-level 'recommendation' mapping in user config saved by this GUI
                if hasattr(loaded, 'recommendation') and getattr(loaded, 'recommendation') is not None:
                    rcfg = loaded.recommendation
                    rec_dyn = float(getattr(rcfg, 'dyn', rec_dyn)) if hasattr(rcfg, 'dyn') else rec_dyn
                    rec_sig = float(getattr(rcfg, 'sig', rec_sig)) if hasattr(rcfg, 'sig') else rec_sig
                    rec_sat = float(getattr(rcfg, 'sat', rec_sat)) if hasattr(rcfg, 'sat') else rec_sat
            except Exception:
                pass

        self.blur_kernel_var = tk.IntVar(value=dp.blur_kernel_size)
        self.basic_ncomp_var = tk.IntVar(value=dp.basic_correction_n_components)
        self.rolling_radius_var = tk.IntVar(value=dp.rolling_ball_radius)
        self.restore_percentile_var = tk.DoubleVar(value=dp.restore_percentile)
        self.homomorphic_cutoff_var = tk.DoubleVar(value=dp.homomorphic_cutoff)
        self.homomorphic_g_low_var = tk.DoubleVar(value=dp.homomorphic_g_low)
        self.homomorphic_g_high_var = tk.DoubleVar(value=dp.homomorphic_g_high)

        # Recommendation weight variables
        self.rec_dyn_var = tk.DoubleVar(value=rec_dyn)
        self.rec_sig_var = tk.DoubleVar(value=rec_sig)
        self.rec_sat_var = tk.DoubleVar(value=rec_sat)

        # Trace changes to variables to persist them automatically
        for var in (self.blur_kernel_var, self.basic_ncomp_var, self.rolling_radius_var,
                    self.restore_percentile_var, self.homomorphic_cutoff_var,
                    self.homomorphic_g_low_var, self.homomorphic_g_high_var,
                    self.rec_dyn_var, self.rec_sig_var, self.rec_sat_var):
            try:
                var.trace_add('write', lambda *args: self._save_user_params())
            except Exception:
                # Older tkinter versions may use trace; fallback
                try:
                    var.trace('w', lambda *args: self._save_user_params())
                except Exception:
                    pass

        # General blur kernel control (used by some preprocessing steps)
        self._params_general_frame = ttk.Frame(self.method_params_frame)
        ttk.Label(self._params_general_frame, text="Gaussian Blur Kernel (odd, 0=disabled):").pack(side=tk.LEFT, padx=5)
        ttk.Spinbox(self._params_general_frame, from_=0, to=101, increment=2, textvariable=self.blur_kernel_var, width=6).pack(side=tk.LEFT, padx=5)
        ttk.Label(self._params_general_frame, text="Blur smooths small noise; 0 disables. Use odd kernel sizes.", foreground="#444").pack(side=tk.LEFT, padx=8)

        # BASIC parameters
        self._params_basic_frame = ttk.Frame(self.method_params_frame)
        ttk.Label(self._params_basic_frame, text="BaSiC n_components:").pack(side=tk.LEFT, padx=5)
        ttk.Spinbox(self._params_basic_frame, from_=1, to=10, textvariable=self.basic_ncomp_var, width=6).pack(side=tk.LEFT, padx=5)
        ttk.Label(self._params_basic_frame, text="BaSiC components: more captures complex background but may remove signal.", foreground="#444").pack(side=tk.LEFT, padx=8)

        # Rolling ball parameters
        self._params_rolling_frame = ttk.Frame(self.method_params_frame)
        ttk.Label(self._params_rolling_frame, text="Rolling Ball Radius (px):").pack(side=tk.LEFT, padx=5)
        ttk.Spinbox(self._params_rolling_frame, from_=1, to=1000, textvariable=self.rolling_radius_var, width=6).pack(side=tk.LEFT, padx=5)
        ttk.Label(self._params_rolling_frame, text="Rolling-ball radius (px): larger removes broader illumination.", foreground="#444").pack(side=tk.LEFT, padx=8)

        # RESTORE parameters
        self._params_restore_frame = ttk.Frame(self.method_params_frame)
        ttk.Label(self._params_restore_frame, text="RESTORE Percentile (0-100):").pack(side=tk.LEFT, padx=5)
        ttk.Spinbox(self._params_restore_frame, from_=0.1, to=99.9, increment=0.1, textvariable=self.restore_percentile_var, width=6).pack(side=tk.LEFT, padx=5)
        ttk.Label(self._params_restore_frame, text="RESTORE percentile tunes background/foreground bias; start small.", foreground="#444").pack(side=tk.LEFT, padx=8)

        # Homomorphic parameters
        self._params_homomorphic_frame = ttk.Frame(self.method_params_frame)
        ttk.Label(self._params_homomorphic_frame, text="Homomorphic Cutoff:").pack(side=tk.LEFT, padx=5)
        ttk.Spinbox(self._params_homomorphic_frame, from_=1, to=1000, textvariable=self.homomorphic_cutoff_var, width=6).pack(side=tk.LEFT, padx=5)
        ttk.Label(self._params_homomorphic_frame, text="g_low:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(self._params_homomorphic_frame, textvariable=self.homomorphic_g_low_var, width=6).pack(side=tk.LEFT, padx=2)
        ttk.Label(self._params_homomorphic_frame, text="g_high:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(self._params_homomorphic_frame, textvariable=self.homomorphic_g_high_var, width=6).pack(side=tk.LEFT, padx=2)
        ttk.Label(self._params_homomorphic_frame, text="Cutoff/gain control frequency filtering; adjust to balance contrast/noise.", foreground="#444").pack(side=tk.LEFT, padx=8)

        # Initially show general params only
        self._show_method_params()

        # Add a small recommendation weights frame below method parameters
        self._params_recommend_frame = ttk.Frame(self.method_params_frame)
        ttk.Label(self._params_recommend_frame, text="Recommendation weights (dyn/sig/sat):").pack(side=tk.LEFT, padx=5)
        # Use scales to allow user-friendly tuning
        ttk.Scale(self._params_recommend_frame, from_=0.0, to=1.0, orient=tk.HORIZONTAL, variable=self.rec_dyn_var, length=120).pack(side=tk.LEFT, padx=3)
        ttk.Scale(self._params_recommend_frame, from_=0.0, to=1.0, orient=tk.HORIZONTAL, variable=self.rec_sig_var, length=120).pack(side=tk.LEFT, padx=3)
        ttk.Scale(self._params_recommend_frame, from_=0.0, to=1.0, orient=tk.HORIZONTAL, variable=self.rec_sat_var, length=120).pack(side=tk.LEFT, padx=3)
        ttk.Button(self._params_recommend_frame, text="Normalize", command=lambda: self._normalize_recommend_weights()).pack(side=tk.LEFT, padx=5)
        self._params_recommend_frame.pack(fill=tk.X, padx=5, pady=4)
        ttk.Label(self.method_params_frame, text="Recommendation weights control the recommender trade-offs (dyn/sig/sat). Use Normalize to sum to 1.", foreground="#444").pack(anchor=tk.W, padx=8, pady=(0,4))

        # Thickness model selection
        self.thickness_model_frame = ttk.LabelFrame(self.analysis_frame, text="Thickness Model")
        self.thickness_model_frame.pack(fill=tk.X, padx=5, pady=5)

        self.thickness_model_var = tk.StringVar(value="beer_lambert")
        thickness_models = [
            ("Beer-Lambert (Physics)", "beer_lambert"),
            ("Linear (Legacy)", "linear"),
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

        # ROI shape selection (rectangle or circle)
        self.roi_shape_var = tk.StringVar(value="rectangle")
        roi_shape_frame = ttk.Frame(self.image_controls_frame)
        roi_shape_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(roi_shape_frame, text="ROI Shape:").pack(side=tk.LEFT, padx=4)
        ttk.Radiobutton(roi_shape_frame, text="Rectangle", variable=self.roi_shape_var, value="rectangle").pack(side=tk.LEFT, padx=4)
        ttk.Radiobutton(roi_shape_frame, text="Circle", variable=self.roi_shape_var, value="circle").pack(side=tk.LEFT, padx=4)

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
        """Populate the file listbox from `self.selected_files`."""
        try:
            self.file_listbox.delete(0, tk.END)
            for p in self.selected_files:
                try:
                    self.file_listbox.insert(tk.END, Path(p).name)
                except Exception:
                    self.file_listbox.insert(tk.END, str(p))
        except Exception:
            # If listbox isn't ready yet, ignore
            pass

    def on_file_select(self, event):
        """Handle when user selects a file from the listbox.

        This loads the selected image into `self.current_image` (PIL.Image)
        and resets zoom/ROI state so the user can interact with it.
        """
        try:
            sel = self.file_listbox.curselection()
            if not sel:
                return
            index = sel[0]
            filepath = self.selected_files[index]

            # Load image using PIL
            try:
                pil_img = Image.open(filepath)
                # Keep as RGB for consistent handling (convert if code expects L later)
                pil_img = pil_img.convert('RGB')
            except Exception:
                # Fallback: try converting via OpenCV then to PIL
                try:
                    arr = cv2.imread(str(filepath), cv2.IMREAD_UNCHANGED)
                    if arr is None:
                        raise IOError("Failed to read image")
                    arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(arr)
                except Exception:
                    messagebox.showerror("Open Image", f"Failed to open image: {filepath}")
                    return

            self.current_image = pil_img

            # Initialize zoom and view state
            self.scale_factor = 1.0
            self.min_scale = 0.1
            self.max_scale = 10.0
            self.is_panning = False
            self.drawing_mode = 'roi'
            self.rect = None
            self.oval = None

            # Reset spatial scale if not set
            if not hasattr(self, 'spatial_scale'):
                self.spatial_scale = None

            # Update the display
            try:
                self.update_image_display()
            except Exception:
                pass

        except Exception as e:
            print(f"on_file_select failed: {e}")

        

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

    def fit_to_window(self):
        """Scale the image so it fits within the canvas viewport."""
        if not self.current_image:
            return
        try:
            c_w = self.canvas.winfo_width() or 1
            c_h = self.canvas.winfo_height() or 1
            img_w, img_h = self.current_image.size
            # Compute scale to fit both dimensions
            scale_x = c_w / img_w
            scale_y = c_h / img_h
            new_scale = min(scale_x, scale_y, 1.0)
            # Clamp to min/max if present
            if not hasattr(self, 'min_scale'):
                self.min_scale = 0.1
            if not hasattr(self, 'max_scale'):
                self.max_scale = 10.0
            new_scale = max(self.min_scale, min(new_scale, self.max_scale))
            self.scale_factor = new_scale
            self.update_image_display()
        except Exception:
            pass

    def reset_zoom(self):
        """Reset zoom to 100% (scale factor 1.0)."""
        try:
            self.scale_factor = 1.0
            if not hasattr(self, 'min_scale'):
                self.min_scale = 0.1
            if not hasattr(self, 'max_scale'):
                self.max_scale = 10.0
            self.update_image_display()
        except Exception:
            pass

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
            # Create rectangular or circular ROI depending on selection
            if self.roi_shape_var.get() == "rectangle":
                if self.rect:
                    self.canvas.delete(self.rect)
                self.rect = self.canvas.create_rectangle(
                    self.start_x, self.start_y, self.start_x, self.start_y,
                    outline="red", width=2, tags="roi"
                )
            else:
                # circle ROI uses an oval canvas item
                if self.oval:
                    self.canvas.delete(self.oval)
                self.oval = self.canvas.create_oval(
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
            if self.roi_shape_var.get() == "rectangle":
                if self.rect:
                    self.canvas.coords(self.rect, self.start_x, self.start_y, canvas_x, canvas_y)
            else:
                if self.oval:
                    self.canvas.coords(self.oval, self.start_x, self.start_y, canvas_x, canvas_y)
        elif self.drawing_mode == "scale":
            if self.oval:
                self.canvas.coords(self.oval, self.start_x, self.start_y, canvas_x, canvas_y)

    def on_button_release(self, event):
        if self.is_panning:
            return

        if self.drawing_mode == "scale":
            self.calculate_scale_from_oval()

        # After finishing any draw action, return to ROI-drawing mode
        self.drawing_mode = "roi"

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
        config = get_default_config()
        
        # Update background leveling setting based on checkbox
        config.processing.leveling.enabled = self.background_leveling_var.get()
        
        analyzer = setup_dependencies(config)

        # Run analysis
        try:
            self.last_analysis_result = analyzer.process_image(Path(filepath), roi=roi, spatial_scale_pixels_per_mm=self.spatial_scale)
            self.display_results(self.last_analysis_result)
        except Exception as e:
            print(f"Error during analysis: {e}")

    def show_heatmap(self):
        if not self.last_analysis_result:
            # Provide a clear popup to the user when no analysis has been run
            try:
                messagebox.showerror("No Analysis", "Please run an analysis first.")
            except Exception:
                # Fallback to console if messagebox fails for some reason
                print("Please run an analysis first.")
            return

        # Create a visualizer with custom configuration
        vis_config = VisualizationConfig()
        vis_config.auto_range_heatmap = self.auto_range_var.get()
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
            
        try:
            # Convert to RGB mode to ensure compatibility
            if img.mode != 'RGB':
                img = img.convert('RGB')
            # Save without EXIF data
            img.save(filepath, format='PNG' if filepath.lower().endswith('.png') else 'JPEG')
            print(f"Image saved to {filepath}")
        except Exception as e:
            print(f"Export failed: {e}")
            # Try alternative save method
            try:
                img_array = np.array(img)
                cv2.imwrite(filepath, cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
                print(f"Image saved to {filepath} using alternative method")
            except Exception as e2:
                print(f"Alternative export failed: {e2}")
                messagebox.showerror("Export Error", f"Failed to save image: {e2}")

    def on_bg_method_change(self):
        """Called when background correction method is changed."""
        # Update visible method-specific parameters when background method changes
        self._show_method_params()

    def _show_method_params(self):
        """Show/hide parameter frames depending on selected background method."""
        # Clear existing packed frames
        for f in (self._params_general_frame,
                  self._params_basic_frame,
                  self._params_rolling_frame,
                  self._params_restore_frame,
                  self._params_homomorphic_frame):
            try:
                f.pack_forget()
            except Exception:
                pass

        # Always show general params
        self._params_general_frame.pack(fill=tk.X, padx=5, pady=2)

        method = self.bg_method_var.get()
        if method == "basic":
            self._params_basic_frame.pack(fill=tk.X, padx=5, pady=2)
        elif method == "rolling_ball":
            self._params_rolling_frame.pack(fill=tk.X, padx=5, pady=2)
        elif method == "restore":
            self._params_restore_frame.pack(fill=tk.X, padx=5, pady=2)
        elif method == "homomorphic":
            self._params_homomorphic_frame.pack(fill=tk.X, padx=5, pady=2)

    def _save_user_params(self):
        """Save current method parameter values to user_config_path if provided."""
        if not self.user_config_path:
            return

        try:
            cfg = get_default_config()
            p = cfg.processing
            # Write current GUI-controlled params
            p.blur_kernel_size = int(self.blur_kernel_var.get())
            p.basic_correction_n_components = int(self.basic_ncomp_var.get())
            p.rolling_ball_radius = int(self.rolling_radius_var.get())
            p.restore_percentile = float(self.restore_percentile_var.get())
            p.homomorphic_cutoff = float(self.homomorphic_cutoff_var.get())
            p.homomorphic_g_low = float(self.homomorphic_g_low_var.get())
            p.homomorphic_g_high = float(self.homomorphic_g_high_var.get())

            # Persist core processing config
            save_config(cfg, Path(self.user_config_path))

            # Additionally persist recommendation weights to a small companion YAML mapping
            try:
                rec_cfg = {
                    'dyn': float(self.rec_dyn_var.get()),
                    'sig': float(self.rec_sig_var.get()),
                    'sat': float(self.rec_sat_var.get())
                }
                # Write a lightweight YAML file alongside the config to avoid modifying core schema
                ppath = Path(self.user_config_path)
                companion = ppath.parent / (ppath.stem + ".recommendation.yaml")
                with open(companion, 'w') as f:
                    yaml.safe_dump({'recommendation': rec_cfg}, f)
            except Exception:
                # ignore companion write errors
                pass

        except Exception:
            # Do not raise in UI; log if needed
            pass

    def preview_background_methods(self):
        """Show a comparison of different background correction methods."""
        if not self.current_image:
            tk.messagebox.showwarning("Warning", "Please select an image first.")
            return

        # Robustly convert current image (PIL or numpy) to grayscale.
        try:
            if isinstance(self.current_image, np.ndarray):
                img_np = self.current_image
                # If already grayscale
                if img_np.ndim == 2:
                    gray_image = img_np
                else:
                    # Drop alpha channel if present
                    if img_np.shape[2] == 4:
                        img_np = img_np[..., :3]
                    # Try converting assuming RGB; fall back to BGR if needed
                    try:
                        gray_image = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
                    except Exception:
                        gray_image = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
            else:
                # Convert PIL images to RGB first to handle RGBA, P, CMYK, etc.
                pil_rgb = self.current_image.convert('RGB')
                img_np = np.array(pil_rgb)
                gray_image = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        except Exception as e:
            tk.messagebox.showerror("Error", f"Unsupported image format for preview: {e}")
            return

        # Defensive: check for empty image
        if gray_image is None or gray_image.size == 0:
            tk.messagebox.showerror("Error", "Failed to load image for preview. Image is empty.")
            return

        # Create preview window
        preview_window = tk.Toplevel(self)
        preview_window.title("Background Correction Methods Comparison")
        preview_window.geometry("1400x800")

        # Create notebook for tabs
        notebook = ttk.Notebook(preview_window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Define methods to test
        methods = [
            ("Default (Normalization)", "none", self._apply_default_correction),
            ("BASIC (Robust)", "basic", self._apply_basic_correction),
            ("Rolling Ball", "rolling_ball", self._apply_rolling_ball_correction),
            ("RESTORE", "restore", self._apply_restore_correction),
            ("Homomorphic", "homomorphic", self._apply_homomorphic_correction)
        ]

        for method_name, method_id, method_func in methods:
            # Create tab for this method
            tab_frame = ttk.Frame(notebook)
            notebook.add(tab_frame, text=method_name)

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
        from esnf_mat_analyzer.processing.background_correction.advanced import AdvancedBackgroundProcessor
        processor = AdvancedBackgroundProcessor()
        return processor.basic_correction(image, 1)

    def _apply_rolling_ball_correction(self, image):
        """Apply rolling ball correction."""
        from esnf_mat_analyzer.processing.background_correction.advanced import AdvancedBackgroundProcessor
        processor = AdvancedBackgroundProcessor()
        return processor.rolling_ball_3d(image, 50)

    def _apply_restore_correction(self, image):
        """Apply RESTORE correction."""
        from esnf_mat_analyzer.processing.background_correction.advanced import AdvancedBackgroundProcessor
        processor = AdvancedBackgroundProcessor()
        return processor.restore_method(image, 5.0)

    def _apply_homomorphic_correction(self, image):
        """Apply homomorphic correction."""
        from esnf_mat_analyzer.processing.background_correction.advanced import AdvancedBackgroundProcessor
        processor = AdvancedBackgroundProcessor()
        return processor.homomorphic_filter(image, 30, 0.5, 2.0)

    def recommend_method_ui(self):
        """Run a lightweight recommendation across available methods and show suggestion."""
        try:
            if not self.current_image:
                tk.messagebox.showwarning("Warning", "Please select an image first.")
                return

            # Convert PIL to grayscale numpy
            img_np = np.array(self.current_image.convert('L'))

            # Use session cache to avoid recomputing on repeated clicks
            if not hasattr(self, '_recommend_cache'):
                self._recommend_cache = {}

            cache_key = (img_np.shape, img_np.mean())
            if cache_key in self._recommend_cache:
                best, scores = self._recommend_cache[cache_key]
            else:
                # Show a tiny progress window
                prog = tk.Toplevel(self)
                prog.title('Recommending...')
                ttk.Label(prog, text='Computing recommendation, please wait...').pack(padx=10, pady=10)
                prog.update_idletasks()

                # Build methods dict (use the same small functions as preview)
                methods = {
                    'none': self._apply_default_correction,
                    'basic': self._apply_basic_correction,
                    'rolling_ball': self._apply_rolling_ball_correction,
                    'restore': self._apply_restore_correction,
                    'homomorphic': self._apply_homomorphic_correction
                }

                from esnf_mat_analyzer.processing.background_correction.recommendation import recommend_method

                # Build weights from GUI variables
                weights = {
                    'dyn': float(self.rec_dyn_var.get()),
                    'sig': float(self.rec_sig_var.get()),
                    'sat': float(self.rec_sat_var.get())
                }

                best, scores = recommend_method(img_np, methods, weights=weights, downsample=max(1, min(4, img_np.shape[0] // 128)))
                self._recommend_cache[cache_key] = (best, scores)

                try:
                    prog.destroy()
                except Exception:
                    pass

            if not best:
                tk.messagebox.showinfo("Recommendation", "No recommendation could be made.")
                return

            # Show scores in a simple messagebox using friendly labels
            try:
                from esnf_mat_analyzer.processing.background_correction.recommendation import id_to_friendly
            except Exception:
                def id_to_friendly(x):
                    return x

            text = f"Recommended method: {id_to_friendly(best)}\n\nScores:\n"
            for k, v in sorted(scores.items(), key=lambda kv: kv[1], reverse=True):
                text += f"  {id_to_friendly(k)}: {v:.3f}\n"

            tk.messagebox.showinfo("Background Method Recommendation", text)

        except Exception as e:
            tk.messagebox.showerror("Recommendation Error", f"Recommendation failed: {e}")

    def _normalize_recommend_weights(self):
        """Normalize the recommendation weight sliders so they sum to 1.0 (if possible)."""
        try:
            vals = [float(self.rec_dyn_var.get()), float(self.rec_sig_var.get()), float(self.rec_sat_var.get())]
            s = sum(vals)
            if s <= 0:
                # reset to defaults
                self.rec_dyn_var.set(0.4)
                self.rec_sig_var.set(0.4)
                self.rec_sat_var.set(0.2)
                return
            self.rec_dyn_var.set(vals[0] / s)
            self.rec_sig_var.set(vals[1] / s)
            self.rec_sat_var.set(vals[2] / s)
        except Exception:
            pass

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
        """Get ROI coordinates from canvas rectangle."""
        # Support rectangle and circle ROI shapes. Return bounding box (orig coords)
        if self.roi_shape_var.get() == "rectangle":
            if not self.rect:
                return None
            x1, y1, x2, y2 = self.canvas.coords(self.rect)
        else:
            if not self.oval:
                return None
            x1, y1, x2, y2 = self.canvas.coords(self.oval)

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

        return (orig_x1, orig_y1, orig_x2, orig_y2)

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
                "basic": BackgroundCorrectionMethod.POLYNOMIAL_SURFACE,
                "rolling_ball": BackgroundCorrectionMethod.LARGE_KERNEL_BLUR,
                "restore": BackgroundCorrectionMethod.COMPLETE_WORKFLOW,
                "homomorphic": BackgroundCorrectionMethod.COMPLETE_WORKFLOW  # Using complete workflow as fallback
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
            # Apply GUI-tunable processing parameters
            try:
                config.processing.blur_kernel_size = int(self.blur_kernel_var.get())
            except Exception:
                pass
            try:
                config.processing.basic_correction_n_components = int(self.basic_ncomp_var.get())
            except Exception:
                pass
            try:
                config.processing.rolling_ball_radius = int(self.rolling_radius_var.get())
            except Exception:
                pass
            try:
                config.processing.restore_percentile = float(self.restore_percentile_var.get())
            except Exception:
                pass
            try:
                config.processing.homomorphic_cutoff = float(self.homomorphic_cutoff_var.get())
                config.processing.homomorphic_g_low = float(self.homomorphic_g_low_var.get())
                config.processing.homomorphic_g_high = float(self.homomorphic_g_high_var.get())
            except Exception:
                pass

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
            # First attempt: let analyzer try to auto-detect ruler if spatial scale not provided
            self.last_analysis_result = analyzer.process_image(
                Path(filepath), roi=roi, spatial_scale_pixels_per_mm=self.spatial_scale
            )

            # If ruler detection failed and config expects ruler detection, prompt user for manual scale
            if (self.last_analysis_result.spatial_scale_pixels_per_mm is None
                    and getattr(config, 'ruler_detection', None) is not None
                    and config.ruler_detection.enabled):
                # Ask user for manual scale (pixels per mm)
                manual_scale = simpledialog.askfloat(
                    "Manual Spatial Scale",
                    "Ruler not detected automatically. Enter spatial scale in pixels/mm (or Cancel to continue without scale):",
                    minvalue=0.0
                )

                if manual_scale and manual_scale > 0:
                    # Re-run analysis with manual scale
                    if not self.background_leveling_var.get():
                        # Ensure settings are applied consistently
                        config.processing.leveling.enabled = self.background_leveling_var.get()
                    try:
                        self.last_analysis_result = analyzer.process_image(
                            Path(filepath), roi=roi, spatial_scale_pixels_per_mm=float(manual_scale)
                        )
                        print(f"Analysis re-run with manual scale: {manual_scale} pixels/mm")
                    except Exception as e:
                        print(f"Re-run with manual scale failed: {e}")
                        messagebox.showerror("Analysis Error", f"Re-run with manual scale failed: {e}")
                else:
                    # User cancelled or entered invalid value; notify and continue without scale
                    messagebox.showwarning("Scale Missing", "No spatial scale set. Results will lack real-world calibration.")

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
            # Convert the image to RGB
            pil_image = Image.fromarray(cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB))
            # Save without EXIF data
            pil_image.save(filepath, format='PNG' if filepath.lower().endswith('.png') else 'JPEG')
            print(f"Image saved to {filepath}")
        except Exception as e:
            print(f"Export failed: {e}")
            # Try alternative save method using OpenCV
            try:
                cv2.imwrite(filepath, self.current_image)
                print(f"Image saved to {filepath} using alternative method")
            except Exception as e2:
                print(f"Alternative export failed: {e2}")
                messagebox.showerror("Export Error", f"Failed to save image: {e2}")

    def display_results(self, result):
        # Create a new window to display results
        results_window = tk.Toplevel(self)
        results_window.title("Analysis Results")
        results_window.geometry("800x600")
        
        # Debug log to check result content
        print(f"Result metrics: {result.metrics}")
        print(f"Traditional metrics: {[k for k, v in result.metrics.items() if not k.startswith(('anisotropy_', 'texture_', 'psd_', 'overall_mat_uniformity'))]}")
        print(f"Mat analysis metrics: {[k for k, v in result.metrics.items() if k.startswith(('anisotropy_', 'texture_', 'psd_', 'overall_mat_uniformity'))]}")

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
                             if not k.startswith(('anisotropy_', 'texture_', 'psd_', 'overall_mat_uniformity', 'scale_'))}
        for name, value in traditional_metrics.items():
            if isinstance(value, float) and not math.isnan(value):
                basic_result += f"  {name}: {value:.4f}\n"
            else:
                basic_result += f"  {name}: {value}\n"

        # Add Multi-Scale Analysis to Basic Metrics with proper formatting
        scale_metrics = {k: v for k, v in result.metrics.items() if k.startswith('scale_')}
        if scale_metrics:
            basic_result += "\nMulti-Scale Uniformity Analysis:\n"
            basic_result += "-----------------------------------\n"
            scale_names = {
                'scale_0_uniformity': 'Fiber-Level Uniformity (Local)',
                'scale_1_uniformity': 'Small Cluster Uniformity',
                'scale_2_uniformity': 'Medium Cluster Uniformity',
                'scale_3_uniformity': 'Large Region Uniformity',
                'scale_4_uniformity': 'Global Mat Uniformity'
            }
            for name in sorted(scale_metrics.keys()):
                value = scale_metrics[name]
                display_name = scale_names.get(name, name.replace('_', ' ').title())
                if isinstance(value, float) and not math.isnan(value):
                    basic_result += f"  {display_name}: {value:.4f}\n"
                else:
                    basic_result += f"  {display_name}: {value}\n"
            basic_result += "\nNote: Values range from 0-1, higher values indicate better uniformity at each scale.\n"
        
        basic_text.insert(tk.END, basic_result)
        basic_text.config(state=tk.DISABLED)

        # Mat-Scale Analysis Tab
        mat_frame = ttk.Frame(notebook)
        notebook.add(mat_frame, text="Mat-Scale Analysis")
        
        mat_scroll = ttk.Scrollbar(mat_frame)
        mat_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        mat_text = tk.Text(mat_frame, wrap=tk.WORD, yscrollcommand=mat_scroll.set)
        mat_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        mat_scroll.config(command=mat_text.yview)

        # Format mat-scale results
        mat_result = "Mat-Scale Uniformity Analysis\n"
        mat_result += "=" * 40 + "\n\n"
        
        # Debug print metrics for troubleshooting
        print("Debug - Available metrics:", result.metrics.keys())
        
        # Overall mat uniformity score
        if 'overall_mat_uniformity' in result.metrics:
            score = result.metrics['overall_mat_uniformity']
            if isinstance(score, float) and not math.isnan(score):
                mat_result += f"Overall Mat Uniformity Score: {score:.4f}\n"
                mat_result += f"Uniformity Rating: {self._get_uniformity_rating(score)}\n\n"
                
        # Print all available metric keys for debugging
        print("\nAvailable Metrics:")
        for key in result.metrics.keys():
            print(f"- {key}")
        
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
            mat_result += "\n"  # Add extra newline for spacing
        
        # Uniformity Indexes Section
        mat_result += "\nUniformity Indexes:\n"
        mat_result += "-" * 35 + "\n"
        
        # Primary indicators section
        mat_result += "Primary Uniformity Indicators:\n"
        
        # Add Anisotropy Index
        anisotropy_value = result.metrics.get('anisotropy_index', float('nan'))
        if isinstance(anisotropy_value, float) and not math.isnan(anisotropy_value):
            mat_result += f"  Anisotropy Index: {anisotropy_value:.4f} (0-1, lower is better)\n"

        # Add Overall Mat Uniformity
        overall_value = result.metrics.get('overall_mat_uniformity', float('nan'))
        if isinstance(overall_value, float) and not math.isnan(overall_value):
            mat_result += f"  Overall Mat Uniformity: {overall_value:.4f} (0-1, higher is better)\n"

        # Add Texture-based metrics with clear labels
        mat_result += "\nTexture-based Uniformity:\n"
        texture_metrics = {
            'texture_homogeneity': ('Homogeneity', '0-1, higher is better'),
            'texture_energy': ('Energy', '0-1, higher is better'),
            'texture_correlation': ('Correlation', '-1 to 1'),
            'texture_contrast': ('Contrast', '0+, lower is better')
        }
        
        for metric_key, (display_name, range_info) in texture_metrics.items():
            value = result.metrics.get(metric_key, float('nan'))
            if isinstance(value, float) and not math.isnan(value):
                mat_result += f"  {display_name}: {value:.4f} ({range_info})\n"

        # Add PSD Uniformity
        mat_result += "\nFrequency Analysis:\n"
        psd_value = result.metrics.get('psd_uniformity', float('nan'))
        if isinstance(psd_value, float) and not math.isnan(psd_value):
            mat_result += f"  PSD Uniformity: {psd_value:.4f} (0-1, higher is better)\n"

        mat_result += "\n"

        # Add detailed interpretation guide (metric definitions, ranges, heuristics)
        mat_result += "=" * 40 + "\n"
        mat_result += "Interpretation Guide (per-metric):\n\n"

        # Anisotropy group
        mat_result += "Anisotropy (FFT-based)\n"
        mat_result += "- anisotropy_index (0 - 1): 0 = perfectly isotropic; higher values indicate stronger directional alignment.\n"
        mat_result += "  Suggested heuristics: <= 0.15 (low anisotropy/isotropic), 0.15-0.30 (moderate), > 0.30 (strong alignment).\n"
        mat_result += "- anisotropy_preferred_angle_degrees (0 - 180): principal orientation of aligned features in degrees.\n\n"

        # Texture group
        mat_result += "Texture (GLCM and related)\n"
        mat_result += "- texture_homogeneity (0 - 1): higher is more uniform / less local variation. Typical desirable: > 0.7.\n"
        mat_result += "- texture_energy (0 - 1): measures order; higher values indicate repeated/regular structure.\n"
        mat_result += "- texture_contrast (0+): measures local intensity differences; lower values indicate smoother texture.\n"
        mat_result += "  Note: scales and image bit-depth affect absolute contrast values — compare within a dataset.\n"
        mat_result += "- texture_correlation (-1 to 1): measures linear dependency between neighboring pixels; values near 0 indicate no strong linear relation.\n\n"

        # PSD group
        mat_result += "Frequency / PSD (Power Spectral Density)\n"
        mat_result += "- psd_uniformity (0 - 1): higher values indicate a more even spread of power across spatial frequencies (perceived as uniform).\n"
        mat_result += "  Suggested heuristics: >= 0.6 often indicates good frequency-uniform mats; < 0.4 may show strong periodic structure or defects.\n"
        mat_result += "- psd_dominant_frequency: frequency (units = cycles/mm if spatial_scale provided, otherwise cycles/pixel) of the strongest spectral peak.\n"
        mat_result += "- psd_periodicity_index: higher values indicate stronger periodic/regular patterns. Use together with dominant_frequency to detect repeating structures.\n\n"

        # Multiscale uniformity
        mat_result += "Multi-scale Uniformity\n"
        mat_result += "- scale_*_uniformity (0 - 1): per-scale uniformity measures from local (fiber-level) to global (whole-mat).\n"
        mat_result += "  Interpretation: values closer to 1 indicate more uniformity at that spatial scale. Use the per-scale profile to identify at which length-scale non-uniformity appears.\n\n"

        # Overall and traditional metrics
        mat_result += "Overall / Traditional Metrics\n"
        mat_result += "- overall_mat_uniformity (0 - 1): composite score combining multiple features. Higher is better. Thresholds: >=0.8 Excellent; 0.6-0.8 Good; 0.4-0.6 Fair; <0.4 Poor.\n"
        mat_result += "- GiniCoefficient (0 - 1): measures inequality of thickness distribution. Higher values = more unequal (worse uniformity).\n"
        mat_result += "- RadialUniformityIndex (0 - 1): radial uniformity; higher means more radially even thickness.\n"
        mat_result += "- Thickness range / CV: report of dispersion in thickness (mm or pixels). Lower coefficient of variation (CV) indicates more uniform thickness.\n\n"

        # Practical notes
        mat_result += "Practical Notes and Recommendations:\n"
        mat_result += "- Units: PSD-related frequencies require a spatial scale (pixels/mm) to convert to cycles/mm; set scale for absolute frequency units.\n"
        mat_result += "- Comparison: many metric thresholds are dataset-dependent — derive thresholds from a labeled representative dataset when possible.\n"
        mat_result += "- ROI shape: the GUI currently passes the ROI bounding box to analyzers. For circular ROI accuracy, consider enabling a true circular mask (future improvement).\n"
        mat_result += "- Use multi-metric decision: combine anisotropy, PSD, and multiscale uniformity to detect alignment, periodic defects, and local heterogeneity.\n"
        mat_result += "- Visual validation: always inspect heatmaps and corrected images alongside numeric scores; metrics are complements, not replacements, for visual QC.\n\n"

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

    def show_help(self):
        """Show a consolidated help dialog explaining controls and parameters."""
        help_text = (
            "File Selection:\n"
            "  - Select Files: choose one or more images to load into the viewer.\n\n"
            "Background Correction Controls:\n"
            "  - Preview Methods: open side-by-side previews for each correction method.\n"
            "  - Recommend Method: compute a recommendation using tunable weights (dyn/sig/sat).\n\n"
            "Method Parameters (advanced):\n"
            "  - Gaussian Blur Kernel: odd integer kernel; 0 disables blur. Small kernels remove noise.\n"
            "  - BaSiC n_components: number of basis components for polynomial/NMF correction; higher may remove background but risk removing signal.\n"
            "  - Rolling Ball Radius: radius in pixels for the large-kernel background estimator; larger removes broader illumination.\n"
            "  - RESTORE Percentile: tunes background/foreground bias in RESTORE method; start with small values.\n"
            "  - Homomorphic cutoff/gains: frequency cutoff and low/high gains for homomorphic filtering; adjust to balance contrast vs noise.\n\n"
            "Recommendation weights:\n"
            "  - dyn/sig/sat: weights for dynamic-range preservation, signal preservation (ROI), and saturation reduction. Use Normalize to sum to 1.\n\n"
            "Scale & Export:\n"
            "  - Set Scale Manually: draw a circle over a known distance and enter that distance (mm) to compute pixels/mm.\n"
            "  - Insert Scale Bar: adds a draggable scale bar to the preview (requires spatial scale).\n"
            "  - Export Image: save the current view (image or heatmap) to disk.\n\n"
            "Tips:\n"
            "  - Use Preview Methods before committing to a correction.\n"
            "  - Tune parameters conservatively and inspect results.\n"
        )

        try:
            help_win = tk.Toplevel(self)
            help_win.title("Help")
            help_win.geometry("600x500")
            txt = tk.Text(help_win, wrap=tk.WORD)
            txt.insert(tk.END, help_text)
            txt.config(state=tk.DISABLED)
            txt.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        except Exception:
            # Fallback to messagebox if Toplevel fails
            messagebox.showinfo("Help", help_text)


if __name__ == "__main__":
    app = MainWindow()
    app.mainloop()
