import os
import threading
import sys
print(sys.executable)
import customtkinter as ctk
from tkinter import filedialog, messagebox

# Set appearance mode and color theme for a professional UI
ctk.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
ctk.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"


def trim_video(
    input_path: str,
    clip_duration: int = 10,
    output_dir: str = "./trimmed_clips",
    start_time: float = 0,
    end_time: float = None,
    max_clips: int = None,
    prefix: str = "clip",
    log_callback=print
):
    """Core function to trim video."""
    try:
        from moviepy.editor import VideoFileClip
    except Exception as e:
        log_callback(f"[ERROR] MoviePy import failed: {e}")
        return False

    if not os.path.isfile(input_path):
        log_callback(f"[ERROR] Input file not found: {input_path}")
        return False

    os.makedirs(output_dir, exist_ok=True)

    log_callback(f"[INFO] Loading video: {input_path}")
    try:
        video = VideoFileClip(input_path)
    except Exception as e:
        log_callback(f"[ERROR] Failed to load video: {e}")
        return False

    total_duration = video.duration

    # Clamp start/end to valid range
    start_time = max(0, start_time)
    if end_time is None or end_time <= 0 or end_time > total_duration:
        end_time = total_duration

    if start_time >= end_time:
        log_callback("[ERROR] Start time must be less than end time.")
        video.close()
        return False

    log_callback(f"[INFO] Video duration  : {total_duration:.2f}s")
    log_callback(f"[INFO] Trimming range  : {start_time:.2f}s → {end_time:.2f}s")
    log_callback(f"[INFO] Clip length     : {clip_duration}s")
    log_callback(f"[INFO] Output directory: {output_dir}")
    log_callback("-" * 40)

    current = start_time
    clip_index = 1
    saved_clips = []

    while current < end_time:
        if max_clips is not None and max_clips > 0 and clip_index > max_clips:
            break

        clip_end = min(current + clip_duration, end_time)

        # Skip clips that are too short (< 1 second)
        if clip_end - current < 1:
            break

        # Determine output file extension from input
        ext = os.path.splitext(input_path)[1] or ".mp4"
        out_filename = f"{prefix}_{clip_index:03d}{ext}"
        out_path = os.path.join(output_dir, out_filename)

        log_callback(f"[CLIP {clip_index:03d}] {current:.2f}s → {clip_end:.2f}s  →  {out_filename}")

        try:
            subclip = video.subclip(current, clip_end)
            subclip.write_videofile(
                out_path,
                codec="libx264",
                audio_codec="aac",
                logger=None,          # suppress verbose ffmpeg output
                verbose=False,
            )
            subclip.close()
            saved_clips.append(out_path)
        except Exception as e:
            log_callback(f"[WARNING] Failed to write clip {clip_index}: {e}")

        current += clip_duration
        clip_index += 1

    video.close()
    log_callback("-" * 40)
    log_callback(f"[DONE] {len(saved_clips)} clip(s) saved.")
    return True


class VideoTrimmerApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Professional Video Trimmer")
        self.geometry("900x650")
        self.minsize(800, 550)
        
        # Configure grid layout (1 row, 2 columns)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- Sidebar Frame ---
        self.sidebar_frame = ctk.CTkFrame(self, width=220, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(5, weight=1)
        
        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="Trimmer Pro", font=ctk.CTkFont(size=24, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(30, 20))
        
        self.appearance_mode_label = ctk.CTkLabel(self.sidebar_frame, text="Appearance Mode:", anchor="w")
        self.appearance_mode_label.grid(row=1, column=0, padx=20, pady=(10, 0))
        
        self.appearance_mode_menu = ctk.CTkOptionMenu(
            self.sidebar_frame, 
            values=["System", "Dark", "Light"],
            command=self.change_appearance_mode_event,
            fg_color="#4a4a4a",
            button_color="#333333"
        )
        self.appearance_mode_menu.grid(row=2, column=0, padx=20, pady=(10, 20))
        self.appearance_mode_menu.set("System")
        
        self.info_label = ctk.CTkLabel(
            self.sidebar_frame, 
            text="Easily split large videos\ninto bite-sized clips\nwith precision.\n\nReady to work.", 
            justify="left", 
            text_color="gray"
        )
        self.info_label.grid(row=5, column=0, padx=20, pady=20, sticky="sw")

        # --- Main Content Area ---
        self.main_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.main_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(3, weight=1) # Log box stretches
        
        self.header_label = ctk.CTkLabel(self.main_frame, text="Current Project", font=ctk.CTkFont(size=26, weight="bold"))
        self.header_label.grid(row=0, column=0, padx=10, pady=(0, 20), sticky="w")

        # --- Card 1: File Selection ---
        self.file_card = ctk.CTkFrame(self.main_frame, corner_radius=10)
        self.file_card.grid(row=1, column=0, padx=10, pady=(0, 15), sticky="ew")
        self.file_card.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(self.file_card, text="File Paths", font=ctk.CTkFont(size=16, weight="bold")).grid(row=0, column=0, padx=20, pady=(15, 5), sticky="w", columnspan=3)

        self.input_label = ctk.CTkLabel(self.file_card, text="Input Video:")
        self.input_label.grid(row=1, column=0, padx=20, pady=(5, 10), sticky="e")
        self.input_entry = ctk.CTkEntry(self.file_card, placeholder_text="Select a video file...")
        self.input_entry.grid(row=1, column=1, padx=5, pady=(5, 10), sticky="ew")
        self.input_btn = ctk.CTkButton(self.file_card, text="Browse", width=80, command=self.browse_input)
        self.input_btn.grid(row=1, column=2, padx=20, pady=(5, 10))

        self.output_label = ctk.CTkLabel(self.file_card, text="Output Dest:")
        self.output_label.grid(row=2, column=0, padx=20, pady=(0, 15), sticky="e")
        self.output_entry = ctk.CTkEntry(self.file_card, placeholder_text="e.g. ./trimmed_clips")
        self.output_entry.insert(0, "./trimmed_clips")
        self.output_entry.grid(row=2, column=1, padx=5, pady=(0, 15), sticky="ew")
        self.output_btn = ctk.CTkButton(self.file_card, text="Browse", width=80, command=self.browse_output)
        self.output_btn.grid(row=2, column=2, padx=20, pady=(0, 15))

        # --- Card 2: Trimming Settings ---
        self.settings_card = ctk.CTkFrame(self.main_frame, corner_radius=10)
        self.settings_card.grid(row=2, column=0, padx=10, pady=(0, 15), sticky="ew")
        self.settings_card.grid_columnconfigure((1, 3), weight=1)

        ctk.CTkLabel(self.settings_card, text="Trimmer Options", font=ctk.CTkFont(size=16, weight="bold")).grid(row=0, column=0, padx=20, pady=(15, 5), sticky="w", columnspan=4)

        self.dur_label = ctk.CTkLabel(self.settings_card, text="Clip Duration (s):")
        self.dur_label.grid(row=1, column=0, padx=20, pady=10, sticky="e")
        self.dur_entry = ctk.CTkComboBox(self.settings_card, values=["5", "10", "15", "30", "60", "120"])
        self.dur_entry.set("10")
        self.dur_entry.grid(row=1, column=1, padx=5, pady=10, sticky="ew")

        self.prefix_label = ctk.CTkLabel(self.settings_card, text="File Prefix:")
        self.prefix_label.grid(row=1, column=2, padx=20, pady=10, sticky="e")
        self.prefix_entry = ctk.CTkEntry(self.settings_card)
        self.prefix_entry.insert(0, "clip")
        self.prefix_entry.grid(row=1, column=3, padx=20, pady=10, sticky="ew")

        self.start_label = ctk.CTkLabel(self.settings_card, text="Start Time (s):")
        self.start_label.grid(row=2, column=0, padx=20, pady=(0, 15), sticky="e")
        self.start_entry = ctk.CTkEntry(self.settings_card)
        self.start_entry.insert(0, "0")
        self.start_entry.grid(row=2, column=1, padx=5, pady=(0, 15), sticky="ew")

        self.end_label = ctk.CTkLabel(self.settings_card, text="End Time (s):")
        self.end_label.grid(row=2, column=2, padx=20, pady=(0, 15), sticky="e")
        self.end_entry = ctk.CTkEntry(self.settings_card, placeholder_text="Full Video")
        self.end_entry.grid(row=2, column=3, padx=20, pady=(0, 15), sticky="ew")

        # --- Card 3: Execution & Logs ---
        self.exec_card = ctk.CTkFrame(self.main_frame, corner_radius=10, fg_color="transparent")
        self.exec_card.grid(row=3, column=0, padx=10, pady=0, sticky="nsew")
        self.exec_card.grid_columnconfigure(0, weight=1)
        self.exec_card.grid_rowconfigure(2, weight=1)

        self.start_process_btn = ctk.CTkButton(
            self.exec_card, 
            text="START TRIMMING", 
            font=ctk.CTkFont(size=14, weight="bold"), 
            height=45,
            fg_color="#2E8B57", # SeaGreen
            hover_color="#1F5F3C",
            command=self.start_trimming
        )
        self.start_process_btn.grid(row=0, column=0, pady=(0, 10), sticky="ew")

        self.progress_bar = ctk.CTkProgressBar(self.exec_card, mode="indeterminate", height=6)
        self.progress_bar.grid(row=1, column=0, pady=(0, 10), sticky="ew")
        self.progress_bar.set(0) # hide somewhat

        self.log_box = ctk.CTkTextbox(self.exec_card, state="disabled", font=ctk.CTkFont(family="Consolas", size=12))
        self.log_box.grid(row=2, column=0, pady=0, sticky="nsew")

    def change_appearance_mode_event(self, new_appearance_mode: str):
        ctk.set_appearance_mode(new_appearance_mode)

    def log(self, message):
        """Append a message to the UI log box."""
        self.log_box.configure(state="normal")
        self.log_box.insert("end", message + "\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def browse_input(self):
        filename = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=(("Video Files", "*.mp4 *.avi *.mov *.mkv *.wmv"), ("All Files", "*.*"))
        )
        if filename:
            self.input_entry.delete(0, "end")
            self.input_entry.insert(0, filename)

    def browse_output(self):
        directory = filedialog.askdirectory(title="Select Output Folder")
        if directory:
            self.output_entry.delete(0, "end")
            self.output_entry.insert(0, directory)

    def start_trimming(self):
        # Validate inputs
        input_path = self.input_entry.get().strip()
        if not input_path:
            messagebox.showerror("Validation Error", "Please select an input video file.")
            return

        try:
            duration = int(self.dur_entry.get().strip())
            if duration <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Validation Error", "Invalid clip duration. Must be a positive integer.")
            return

        output_dir = self.output_entry.get().strip() or "./trimmed_clips"
        
        try:
            start_t = float(self.start_entry.get().strip() or 0)
            if start_t < 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Validation Error", "Start time must be a positive number.")
            return

        end_str = self.end_entry.get().strip()
        end_t = None
        if end_str:
            try:
                end_t = float(end_str)
                if end_t <= start_t:
                    messagebox.showerror("Validation Error", "End time must be greater than start time.")
                    return
            except ValueError:
                messagebox.showerror("Validation Error", "End time must be a valid number.")
                return

        prefix = self.prefix_entry.get().strip() or "clip"

        # Lock UI & Start Animation
        self.start_process_btn.configure(state="disabled", text="PROCESSING...", fg_color="#555555")
        self.log_box.configure(state="normal")
        self.log_box.delete("1.0", "end")
        self.log_box.configure(state="disabled")
        self.progress_bar.start()

        # Run processing in a separate thread so GUI doesn't freeze
        thread = threading.Thread(
            target=self.run_trimmer, 
            args=(input_path, duration, output_dir, start_t, end_t, prefix)
        )
        thread.start()

    def run_trimmer(self, input_path, duration, output_dir, start_t, end_t, prefix):
        success = trim_video(
            input_path=input_path,
            clip_duration=duration,
            output_dir=output_dir,
            start_time=start_t,
            end_time=end_t,
            max_clips=None,
            prefix=prefix,
            log_callback=self.log
        )
        
        # Reset UI on original thread
        def reset_btn():
            self.progress_bar.stop()
            self.progress_bar.set(0)
            self.start_process_btn.configure(state="normal", text="START TRIMMING", fg_color="#2E8B57")
            if success:
                self.log("\n✅ Video trimming completed successfully!")
                messagebox.showinfo("Success", "Video trimming completed successfully!")

        self.after(0, reset_btn)


if __name__ == "__main__":
    app = VideoTrimmerApp()
    app.mainloop()
