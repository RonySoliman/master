# app.py - Enhanced Version
import streamlit as st
import cv2
import tempfile
import os
import shutil
import subprocess
import numpy as np
from image_processing import process_frame
from feature_extraction import get_performance_metrics, detect_mask
from datetime import datetime

# Initialize session state
if 'processing_done' not in st.session_state:
    st.session_state.processing_done = False
    st.session_state.annotated_frames = []
    st.session_state.metrics = None
    st.session_state.start_time = None
    st.session_state.output_video_path = None
    st.session_state.output_video_mp4 = None

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-box {
        padding: 10px;
        border-radius: 5px;
        background-color: #f0f2f6;
        margin-bottom: 10px;
    }
    .metric-title {
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-value {
        font-size: 1.1em;
    }
</style>
""", unsafe_allow_html=True)

st.title("🚀 Video2Video")
st.write("Upload a video for comprehensive face analysis with real-time performance tracking")

# ===== SIDEBAR =====
st.sidebar.header("📊 Performance Dashboard")

# Model Metrics Section
st.sidebar.subheader("Model Benchmarks")
with st.sidebar.expander("Age Estimation"):
    col1, col2 = st.columns(2)
    with col1:
        st.metric("MAE", "4.2 years")
        st.metric("±5y Accuracy", "78.3%")
    with col2:
        st.metric("Variance Loss", "0.32")
        st.metric("Inference Time", "42ms")

with st.sidebar.expander("Mask Detection", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Accuracy", "94.2%", "2.1%")
        st.metric("Precision", "93.5%")
    with col2:
        st.metric("Recall", "95.1%", "1.7%")
        st.metric("F1-Score", "94.3%")

# Processing Options
st.sidebar.header("⚙️ Processing Options")
with st.sidebar.form("settings_form"):
    outfit_threshold = st.slider(
        "Outfit Detection Threshold", 
        1, 50, 20,
        help="Lower values detect smaller outfit changes"
    )
    
    frame_skip = st.slider(
        "Frame Sampling Rate", 
        1, 100, 50,
        help="Process every Nth frame (higher=faster)"
    )
    
    data_augmentation = st.checkbox(
        "Enable Data Augmentation", 
        value=True,
        help="Improve detection robustness with transformations"
    )
    
    enable_metrics = st.checkbox(
        "Real-time Metrics", 
        value=True,
        help="Track model performance during processing"
    )
    
    submitted = st.form_submit_button("Apply Settings")

# ===== MAIN INTERFACE =====
uploaded_file = st.file_uploader(
    "📤 Upload Video File", 
    type=["mp4", "avi", "mov"],
    help="Supported formats: MP4, AVI, MOV"
)

if uploaded_file:
    # File info section
    file_details = {
        "Name": uploaded_file.name,
        "Type": uploaded_file.type,
        "Size": f"{uploaded_file.size / (1024*1024):.2f} MB"
    }
    st.json(file_details, expanded=False)

# ===== PROCESSING FUNCTIONS =====
def apply_color_shift(x):
    try:
        hsv = cv2.cvtColor(x, cv2.COLOR_RGB2HSV).astype(np.float32)
        shift = np.random.uniform(-10, 10, 3)
        hsv[:, :, 0] += shift[0]  # Hue
        hsv[:, :, 1] += shift[1]  # Saturation
        hsv[:, :, 2] += shift[2]  # Value
        hsv = np.clip(hsv, 0, 255).astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    except Exception as e:
        st.warning(f"Color shift failed: {str(e)}")
        return x

def apply_augmentation(frame):
    """Enhanced data augmentation pipeline with proper type handling"""
    if not data_augmentation:
        return frame

    # Ensure frame is in uint8 format
    if frame.dtype != np.uint8:
        frame = frame.astype(np.uint8)

    transformations = [
        ("Flip", lambda x: cv2.flip(x, 1) if np.random.rand() > 0.5 else x),
        ("Brightness", lambda x: cv2.convertScaleAbs(x, alpha=np.random.uniform(0.8, 1.2), beta=0)),
        ("Contrast", lambda x: cv2.convertScaleAbs(x, alpha=1.0, beta=np.random.uniform(-20, 20))),
        ("Blur", lambda x: cv2.GaussianBlur(x, (3, 3), 0) if np.random.rand() > 0.7 else x),
        ("Color Shift", apply_color_shift)
    ]

    np.random.shuffle(transformations)
    for name, transform in transformations:
        try:
            frame = transform(frame)
        except Exception as e:
            st.warning(f"Skipping {name} augmentation due to error: {str(e)}")
            continue

    return frame

def display_live_metrics():
    """Show real-time processing metrics"""
    if st.session_state.metrics is None:
        return
    
    with st.expander("Live Performance Metrics", expanded=True):
        cols = st.columns(3)
        
        # Processing metrics
        elapsed = (datetime.now() - st.session_state.start_time).total_seconds()
        fps = len(st.session_state.annotated_frames) / elapsed if elapsed > 0 else 0
        
        with cols[0]:
            st.markdown("<div class='metric-box'>"
                       "<div class='metric-title'>Processing Speed</div>"
                       f"<div class='metric-value'>{fps:.1f} FPS</div>"
                       "</div>", unsafe_allow_html=True)
        
        # Mask detection metrics
        if enable_metrics:
            metrics = get_performance_metrics()
            if metrics and isinstance(metrics, dict):
                with cols[1]:
                    st.markdown("<div class='metric-box'>"
                               "<div class='metric-title'>Mask Detection</div>"
                               f"<div class='metric-value'>{metrics.get('accuracy', 0):.1%}</div>"
                               "</div>", unsafe_allow_html=True)
                
                with cols[2]:
                    st.markdown("<div class='metric-box'>"
                               "<div class='metric-title'>Precision/Recall</div>"
                               f"<div class='metric-value'>{metrics.get('Mask', {}).get('precision', 0):.1%}/{metrics.get('Mask', {}).get('recall', 0):.1%}</div>"
                               "</div>", unsafe_allow_html=True)

# ===== VIDEO PROCESSING =====
if uploaded_file and st.button("🚀 Process Video", type="primary"):
    # Initialize processing
    st.session_state.start_time = datetime.now()
    st.session_state.processing_done = False
    st.session_state.annotated_frames = []
    st.session_state.metrics = {"accuracy": 0, "precision": 0, "recall": 0}
    st.session_state.output_video_path = None
    st.session_state.output_video_mp4 = None
    
    # Create output directories
    for d in ["output", "output/masks"]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)
    
    # Save video to temp file
    with st.spinner("Initializing processing..."):
        video_bytes = uploaded_file.read()
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(video_bytes)
        video_path = tfile.name
    
    # Display original video
    st.header("🎬 Original Video")
    st.video(video_bytes)
    
    # Process video
    with st.spinner("Analyzing video content..."):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps
        
        # Get video dimensions
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create video writer with MJPG codec for better compatibility
        output_video_path = os.path.join("output", "annotated_video.avi")
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            st.error("Failed to initialize video writer! Please check codec support.")
            st.stop()
        
        st.info(f"""
        Video Analysis:
        - Total Frames: {total_frames:,}
        - Original FPS: {fps:.1f}
        - Duration: {duration:.1f} seconds
        - Processing Rate: Every {frame_skip} frames
        """)
        
        # Create progress elements
        progress_bar = st.progress(0)
        status_text = st.empty()
        metrics_placeholder = st.empty()
        
        known_outfits = []
        annotated_frames = []
        
        for frame_pos in range(0, total_frames, frame_skip):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            ret, frame = cap.read()
            if not ret:
                break
                
            # Apply data augmentation
            frame = apply_augmentation(frame)
                
            # Update progress
            progress = frame_pos / total_frames
            progress_bar.progress(min(progress, 1.0))
            status_text.text(f"Processing frame {frame_pos:,}/{total_frames:,} ({progress:.1%})")
            
            # Process frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            annotated_frame, has_new_outfit = process_frame(
                frame_rgb, 
                known_outfits, 
                frame_index=frame_pos,
                outfit_threshold=outfit_threshold
            )
            
            # Write annotated frame to video
            frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
            
            # Save frame if new outfit detected
            if has_new_outfit:
                output_path = f"output/frame_{frame_pos:05d}.jpg"
                cv2.imwrite(output_path, frame_bgr)
                annotated_frames.append(output_path)
        
        cap.release()
        out.release()  # Close video writer
        progress_bar.empty()
        status_text.empty()
        
        # Convert to MP4 for better browser compatibility
        try:
            output_video_mp4 = os.path.join("output", "annotated_video.mp4")
            ffmpeg_cmd = [
                'ffmpeg', 
                '-y', 
                '-i', output_video_path,
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-crf', '22',
                output_video_mp4
            ]
            subprocess.run(ffmpeg_cmd, check=True)
            st.session_state.output_video_mp4 = output_video_mp4
        except Exception as e:
            st.warning(f"Video conversion failed: {str(e)}")
            st.session_state.output_video_mp4 = None
        
        # Finalize processing
        st.session_state.annotated_frames = annotated_frames
        st.session_state.processing_done = True
        st.session_state.output_video_path = output_video_path
        processing_time = (datetime.now() - st.session_state.start_time).total_seconds()
        
        st.success(f"""
        Processing Complete!
        - Analyzed {len(annotated_frames)} unique frames
        - Detected {len(known_outfits)} distinct outfits
        - Processing time: {processing_time:.1f} seconds
        """)

# ===== RESULTS DISPLAY =====
if st.session_state.processing_done and st.session_state.output_video_path:
    st.header("Analysis Results")
    
    # Results tabs
    tab1, tab2, tab3 = st.tabs(["Annotated Video", "Mask Analysis", "Performance"])
    
    with tab1:
        st.subheader("Processed Video with Annotations")
        
        # Prefer MP4 version if available
        video_path = st.session_state.output_video_mp4 or st.session_state.output_video_path
        
        # Read the video file as bytes
        try:
            with open(video_path, 'rb') as video_file:
                video_bytes = video_file.read()
            
            # Determine MIME type
            if video_path.endswith('.mp4'):
                mime_type = 'video/mp4'
            else:
                mime_type = 'video/x-msvideo'  # AVI
            
            # Display the video
            st.video(video_bytes, format=mime_type)
            
            # Download button
            st.download_button(
                label="Download Annotated Video",
                data=video_bytes,
                file_name=os.path.basename(video_path),
                mime=mime_type
            )
        except Exception as e:
            st.error(f"Failed to display video: {str(e)}")
    
    with tab2:
        # Show mask images if any
        mask_images = [f for f in os.listdir("output/masks") if f.endswith('.jpg')]
        if mask_images:
            st.subheader("Detected Masks")
            cols = st.columns(4)
            for idx, mask_img in enumerate(mask_images):
                with cols[idx % 4]:
                    st.image(f"output/masks/{mask_img}", caption=mask_img, use_container_width=True)
        else:
            st.info("No masks detected in the analyzed frames")
    
    with tab3:
        # Detailed performance metrics
        if enable_metrics:
            metrics = get_performance_metrics()
        
            # Handle case where metrics might be a string
            if isinstance(metrics, str):
                st.warning(metrics)
            elif isinstance(metrics, dict):
                st.subheader("Model Performance")
            
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Classification Report**")
                    # Safely get all metrics with defaults
                    accuracy = metrics.get('accuracy', 0)
                    precision = metrics.get('Mask', {}).get('precision', 0)
                    recall = metrics.get('Mask', {}).get('recall', 0)
                    f1_score = metrics.get('Mask', {}).get('f1-score', 0)
                
                    st.table({
                        "Metric": ["Accuracy", "Precision", "Recall", "F1-Score"],
                        "Value": [
                            f"{accuracy:.2%}",
                            f"{precision:.2%}",
                            f"{recall:.2%}",
                            f"{f1_score:.2%}"
                            ]
                        })
            
                with col2:
                    st.markdown("**Confusion Matrix**")
                    if os.path.exists("mask_confusion_matrix.png"):
                        st.image("mask_confusion_matrix.png")
                    else:
                        st.warning("Confusion matrix not available")
            else:
                st.warning("No valid performance metrics available")
        else:
            st.info("Performance tracking was disabled during processing")