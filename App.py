import streamlit as st
import os
import sys
import tempfile
import shutil
import zipfile
import time
from pathlib import Path
import pandas as pd
import plotly.express as px
import importlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from scipy import signal as scipy_signal

try:
    import librosa
    import torch
    import torch.nn as nn
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    st.warning("⚠️ Librosa hoặc PyTorch không được cài đặt. Mel-spectrogram visualization sẽ không khả dụng.")

from components.header import show_header
from components.footer import show_footer
from components.styles import load_css
from components.tutorial import show_tutorial
from components.result import display_result

# Cấu hình đường dẫn
current_dir = os.path.dirname(os.path.abspath(__file__))
pure_dir = os.path.join(current_dir, 'pure')
improvement_dir = os.path.join(current_dir, 'improvement')
stacking_dir = os.path.join(current_dir, 'stack')

# Thêm các đường dẫn vào sys.path
sys.path.append(current_dir)
sys.path.append(pure_dir)
sys.path.append(improvement_dir)
sys.path.append(stacking_dir)

# --- CORRECTED: Ensure helper_code functions are imported ---
try:
    from helper_code import load_text_file, get_variable, get_cpc
except ImportError as e:
    st.error(f"Không thể import helper_code: {e}. Đảm bảo helper_code.py tồn tại trong {current_dir}.")
    st.stop()

# --- Dynamic Importer with Debug Output ---
def get_model_functions(model_module_name, model_type="pure"):
    """
    Dynamically imports load_challenge_models and run_challenge_models
    from the specified model module. For stacking, only run_challenge_models is imported.
    
    Args:
        model_module_name: Tên module (ví dụ: "team_code_densenet" hoặc "stacking")
        model_type: "pure", "improvement", hoặc "stacking"
    """
    try:
        # Tạo tên module đầy đủ với prefix folder
        if model_type == "pure":
            full_module_name = f"pure.{model_module_name}"
        elif model_type == "improvement":
            full_module_name = f"improvement.{model_module_name}"
        else:  # stacking
            full_module_name = f"stack.{model_module_name}"
        
        module = importlib.import_module(full_module_name)
        
        if model_type == "stacking":
            run_models_func = getattr(module, 'run_challenge_models')
            load_models_func = getattr(module, 'get_challenge_models')
            return load_models_func, run_models_func  # Chỉ trả về run_challenge_models, không có load_challenge_models
        else:
            load_models_func = getattr(module, 'load_challenge_models')
            run_models_func = getattr(module, 'run_challenge_models')
            return load_models_func, run_models_func
        
    except ImportError as e:
        st.error(f"❌ Không thể import module: {full_module_name}. Error: {str(e)}. Đảm bảo file {model_module_name}.py tồn tại trong folder {model_type}/.")
        return None, None
    except AttributeError as e:
        if model_type == "stacking":
            st.error(f"❌ Module {full_module_name} thiếu hàm 'run_challenge_models'. Error: {str(e)}")
        else:
            st.error(f"❌ Module {full_module_name} thiếu hàm 'load_challenge_models' hoặc 'run_challenge_models'. Error: {str(e)}")
        return None, None
    except Exception as e:
        st.error(f"❌ Lỗi không xác định khi import từ {full_module_name}: {str(e)}")
        return None, None

# Cấu hình trang
st.set_page_config(
    page_title="EEG Prediction App",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown(load_css(), unsafe_allow_html=True)

class EEGPredictor:
    def __init__(self):
        self.models = None
        self.is_loaded = False
        self.current_model_name = None
        self.load_challenge_models_dynamic = None
        self.run_challenge_models_dynamic = None

    def set_model_functions(self, model_name, load_func, run_func):
        if self.current_model_name != model_name:
            self.models = None
            self.is_loaded = False
            self.current_model_name = model_name
        self.load_challenge_models_dynamic = load_func
        self.run_challenge_models_dynamic = run_func

    def load_models(self, model_physical_folder):
        if not self.load_challenge_models_dynamic:
            st.error("❌ Hàm tải model chưa được thiết lập. Vui lòng chọn model hợp lệ.")
            return False
        try:
            if not self.is_loaded:
                with st.spinner(f"Đang tải models cho {self.current_model_name} từ {model_physical_folder}..."):
                    self.models = self.load_challenge_models_dynamic(model_physical_folder, verbose=1)
                    self.is_loaded = True
                st.success(f"✅ Models cho {self.current_model_name} từ {model_physical_folder} đã được tải thành công!")
            else:
                st.info(f"Models cho {self.current_model_name} đã được tải.")
            return True
        except Exception as e:
            st.error(f"❌ Lỗi khi tải models cho {self.current_model_name} từ {model_physical_folder}: {str(e)}")
            self.is_loaded = False
            return False

    def predict_single_patient(self, temp_data_folder, patient_id, model_physical_folder):
        if not self.run_challenge_models_dynamic:
            st.error("❌ Hàm predict model chưa được thiết lập. Vui lòng chọn model hợp lệ.")
            return None, None, None
        try:
            if not self.is_loaded:
                st.warning(f"Models cho {self.current_model_name} chưa được tải. Đang thử tải...")
                if not self.load_models(model_physical_folder):
                    return None, None, None

            patient_folder = os.path.join(temp_data_folder, patient_id)
            if not os.path.exists(patient_folder):
                st.error(f"Không tìm thấy folder patient: {patient_id}")
                return None, None, None
            files_in_folder = os.listdir(patient_folder)
            hea_files = [f for f in files_in_folder if f.endswith('.hea')]
            mat_files = [f for f in files_in_folder if f.endswith('.mat')]
            if not hea_files or not mat_files:
                st.error(f"Thiếu file .hea hoặc .mat trong folder {patient_id}")
                return None, None, None
            metadata_file = os.path.join(patient_folder, f"{patient_id}.txt")
            actual_outcome = None
            if os.path.exists(metadata_file):
                try:
                    meta_data = load_text_file(metadata_file)
                    actual_outcome = get_variable(meta_data, 'Outcome', str) if meta_data else None
                except Exception as e_meta:
                    st.warning(f"Không thể đọc outcome từ metadata file {metadata_file}: {e_meta}")
                    pass
            else:
                with open(metadata_file, 'w') as f:
                    f.write(f"Patient: {patient_id}\n")
                    f.write("Age: Unknown\n")
                    f.write("Sex: Unknown\n")
                    f.write("Outcome: Unknown\n")

            with st.spinner(f"Đang predict cho patient {patient_id} sử dụng {self.current_model_name}..."):
                outcome_binary, outcome_probability = self.run_challenge_models_dynamic(
                    self.models, temp_data_folder, patient_id, verbose=0
                )
            return outcome_binary, outcome_probability, actual_outcome
        except Exception as e:
            st.error(f"Lỗi khi predict patient {patient_id} với {self.current_model_name}: {str(e)}")
            return None, None, None

def debug_folder_structure(base_path, level=0, max_level=3):
    debug_info = []
    if level > max_level:
        return debug_info
    try:
        items = os.listdir(base_path)
        for item in items:
            item_path = os.path.join(base_path, item)
            indent = "  " * level
            if os.path.isdir(item_path):
                debug_info.append(f"{indent}📁 {item}/")
                try:
                    files = os.listdir(item_path)
                    hea_files = [f for f in files if f.endswith('.hea')]
                    mat_files = [f for f in files if f.endswith('.mat')]
                    if hea_files or mat_files:
                        debug_info.append(f"{indent}  → .hea files: {len(hea_files)}, .mat files: {len(mat_files)}")
                    if level < max_level:
                        sub_debug = debug_folder_structure(item_path, level + 1, max_level)
                        debug_info.extend(sub_debug)
                except PermissionError:
                    debug_info.append(f"{indent}  → (Permission denied)")
            else:
                file_ext = os.path.splitext(item)[1]
                if file_ext in ['.hea', '.mat', '.txt']:
                    debug_info.append(f"{indent}📄 {item}")
    except Exception as e:
        debug_info.append(f"{indent}❌ Error reading {base_path}: {str(e)}")
    return debug_info

def load_recording_data(recording_location):
    """Load EEG recording data from .mat file"""
    try:
        mat_file = recording_location + '.mat'
        hea_file = recording_location + '.hea'
        
        if not os.path.exists(mat_file):
            st.error(f"❌ .mat file not found: {mat_file}")
            return None, None, None
            
        # Load .mat file
        mat_data = sio.loadmat(mat_file)
        
        # Extract signal data
        recording_data = None
        if 'val' in mat_data:
            recording_data = mat_data['val']
        else:
            # Find key containing signal data (ignore metadata keys)
            data_keys = [k for k in mat_data.keys() if not k.startswith('__')]
            if data_keys:
                key = data_keys[0]
                recording_data = mat_data[key]
            else:
                st.error("❌ No valid data keys found in .mat file")
                return None, None, None
        
        # Load header file for channel names and sampling frequency
        channels = []
        sampling_frequency = 250  # default
        
        if os.path.exists(hea_file):
            with open(hea_file, 'r') as f:
                lines = f.readlines()
                
                if lines:
                    # First line contains basic info
                    first_line = lines[0].strip().split()
                    if len(first_line) >= 3:
                        try:
                            sampling_frequency = int(first_line[2])
                        except:
                            pass  # Use default
                    
                    # Following lines contain channel info
                    for line in lines[1:]:
                        if line.strip():
                            parts = line.strip().split()
                            if len(parts) >= 2:
                                channels.append(parts[1])
        
        # If no channels from header, use standard EEG channel names
        if not channels:
            standard_eeg_channels = [
                'Fp1', 'Fp2', 'F7', 'F8', 'F3', 'F4', 'T3', 'T4', 
                'C3', 'C4', 'T5', 'T6', 'P3', 'P4', 'O1', 'O2', 
                'Fz', 'Cz', 'Pz', 'Fpz', 'Oz', 'F9'
            ]
            num_channels = recording_data.shape[0]
            if num_channels <= len(standard_eeg_channels):
                channels = standard_eeg_channels[:num_channels]
            else:
                channels = standard_eeg_channels + [f"EEG_{i}" for i in range(len(standard_eeg_channels), num_channels)]
            
        return recording_data, channels, sampling_frequency
        
    except Exception as e:
        st.error(f"❌ Error loading recording data: {str(e)}")
        return None, None, None

def add_eeg_visualization_section(results, all_patient_folders_info, selected_model_display_name, model_type):
    """EEG visualization section with auto display"""
    
    if not results or len([r for r in results if 'Error' not in r['Prediction']]) == 0:
        st.info("Chạy prediction trước để có thể visualize EEG signals.")
        return
    
    st.markdown("---")
    st.header("📊 EEG Signal Visualization")
    
    # Get successful patients
    successful_patients = [r['Patient ID'] for r in results if 'Error' not in r['Prediction']]
    
    if not successful_patients:
        st.info("Không có patient nào để visualize (tất cả đều gặp lỗi prediction).")
        return
    
    # UI controls
    col_viz1, col_viz2 = st.columns([2, 1])
    
    with col_viz1:
        selected_patient = st.selectbox(
            "Chọn Patient để xem EEG:",
            options=successful_patients,
            help="Chọn bệnh nhân để hiển thị tín hiệu EEG thô",
            key="patient_selector"
        )
    
    with col_viz2:
        viz_col1, viz_col2 = st.columns(2)
        with viz_col1:
            channels_to_plot = st.number_input(
                "Số kênh hiển thị:",
                min_value=1,
                max_value=8,
                value=4,
                help="Số lượng kênh EEG để hiển thị",
                key="channels_input"
            )
        with viz_col2:
            minutes_to_plot = st.number_input(
                "Thời gian (phút):",
                min_value=0.5,
                max_value=10.0,
                value=2.0,
                step=0.5,
                help="Thời gian tín hiệu EEG để hiển thị",
                key="minutes_input"
            )
    
    # Display patient info
    try:
        selected_result = next(r for r in results if r['Patient ID'] == selected_patient)
        
        col_info1, col_info2, col_info3 = st.columns(3)
        with col_info1:
            st.metric("Patient ID", selected_patient)
        with col_info2:
            pred_color = "🟢" if selected_result['Prediction'] == 'Good' else "🔴"
            st.metric("Prediction", f"{pred_color} {selected_result['Prediction']}")
        with col_info3:
            actual_color = "🟢" if selected_result['Actual'] == 'Good' else ("🔴" if selected_result['Actual'] == 'Poor' else "⚫")
            st.metric("Actual", f"{actual_color} {selected_result['Actual']}")
    except Exception as e:
        st.error(f"Error displaying patient info: {str(e)}")
        return
    
    # Auto display EEG signals
    st.markdown("### 📈 EEG Signals Display")
    
    try:
        with st.spinner(f"Đang tải và xử lý tín hiệu EEG cho patient {selected_patient}..."):
            
            # Find patient source path
            patient_source_path = None
            for patient_id, patient_path in all_patient_folders_info:
                if patient_id == selected_patient:
                    patient_source_path = patient_path
                    break
            
            if not patient_source_path:
                st.error(f"❌ Không tìm thấy đường dẫn data cho patient {selected_patient}")
                return
            
            # Create visualization
            fig = visualize_eeg_signals_safe(
                patient_source_path,
                selected_patient,
                int(channels_to_plot),
                float(minutes_to_plot)
            )
            
            if fig:
                st.pyplot(fig)
                plt.close(fig)  # Clean up to prevent memory issues
                
                # Additional info
                with st.expander("ℹ️ Thông tin về EEG Visualization", expanded=False):
                    st.markdown(f"""
                    **Thông tin hiển thị:**
                    - **Patient ID**: {selected_patient}
                    - **Prediction**: {selected_result['Prediction']}
                    - **Actual Outcome**: {selected_result['Actual']}
                    - **Số kênh hiển thị**: {int(channels_to_plot)} kênh EEG chuẩn
                    - **Thời gian**: {float(minutes_to_plot)} phút (từ giữa recording)
                    - **Model sử dụng**: {selected_model_display_name} ({model_type})
                    
                    **Tên kênh EEG chuẩn**: Fp1, Fp2, F7, F8, F3, F4, T3, T4, C3, C4, T5, T6, P3, P4, O1, O2, Fz, Cz, Pz, Fpz, Oz, F9
                    """)
            else:
                st.error("❌ Không thể tạo visualization cho patient này.")
                
    except Exception as e:
        st.error(f"❌ Lỗi khi tạo EEG visualization: {str(e)}")

def select_n_labels(labels, n):
    """Select n evenly spaced labels from a list"""
    if n >= len(labels):
        return labels
    indices = np.linspace(0, len(labels)-1, n, dtype=int)
    return [labels[i] for i in indices]

def visualize_eeg_signals_safe(patient_folder_path, patient_id, channels_to_plot=4, minutes_to_plot=2):
    """Create EEG visualization with proper channel names and clean display"""
    try:
        # Find .mat and .hea files in folder
        files = os.listdir(patient_folder_path)
        mat_files = [f for f in files if f.endswith('.mat')]
        
        if not mat_files:
            st.error(f"No .mat files found in {patient_folder_path}")
            return None
            
        # Use first .mat file
        mat_file = mat_files[0]
        base_name = mat_file.replace('.mat', '')
        recording_location = os.path.join(patient_folder_path, base_name)
        
        # Load recording data
        recording_data, channels, sampling_frequency = load_recording_data(recording_location)
        
        if recording_data is None:
            return None
        
        # Calculate time window parameters
        samples_per_minute = int(60 * sampling_frequency)
        i_to_plot = int(minutes_to_plot * samples_per_minute)
        sig_len = recording_data.shape[1]
        signal_mid = sig_len // 2
        signal_start = max(0, int(signal_mid - i_to_plot//2))
        signal_end = min(sig_len, int(signal_mid + i_to_plot//2))
        
        # Select channels randomly (similar to reference code)
        num_channels = recording_data.shape[0]
        num_channels_to_plot = min(channels_to_plot, num_channels)
        
        # Set random seed for reproducible results
        np.random.seed(42)
        rand_channel_ids = np.random.choice(num_channels, num_channels_to_plot, replace=False)
        rand_channels = [channels[i] if i < len(channels) else f"EEG_{i}" for i in rand_channel_ids]
        
        # Extract signal segments from middle of recording
        rand_signal_selection = []
        for ch_idx in rand_channel_ids:
            signal_segment = recording_data[ch_idx, signal_start:signal_end]
            rand_signal_selection.append(signal_segment)
        
        # Create time ticks in minutes
        total_samples = signal_end - signal_start
        time_minutes = np.arange(total_samples) / (60 * sampling_frequency)
        
        # Set up the figure
        fig, axs = plt.subplots(num_channels_to_plot, 1, figsize=(15, 12))
        if num_channels_to_plot == 1:
            axs = [axs]
        
        # Plot each channel
        num_ticks = 8
        for i in range(num_channels_to_plot):
            # Plot the signal
            axs[i].plot(rand_signal_selection[i], 'b-', linewidth=0.8)
            
            # Set title and labels
            axs[i].set_title(rand_channels[i], fontsize=18, fontweight='bold')
            axs[i].set_xlabel("Time (min)", fontsize=16)
            axs[i].set_ylabel("Voltage (μV)", fontsize=16)
            axs[i].tick_params(axis='both', which='major', labelsize=14)
            
            # Set up time axis with proper labels
            time_labels = time_minutes
            selected_labels = select_n_labels(time_labels, num_ticks)
            selected_ticks = np.linspace(0, len(rand_signal_selection[i])-1, num_ticks, dtype=int)
            
            axs[i].set_xticks(selected_ticks)
            axs[i].set_xticklabels([f"{label:.1f}" for label in selected_labels])
            
            # Add grid for better readability
            axs[i].grid(True, alpha=0.3, linestyle='--')
        
        # Overall title
        plt.suptitle(f"Patient {patient_id} - EEG Signals\n"
                    f"Sampling Rate: {sampling_frequency} Hz | Duration: {minutes_to_plot} min", 
                    fontsize=20, fontweight='bold')
        
        # Adjust layout
        plt.subplots_adjust(hspace=0.6, top=0.85)
        
        return fig
        
    except Exception as e:
        st.error(f"Error in EEG visualization: {str(e)}")
        return None

def extract_uploaded_files(uploaded_files, temp_dir):
    extracted_folders_map = {}
    for uploaded_file in uploaded_files:
        file_name = uploaded_file.name
        file_path = os.path.join(temp_dir, file_name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        if file_name.endswith('.zip'):
            try:
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                    extracted_folders_map[temp_dir] = True
                    st.success(f"✅ Đã giải nén: {file_name} vào {temp_dir}")
            except Exception as e:
                st.error(f"❌ Lỗi khi giải nén {file_name}: {str(e)}")
    return list(extracted_folders_map.keys())

def find_patient_folders(base_path, debug_mode=False):
    patient_folders_dict = {}
    if debug_mode:
        st.info(f"🔍 Scanning directory: {base_path}")
        st.markdown("**📁 Folder Structure (during find_patient_folders):**")
        debug_info = debug_folder_structure(base_path, max_level=2)
        if debug_info:
            for line in debug_info[:30]:
                st.text(line)
            if len(debug_info) > 30:
                st.text(f"... and {len(debug_info) - 30} more items")
    for root, dirs, files in os.walk(base_path):
        hea_files = [f for f in files if f.endswith('.hea')]
        mat_files = [f for f in files if f.endswith('.mat')]
        if debug_mode and (hea_files or mat_files or dirs):
            relative_path = os.path.relpath(root, base_path)
        if hea_files and mat_files:
            folder_name = os.path.basename(root)
            if folder_name not in patient_folders_dict:
                if root != base_path or (root == base_path and not any(os.path.isdir(os.path.join(root, d)) for d in dirs if d != "prediction_input_data")):
                    patient_folders_dict[folder_name] = root
                    if debug_mode:
                        st.success(f"✅ Tentatively found patient: {folder_name} at {root}")
    patient_folders = list(patient_folders_dict.items())
    if debug_mode and not patient_folders:
        st.warning(f"No patient folders found directly in {base_path} or its subdirectories.")
    return patient_folders

def main():
    show_header()
    # HEADER & SIDEBAR CONFIGURATION
    st.sidebar.header("⚙️ Cấu hình")
    debug_mode = st.sidebar.checkbox("🐛 Debug Mode", value=False, help="Show detailed folder structure and debugging info")

    # --- Model Type Selection ---
    st.sidebar.subheader("🎯 Chọn Phiên Bản Model")
    model_type = st.sidebar.radio(
        "Loại Model:",
        options=["pure", "improvement", "stacking"],
        format_func=lambda x: "🔵 Pure (Gốc)" if x == "pure" else "🟢 Improvement (Cải tiến)" if x == "improvement" else "🟡 Stacking (Stacked)",
        help="Chọn giữa phiên bản gốc (pure) và phiên bản cải tiến (improvement) và stacking"
    )

    # --- Model Configuration ---
    model_config = {
        "DenseNet-121": {
            "module": "team_code_densenet121",
            "path": {
                "pure": "diikhanh/pure-densenet121",
                "improvement": "diikhanh/improvement-densenet121"
            }
        },
        "ResNet-50": {
            "module": "team_code_resnet50",
            "path": {
                "pure": "diikhanh/pure-resnet50",
                "improvement": "diikhanh/improvement-resnet50"
            }
        },
        "ConvNeXt": {
            "module": "team_code_convnext",
            "path": {
                "pure": "diikhanh/pure-convnext",
                "improvement": "diikhanh/improvement-convnext"
            }
        },
        "EfficientNet-V2": {
            "module": "team_code_efficientnetv2",
            "path": {
                "pure": "diikhanh/pure-efficentnetv2",
                "improvement": "diikhanh/improvement-efficentnetv2"
            }  
        },
        "Stacking": {
            "module": "stacking",
            "path": {
                "stacking": "diikhanh/stacking"
            }
        }
    }

    # --- Model Selection ---
    # if model_type != "stacking":  # Không hiển thị nút tải model nếu là stacking
    st.sidebar.subheader("🤖 Chọn Model")
    selected_model_display_name = st.sidebar.selectbox(
        "Model Architecture:",
        options=list(model_config.keys()),
        help="Chọn kiến trúc model để sử dụng cho prediction."
    )

    # Hiển thị thông tin model đã chọn
    if model_type == "stacking":
        selected_model_module_name = model_config["Stacking"]["module"]
        selected_model_physical_path = model_config["Stacking"]["path"]['stacking']
    else:
        selected_model_module_name = model_config[selected_model_display_name]["module"]
        selected_model_physical_path = model_config[selected_model_display_name]["path"][model_type]

    # Debug information
    if debug_mode:
        st.sidebar.subheader("🐛 Debug Information")
        st.sidebar.code(f"""
    Current Directory: {current_dir}
    Pure Directory: {pure_dir}
    Improvement Directory: {improvement_dir}
    Stacking Directory: {stacking_dir}
    Selected Module: {model_type}.{selected_model_module_name}
    Model Path: {selected_model_physical_path}
    Sys Path: {sys.path[-3:]}  # Last 3 entries
        """)

    # --- Initialize Predictor ---
    if 'predictor' not in st.session_state:
        st.session_state.predictor = EEGPredictor()

    # --- Load Model Functions ---
    load_fn, run_fn = get_model_functions(selected_model_module_name, model_type)
    if load_fn and run_fn:
        st.session_state.predictor.set_model_functions(selected_model_display_name, load_fn, run_fn)
        st.sidebar.success(f"✅ Đã tải functions cho {selected_model_display_name} ({model_type})")
    elif model_type == "stacking":
        st.session_state.predictor.set_model_functions("Stacking", load_fn, run_fn)
        st.sidebar.success(f"✅ Đã tải functions cho {selected_model_display_name} ({model_type})")
    else:
        st.sidebar.error(f"❌ Không thể tải các hàm cho model {selected_model_display_name} ({model_type}). Kiểm tra module '{model_type}.{selected_model_module_name}'.")

    # --- Load Models Button ---
    
    if st.sidebar.button("🔄 Tải Models", key="load_models_button"):
        if st.session_state.predictor.load_challenge_models_dynamic:
            with st.spinner(f"Đang tải {selected_model_display_name} ({model_type})..."):
                try:
                    st.session_state.predictor.load_models(selected_model_physical_path)
                    st.sidebar.success(f"✅ Đã tải thành công {selected_model_display_name} ({model_type})")
                except Exception as e:
                    st.sidebar.error(f"❌ Lỗi khi tải model: {str(e)}")
        else:
            st.sidebar.error("❌ Hàm tải model chưa được thiết lập do lỗi import. Vui lòng chọn model hợp lệ và kiểm tra thông báo lỗi.")

    # --- Optional: Model Status Display ---
    if 'predictor' in st.session_state and hasattr(st.session_state.predictor, 'current_model_info'):
        current_info = st.session_state.predictor.current_model_info
        if current_info:
            st.sidebar.markdown("---")
            st.sidebar.markdown("**🎯 Model hiện tại:**")
            st.sidebar.markdown(f"- {current_info.get('name', 'Unknown')}")
            st.sidebar.markdown(f"- Type: {current_info.get('type', 'Unknown')}")
    col1, col2 = st.columns([2, 1])
    with col1:
        show_tutorial()
    with col2:
        uploaded_files = st.file_uploader(
            "Chọn file ZIP chứa dữ liệu EEG",
            accept_multiple_files=True,
            type=['zip'],
            help="Upload file ZIP chứa folder bệnh nhân với file .hea và .mat"
        )
        if uploaded_files:
                st.success(f"✅ Đã upload {len(uploaded_files)} file(s)")
                file_info = [{"Tên File": f.name, "Kích thước": f"{f.size / (1024*1024):.2f} MB", "Loại": "ZIP Archive"} for f in uploaded_files]
                st.dataframe(pd.DataFrame(file_info), use_container_width=True)

    st.header("🎯 Prediction")
    if st.button("🚀 Bắt đầu Predict", type="primary", use_container_width=True, key="predict_button"):
        if not uploaded_files:
            st.warning("⚠️ Vui lòng upload files EEG trước!")
            return

        if not st.session_state.predictor.load_challenge_models_dynamic or \
            not st.session_state.predictor.run_challenge_models_dynamic:
            st.error("❌ Model functions không được tải đúng cách. Vui lòng kiểm tra lựa chọn model và thông báo lỗi ở sidebar.")
            return

        if not st.session_state.predictor.is_loaded:
            st.warning(f"⚠️ Models cho {st.session_state.predictor.current_model_name} chưa được tải! Đang thử tải...")
            if not st.session_state.predictor.load_models(selected_model_physical_path):
                st.error("Không thể tải models. Prediction bị hủy.")
                return

        with tempfile.TemporaryDirectory() as temp_dir:
            if debug_mode: st.info(f"🔧 Debug: Using temp directory: {temp_dir}")
            # st.info("📦 Đang xử lý files upload...") # Can be noisy
            base_extraction_path = temp_dir
            for uploaded_file in uploaded_files:
                file_path = os.path.join(base_extraction_path, uploaded_file.name)
                with open(file_path, "wb") as f: f.write(uploaded_file.getbuffer())
                if uploaded_file.name.endswith('.zip'):
                    try:
                        with zipfile.ZipFile(file_path, 'r') as zip_ref:
                            zip_ref.extractall(base_extraction_path)
                            # st.success(f"✅ Đã giải nén: {uploaded_file.name} vào {base_extraction_path}")
                    except Exception as e:
                        st.error(f"❌ Lỗi khi giải nén {uploaded_file.name}: {str(e)}")
                        continue
                else: st.warning(f"Skipping non-ZIP file: {uploaded_file.name}")

            if debug_mode:
                st.markdown(f"### 🐛 Debug: Structure of temp_dir after extraction: {base_extraction_path}")
                debug_tree = debug_folder_structure(base_extraction_path, max_level=2)
                for line in debug_tree: st.text(line)

            st.info("🔍 Đang tìm patient folders...")
            all_patient_folders_info = find_patient_folders(base_extraction_path, debug_mode=debug_mode)

            if not all_patient_folders_info:
                st.error("❌ Không tìm thấy patient data hợp lệ trong files upload.")
                if debug_mode:
                    st.markdown("### 🐛 Debug Help for No Patients Found:")
                    st.markdown(f"""
                    **Kiểm tra các vấn đề sau:**
                    1. File ZIP có thực sự chứa các **folder con** không? (ví dụ: `patient_ID_1/`, `patient_ID_2/`)
                    2. Mỗi folder con (ví dụ: `patient_ID_1/`) có chứa cả file `.hea` và `.mat` không?
                    3. Tên file có đúng định dạng không?
                    4. Cấu trúc thư mục có khớp với hướng dẫn không?
                    **Cấu trúc thư mục được quét trong `{base_extraction_path}`:**
                    Ví dụ: `{base_extraction_path}/0391/0391.hea` và `{base_extraction_path}/0391/0391.mat`
                    """)
                return

            st.success(f"✅ Tìm thấy {len(all_patient_folders_info)} patient(s).")
            if debug_mode:
                st.markdown("### 📋 Found Patients for Prediction:")
                for patient_id, patient_original_path in all_patient_folders_info:
                    st.text(f"  👤 {patient_id} (source: {patient_original_path})")

            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            prediction_input_dir = os.path.join(temp_dir, "prediction_input_data")
            os.makedirs(prediction_input_dir, exist_ok=True)

            for i, (patient_id, patient_original_path) in enumerate(all_patient_folders_info):
                progress_bar.progress((i + 1) / len(all_patient_folders_info))
                status_text.text(f"🔄 Đang predict cho Patient {patient_id} ({i+1}/{len(all_patient_folders_info)})")
                temp_patient_run_folder = os.path.join(prediction_input_dir, patient_id)
                os.makedirs(temp_patient_run_folder, exist_ok=True)
                try:
                    for item_name in os.listdir(patient_original_path):
                        src_item = os.path.join(patient_original_path, item_name)
                        dst_item = os.path.join(temp_patient_run_folder, item_name)
                        if os.path.isfile(src_item): shutil.copy2(src_item, dst_item)
                    if debug_mode:
                        copied_files = os.listdir(temp_patient_run_folder)
                        # st.text(f"  Copied {len(copied_files)} files to {temp_patient_run_folder} for patient {patient_id}")
                    # --- BẮT ĐẦU CODE THÊM ĐỂ HIỂN THỊ NỘI DUNG FILE .TXT ---
                    patient_txt_filename = f"{patient_id}.txt"
                    patient_txt_file_path = os.path.join(temp_patient_run_folder, patient_txt_filename)

                    if os.path.exists(patient_txt_file_path):
                        with st.expander(f"📄 Metadata cho Patient {patient_id} (File: {patient_txt_filename})", expanded=True):
                            try:
                                with open(patient_txt_file_path, 'r', encoding='utf-8') as f_txt:
                                    lines = f_txt.readlines()
                                
                                metadata = []
                                for line in lines:
                                    line = line.strip()
                                    if line and ": " in line:
                                        parts = line.split(": ", 1)
                                        if len(parts) == 2:
                                            key = parts[0].strip()
                                            value = parts[1].strip()
                                            
                                            # Bỏ qua những dòng có key là "Outcome" hoặc "CPC"
                                            if key.lower() not in ['outcome', 'cpc']:
                                                metadata.append([key, value])
                                
                                if metadata:
                                    df = pd.DataFrame(metadata, columns=['Key', 'Value'])
                                    st.dataframe(df, use_container_width=True)
                                else:
                                    st.text("Không có dữ liệu metadata có định dạng hợp lệ.")
                            except Exception as e_read_txt:
                                st.warning(f"⚠️ Không thể đọc file metadata {patient_txt_filename}: {str(e_read_txt)}")
                    else:
                        st.text(f"ℹ️ Không tìm thấy file metadata ({patient_txt_filename}) cho patient {patient_id} tại {temp_patient_run_folder}.")
                    # --- KẾT THÚC CODE THÊM ---
                except Exception as e:
                    st.error(f"Error copying files for {patient_id}: {str(e)}")
                    results.append({'Patient ID': patient_id, 'Prediction': 'Error - File Prep', 'Actual': "N/A"})
                    continue
                
                outcome_binary, outcome_prob, actual_outcome = st.session_state.predictor.predict_single_patient(
                    prediction_input_dir, patient_id, selected_model_physical_path
                )

                if outcome_binary is not None:
                    results.append({
                        'Patient ID': patient_id,
                        'Prediction': 'Good' if outcome_binary == 0 else 'Poor',
                        'Actual': actual_outcome if actual_outcome else "Unknown"
                    })
                else:
                    results.append({
                        'Patient ID': patient_id,
                        'Prediction': 'Error - Prediction Failed',
                        'Actual': actual_outcome if actual_outcome else "N/A" # Keep actual if read
                    })

            progress_bar.empty()
            status_text.empty()
            # --- EEG VISUALIZATION SECTION (SAFE VERSION) ---
            try:
                add_eeg_visualization_section(results, all_patient_folders_info, selected_model_display_name, model_type)
            except Exception as e:
                st.error(f"Error in EEG visualization section: {str(e)}")
                st.exception(e)

            if results:
                display_result(results, selected_model_display_name)
            else:
                st.error("❌ Không có kết quả prediction nào!")

    show_footer()

if __name__ == "__main__":
    main()