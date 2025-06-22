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
            return load_models_func, run_models_func
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
# Tăng giới hạn upload file lên 1GB
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

    def load_models(self, model_physical_folder, type='normal'):
        if not self.load_challenge_models_dynamic:
            st.error("❌ Hàm tải model chưa được thiết lập. Vui lòng chọn model hợp lệ.")
            return False
        try:
            if not self.is_loaded:
                with st.spinner(f"Đang tải models cho {self.current_model_name} từ {model_physical_folder}..."):
                    print("tai ne: ", type)
                    if (type=='stacking'):
                        print("Not load model stacking")
                        # self.models = self.load_challenge_models_dynamic(model_physical_folder, 'densenet121')
                    else: 
                        self.models = self.load_challenge_models_dynamic(model_physical_folder, verbose=1)
                    self.is_loaded = True
                st.success(f"✅ Models cho {self.current_model_name} từ {model_physical_folder} đã được tải thành công!")
            else:
                st.info(f"Models cho {self.current_model_name} đã được tải.")
            return True
        except Exception as e:
            st.error(f"❌ Lỗi khi tải models cho {self.current_model_name} từ {model_physical_folder}: {str(e)}")
            self.is_loaded = False
            return True

    def predict_single_patient(self, temp_data_folder, patient_id, model_physical_folder):
        if not self.run_challenge_models_dynamic:
            st.error("❌ Hàm predict model chưa được thiết lập. Vui lòng chọn model hợp lệ.")
            return None, None, None, None
        try:
            if not self.is_loaded:
                st.warning(f"Models cho {self.current_model_name} chưa được tải. Đang thử tải...")
                if not self.load_models(model_physical_folder):
                    return None, None, None, None

            patient_folder = os.path.join(temp_data_folder, patient_id)
            if not os.path.exists(patient_folder):
                st.error(f"Không tìm thấy folder patient: {patient_id}")
                return None, None, None, None
            files_in_folder = os.listdir(patient_folder)
            hea_files = [f for f in files_in_folder if f.endswith('.hea')]
            mat_files = [f for f in files_in_folder if f.endswith('.mat')]
            if not hea_files or not mat_files:
                st.error(f"Thiếu file .hea hoặc .mat trong folder {patient_id}")
                return None, None, None, None
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
                if self.models == None: 
                    outcome_binary, outcome_probability, segment_outcomes = self.run_challenge_models_dynamic(
                        model_physical_folder, temp_data_folder, patient_id, verbose = 0
                    )
                else: 
                    outcome_binary, outcome_probability, segment_outcomes = self.run_challenge_models_dynamic(
                        self.models, temp_data_folder, patient_id, verbose=0
                    )
            
            return outcome_binary, outcome_probability, actual_outcome, segment_outcomes
        except Exception as e:
            st.error(f"Lỗi khi predict {patient_id} với {self.current_model_name}: {str(e)}")
            return None, None, None, None

def load_recording_data(recording_location):
    """Load EEG recording data from .mat file with improved .hea parsing"""
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
                    # First line contains: record_name num_channels sampling_freq duration
                    first_line = lines[0].strip().split()
                    if len(first_line) >= 3:
                        try:
                            sampling_frequency = int(first_line[2])
                        except:
                            pass  # Use default
                    
                    # Following lines contain channel info
                    # Format: filename format gain(baseline)/units resolution checksum blocksize channel_name
                    for line in lines[1:]:
                        line = line.strip()
                        if line and not line.startswith('#'):  # Skip comments
                            parts = line.split()
                            if len(parts) >= 9:  # Ensure we have enough parts
                                # The channel name is the last part (index -1)
                                channel_name = parts[-1]
                                channels.append(channel_name)
        
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

def cleanup_temp_patients():
    """Clean up temporary patient directories."""
    temp_patients_dir = os.path.join(current_dir, "temp_patients")
    if os.path.exists(temp_patients_dir):
        try:
            shutil.rmtree(temp_patients_dir)
            st.success("✅ Đã xóa dữ liệu tạm thời.")
        except Exception as e:
            st.error(f"❌ Lỗi khi xóa dữ liệu tạm thời: {str(e)}")

def create_patient_info_display(patient_id, selected_result):
    """Create the patient information display with metrics."""
    col_info1, col_info2, col_info3 = st.columns(3)
    with col_info1:
        st.metric("Patient ID", patient_id)
    with col_info2:
        pred_color = "🟢" if selected_result['Prediction'] == 'Good' else "🔴"
        st.metric("Prediction", f"{pred_color} {selected_result['Prediction']}")
    with col_info3:
        actual_color = "🟢" if selected_result['Actual'] == 'Good' else ("🔴" if selected_result['Actual'] == 'Poor' else "⚫")
        st.metric("Actual", f"{actual_color} {selected_result['Actual']}")

def handle_recording_selection(patient_id, mat_files):
    """Handle the recording file selection dropdown."""
    dropdown_key = f"recording_select_{patient_id}"
    selected_recording = st.selectbox(
        "Chọn recording file:",
        options=mat_files,
        key=dropdown_key,
        help="Chọn file recording để hiển thị tín hiệu EEG"
    )
    return selected_recording

def display_eeg_visualization(patient_source_path, patient_id, selected_recording, channels_to_plot, selected_result, selected_model_display_name, model_type):
    """Display the EEG visualization for the selected recording."""
    with st.container():
        st.markdown(f"**Recording file:** `{selected_recording}`")
        with st.spinner(f"Đang tải và xử lý tín hiệu EEG cho {patient_id} - {selected_recording}..."):
            fig = visualize_eeg_signals_safe(
                patient_source_path,
                patient_id,
                int(channels_to_plot),
                actual_outcome=selected_result['Actual'],
                selected_mat_file=selected_recording
            )
            if fig:
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
                st.caption(f'''
                    **Thông tin hiển thị:**
                    - **Patient ID**: {patient_id}
                    - **Recording file**: `{selected_recording}`
                    - **Prediction**: {selected_result['Prediction']}
                    - **Actual Outcome**: {selected_result['Actual']}
                    - **Số kênh hiển thị**: {int(channels_to_plot)} kênh EEG
                    - **Model sử dụng**: {selected_model_display_name} ({model_type})
                    **Tên kênh EEG được đọc từ file .hea**
                ''')
            else:
                st.error(f"❌ Không thể tạo visualization cho recording {selected_recording} của patient {patient_id}.")

def add_eeg_visualization_section(results, all_patient_folders_info, selected_model_display_name, model_type):
    """EEG visualization section: visualize selected .mat files for each patient using dropdown selection."""
    if not results or len([r for r in results if 'Error' not in r['Prediction']]) == 0:
        st.info("Chạy prediction trước để có thể visualize EEG signals.")
        return
    
    st.markdown("---")
    st.header("📊 EEG Signal Visualization")
    
    successful_patients = [r['Patient ID'] for r in results if 'Error' not in r['Prediction']]
    if not successful_patients:
        st.info("Không có patient nào để visualize (tất cả đều gặp lỗi prediction).")
        return
    
    channels_to_plot = st.number_input(
        "Số kênh hiển thị:",
        min_value=1,
        max_value=22,
        value=19,
        help="Số lượng kênh EEG để hiển thị",
        key="channels_input"
    )
    
    for patient_id in successful_patients:
        # Tìm path folder của patient
        patient_source_path = None
        for pid, ppath in all_patient_folders_info:
            if pid == patient_id:
                patient_source_path = ppath
                break
        
        if not patient_source_path or not os.path.isdir(patient_source_path):
            continue
        
        mat_files = [f for f in os.listdir(patient_source_path) if f.endswith('.mat')]
        if not mat_files:
            continue
        
        # Lấy thông tin prediction/actual
        try:
            selected_result = next(r for r in results if r['Patient ID'] == patient_id)
        except Exception as e:
            st.error(f"Error displaying patient info for {patient_id}: {str(e)}")
            continue
        
        with st.expander(f"🧑‍⚕️ Patient {patient_id} ({len(mat_files)} recording(s))", expanded=False):
            st.markdown("### 📈 EEG Signals Display")
            
            # Handle recording selection
            selected_recording = handle_recording_selection(patient_id, mat_files)
            
            # Display EEG visualization if recording is selected
            if selected_recording:
                with st.container():
                    st.markdown(f"**Recording file:** `{selected_recording}`")
                    with st.spinner(f"Đang tải và xử lý tín hiệu EEG cho {patient_id} - {selected_recording}..."):
                        fig = visualize_eeg_signals_safe(
                            patient_source_path,
                            patient_id,
                            int(channels_to_plot),
                            actual_outcome=None,
                            selected_mat_file=selected_recording
                        )
                        if fig:
                            st.pyplot(fig, use_container_width=True)
                            plt.close(fig)
                            st.caption(f'''
                                **Thông tin hiển thị:**
                                - **Patient ID**: {patient_id}
                                - **Recording file**: `{selected_recording}`
                                - **Số kênh hiển thị**: {int(channels_to_plot)} kênh EEG
                                **Tên kênh EEG được đọc từ file .hea**
                            ''')
                        else:
                            st.error(f"❌ Không thể tạo visualization cho recording {selected_recording} của patient {patient_id}.")

def select_n_labels(labels, n):
    """Select n evenly spaced labels from a list"""
    if n >= len(labels):
        return labels
    indices = np.linspace(0, len(labels)-1, n, dtype=int)
    return [labels[i] for i in indices]

def visualize_eeg_signals_safe(patient_folder_path, patient_id, channels_to_plot=19, actual_outcome=None, selected_mat_file=None):
    """Create EEG visualization with proper channel names. Always show full signal. Now supports selecting .mat file."""
    try:
        files = os.listdir(patient_folder_path)
        mat_files = [f for f in files if f.endswith('.mat')]
        if not mat_files:
            st.error(f"No .mat files found in {patient_folder_path}")
            return None
        if selected_mat_file and selected_mat_file in mat_files:
            mat_file = selected_mat_file
        else:
            mat_file = mat_files[0]
        base_name = mat_file.replace('.mat', '')
        recording_location = os.path.join(patient_folder_path, base_name)
        recording_data, channels, sampling_frequency = load_recording_data(recording_location)
        if recording_data is None:
            return None
        recording_data = recording_data.astype(np.float32)  # Thêm dòng này
        num_channels = recording_data.shape[0]
        sig_len = recording_data.shape[1]
        # Luôn hiển thị toàn bộ tín hiệu
        signal_start = 0
        signal_end = sig_len
        # Chọn số kênh
        num_channels_to_plot = min(channels_to_plot, num_channels)
        np.random.seed(2)
        rand_channel_ids = np.random.choice(num_channels, num_channels_to_plot, replace=False)
        rand_channels = [channels[i] if i < len(channels) else f"EEG_{i}" for i in rand_channel_ids]
        rand_signals = recording_data[rand_channel_ids]
        rand_signal_selection = [signal[signal_start:signal_end] for signal in rand_signals]
        num_ticks = 8
        total_time_minutes = (signal_end - signal_start) / (60 * sampling_frequency)
        start_time_minutes = signal_start / (60 * sampling_frequency)
        fig_height = max(14, num_channels_to_plot * 4)
        fig, axs = plt.subplots(num_channels_to_plot, 1, figsize=(16, fig_height), dpi=100)
        if num_channels_to_plot == 1:
            axs = [axs]
        for i in range(num_channels_to_plot):
            axs[i].plot(rand_signal_selection[i], linewidth=0.6, color='#1f77b4', alpha=0.8)
            axs[i].set_title(rand_channels[i], fontsize=16, fontweight='bold', pad=20)
            axs[i].set_xlabel("Time (min)", fontsize=14)
            axs[i].set_ylabel("μV", fontsize=14)
            axs[i].tick_params(axis='both', which='major', labelsize=12)
            axs[i].grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
            axs[i].set_axisbelow(True)
            signal_length = len(rand_signal_selection[i])
            selected_ticks = np.linspace(0, signal_length-1, num_ticks, dtype=int)
            time_labels = np.linspace(start_time_minutes, start_time_minutes + total_time_minutes, num_ticks)
            axs[i].set_xticks(selected_ticks)
            axs[i].set_xticklabels([f"{label:.1f}" for label in time_labels])
            y_data = rand_signal_selection[i]
            y_min = np.min(y_data)
            y_max = np.max(y_data)
            y_range = y_max - y_min
            if y_range == 0:
                # Nếu tín hiệu phẳng, đặt biên độ mặc định ±1
                axs[i].set_ylim(y_min - 1, y_max + 1)
            else:
                y_margin = y_range * 0.05
                axs[i].set_ylim(y_min - y_margin, y_max + y_margin)
            axs[i].ticklabel_format(style='plain', axis='y')
            axs[i].set_facecolor('#fafafa')
        recording_id = base_name.split('_')[-1] if '_' in base_name else base_name
        if actual_outcome:
            outcome = "good" if actual_outcome == 'Good' else "poor"
            title_text = f"Patient {patient_id} with {outcome} outcome from recording {recording_id} (full recording)"
        else:
            title_text = f"Patient {patient_id} from recording {recording_id} (full recording)"
        plt.suptitle(title_text, fontsize=18, fontweight='bold', y=0.96)
        plt.tight_layout()
        plt.subplots_adjust(top=0.93, hspace=0.5, bottom=0.08)
        fig.patch.set_facecolor('white')
        fig.patch.set_edgecolor('lightgray')
        fig.patch.set_linewidth(1)
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
                "pure": "diikhanh/pure-efficientnetv2",
                "improvement": "diikhanh/improvement-efficientnetv2"
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
                    if (model_type == 'stacking'):
                        print("Not load model stacking")
                        # st.session_state.predictor.load_models(selected_model_physical_path, type='stacking')
                    else:
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
            if not st.session_state.predictor.load_models(selected_model_physical_path, type = model_type):
                st.error("Không thể tải models. Prediction bị hủy.")
                return

        with tempfile.TemporaryDirectory() as temp_dir:
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

            st.info("🔍 Đang tìm patient folders...")
            all_patient_folders_info = find_patient_folders(base_extraction_path, debug_mode=False)

            if not all_patient_folders_info:
                st.error("❌ Không tìm thấy patient data hợp lệ trong files upload.")
                return

            st.success(f"✅ Tìm thấy {len(all_patient_folders_info)} patient(s).")

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
                
                outcome_binary, outcome_prob, actual_outcome, segment_outcomes = st.session_state.predictor.predict_single_patient(
                    prediction_input_dir, patient_id, selected_model_physical_path
                )

                if outcome_binary is not None:
                    results.append({
                        'Patient ID': patient_id,
                        'Prediction': 'Good' if outcome_binary == 0 else 'Poor',
                        'Actual': actual_outcome if actual_outcome else "Unknown",
                    })
                else:
                    results.append({
                        'Patient ID': patient_id,
                        'Prediction': 'Error - Prediction Failed',
                        'Actual': actual_outcome if actual_outcome else "N/A", # Keep actual if read
                    })

            progress_bar.empty()
            status_text.empty()
            
            # Store results and patient folders info in session state for persistence
            st.session_state.prediction_results = results
            st.session_state.patient_folders_info = all_patient_folders_info
            st.session_state.selected_model_display_name = selected_model_display_name
            st.session_state.model_type = model_type
            
            if results:
                display_result(results, selected_model_display_name)
                
                # Display all recording files and their predictions with segment values
                st.markdown("---")
                st.header("📋 Chi tiết Prediction cho từng Recording")
                
                for patient_id, patient_original_path in all_patient_folders_info:
                    # Find the result for this patient
                    patient_result = None
                    for result in results:
                        if result['Patient ID'] == patient_id:
                            patient_result = result
                            break
                    
                    if not patient_result:
                        continue
                    
                    # Get all recording files for this patient
                    mat_files = [f for f in os.listdir(patient_original_path) if f.endswith('.mat')]
                    hea_files = [f for f in os.listdir(patient_original_path) if f.endswith('.hea')]
                    
                    if not mat_files:
                        continue
                    
                    with st.expander(f"🧑‍⚕️ Patient {patient_id} - {len(mat_files)} Recording(s)", expanded=True):
                        # Display overall patient prediction
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Patient ID", patient_id)
                        with col2:
                            pred_color = "🟢" if patient_result['Prediction'] == 'Good' else "🔴"
                            st.metric("Overall Prediction", f"{pred_color} {patient_result['Prediction']}")
                        with col3:
                            actual_color = "🟢" if patient_result['Actual'] == 'Good' else ("🔴" if patient_result['Actual'] == 'Poor' else "⚫")
                            st.metric("Actual Outcome", f"{actual_color} {patient_result['Actual']}")
                        
                        # Display individual recording predictions with segment values
                        st.markdown("#### 📊 Chi tiết từng Recording:")
                        
                        # Sort recording files (assuming they have numeric order in filename)
                        mat_files.sort()
                        
                        for i, mat_file in enumerate(mat_files, 1):
                            recording_base = mat_file.replace('.mat', '')
                            
                            # Create a display for each recording with segment values
                            col1, col2, col3 = st.columns([2, 1, 1])
                            with col1:
                                st.markdown(f"**Recording {i}:** `{mat_file}`")
                            with col2:
                                st.markdown(f"**Status:** ✅ Processed")
                            with col3:
                                # Display actual segment values if available
                                print('segment coutcomes neee:',segment_outcomes)
                               
                                    
                                segment_data = segment_outcomes[i-1]
                                
                                segment_label = 'Good' if segment_data == 0 else 'Poor'
                                
                                st.markdown(segment_label)
                                
                        # Show detailed segment outputs if available
                        
            else:
                st.error("❌ Không có kết quả prediction nào!")

    # --- INDEPENDENT EEG VISUALIZATION SECTION ---
    st.header("📊 EEG Signal Visualization")
    
    # Add cleanup button
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🗑️ Xóa dữ liệu tạm", help="Xóa dữ liệu patient đã extract"):
            cleanup_temp_patients()
            if 'extracted_patient_folders' in st.session_state:
                del st.session_state.extracted_patient_folders
            st.rerun()
    
    if not uploaded_files:
        st.info("📁 Vui lòng upload files EEG trước để có thể visualize tín hiệu.")
    else:
        # Extract and store patient folders info for visualization
        if 'extracted_patient_folders' not in st.session_state:
            with tempfile.TemporaryDirectory() as temp_dir:
                base_extraction_path = temp_dir
                for uploaded_file in uploaded_files:
                    file_path = os.path.join(base_extraction_path, uploaded_file.name)
                    with open(file_path, "wb") as f: 
                        f.write(uploaded_file.getbuffer())
                    if uploaded_file.name.endswith('.zip'):
                        try:
                            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                                zip_ref.extractall(base_extraction_path)
                        except Exception as e:
                            st.error(f"❌ Lỗi khi giải nén {uploaded_file.name}: {str(e)}")
                            continue
                
                # Find patient folders and store their data
                all_patient_folders_info = find_patient_folders(base_extraction_path, debug_mode=False)
                
                # Store patient data in session state (copy files to persistent location)
                if all_patient_folders_info:
                    st.session_state.extracted_patient_folders = {}
                    for patient_id, patient_path in all_patient_folders_info:
                        # Create a persistent directory for this patient
                        persistent_patient_dir = os.path.join(current_dir, "temp_patients", patient_id)
                        os.makedirs(persistent_patient_dir, exist_ok=True)
                        
                        # Copy all files from temp directory to persistent directory
                        for item_name in os.listdir(patient_path):
                            src_item = os.path.join(patient_path, item_name)
                            dst_item = os.path.join(persistent_patient_dir, item_name)
                            if os.path.isfile(src_item):
                                shutil.copy2(src_item, dst_item)
                        
                        st.session_state.extracted_patient_folders[patient_id] = persistent_patient_dir
                    
                    st.success(f"✅ Đã chuẩn bị {len(all_patient_folders_info)} patient(s) cho visualization.")
                else:
                    st.warning("⚠️ Không tìm thấy patient data hợp lệ trong files upload.")
        
        # Show visualization if we have patient data
        if 'extracted_patient_folders' in st.session_state and st.session_state.extracted_patient_folders:
            channels_to_plot = st.number_input(
                "Số kênh hiển thị:",
                min_value=1,
                max_value=22,
                value=3,
                help="Số lượng kênh EEG để hiển thị",
                key="channels_input"
            )
            
            for patient_id, patient_source_path in st.session_state.extracted_patient_folders.items():
                if not os.path.isdir(patient_source_path):
                    continue
                
                mat_files = [f for f in os.listdir(patient_source_path) if f.endswith('.mat')]
                if not mat_files:
                    continue
                
                # Get prediction info if available
                prediction_info = None
                if 'prediction_results' in st.session_state:
                    for result in st.session_state.prediction_results:
                        if result['Patient ID'] == patient_id:
                            prediction_info = result
                            break
                
                with st.expander(f"🧑‍⚕️ Patient {patient_id} ({len(mat_files)} recording(s))", expanded=False):
                    st.markdown("### 📈 EEG Signals Display")
                    
                    # Handle recording selection
                    selected_recording = handle_recording_selection(patient_id, mat_files)
                    
                    # Display EEG visualization if recording is selected
                    if selected_recording:
                        with st.container():
                            st.markdown(f"**Recording file:** `{selected_recording}`")
                            with st.spinner(f"Đang tải và xử lý tín hiệu EEG cho {patient_id} - {selected_recording}..."):
                                fig = visualize_eeg_signals_safe(
                                    patient_source_path,
                                    patient_id,
                                    int(channels_to_plot),
                                    actual_outcome=None,
                                    selected_mat_file=selected_recording
                                )
                                if fig:
                                    st.pyplot(fig, use_container_width=True)
                                    plt.close(fig)
                                    st.caption(f'''
                                        **Thông tin hiển thị:**
                                        - **Patient ID**: {patient_id}
                                        - **Recording file**: `{selected_recording}`
                                        - **Số kênh hiển thị**: {int(channels_to_plot)} kênh EEG
                                        **Tên kênh EEG được đọc từ file .hea**
                                    ''')
                                else:
                                    st.error(f"❌ Không thể tạo visualization cho recording {selected_recording} của patient {patient_id}.")

    show_footer()

if __name__ == "__main__":
    main()