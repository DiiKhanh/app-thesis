import streamlit as st
from PIL import Image
import base64
from io import BytesIO

# Load image and convert to base64
def get_base64_image(image_path):
    img = Image.open(image_path)
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

# Base64 version of your local image
logo_local_base64 = get_base64_image("assets/logo-httt.png")

def show_header():
    st.markdown("""
    <style>
    .header-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 10px 20px;
        background-color: #fff;
        border-bottom: 2px solid #444;
        color: #ffffff;
    }
    .left-section, .right-section {
        display: flex;
        align-items: center;
    }
    .left-section img, .right-section img {
        height: 80px;
        margin-right: 15px;
        margin-left: 15px;
        filter: brightness(0.9);
    }
    .left-text, .right-text {
        font-size: 18px;
        font-weight: bold;
        line-height: 1.4;
        color: #019ed8;
    }
    .text-sub {
        font-size: 16px;
        font-weight: bold;
        line-height: 1.4;
        color: #3a5eab;
    }
    .text-center {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }
    </style>
    <div class="header-container">
        <div class="left-section">
            <img src="{logo_local_base64}" alt="Logo Khoa HTTT">
            <div class="text-center">
                <div class="left-text">
                TRƯỜNG ĐẠI HỌC CÔNG NGHỆ THÔNG TIN - ĐHQG-HCM
                </div>
                <div class="text-sub">KHOA HỆ THỐNG THÔNG TIN</div>
            </div>
        </div>
        <div class="right-section">
            <div class="text-center">
                <div class="right-text">
                CÔNG CỤ DỰ ĐOÁN KHẢ NĂNG PHỤC HỒI
                </div>
                <div class="text-sub">
                THẦN KINH Ở BỆNH NHÂN HÔN MÊ SAU NGỪNG TIM
                </div>
                </div>
            <img src="https://cdn-icons-png.flaticon.com/512/9851/9851782.png" alt="Logo Đề tài">
        </div>
    </div>
    """, unsafe_allow_html=True)