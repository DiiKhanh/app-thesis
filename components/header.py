import streamlit as st

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
            <img src="https://fe-ecg-tool.onrender.com/assets/Logo-Khoa-HTTT-Ca8OdENF.png" alt="Logo Trường">
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