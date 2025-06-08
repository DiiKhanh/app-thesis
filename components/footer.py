import streamlit as st

def show_footer():
    st.markdown("---")
    st.markdown("""
    <style>
    .header-container {
        display: flex;
        justify-content: space-between;
        margin-bottom: 20px;
    }
    .left-column {
        flex: 1;
        text-align: left;
        padding: 10px;
        color: #3a5eab;
    }
    .middle-column {
        flex: 1;
        text-align: center;
        padding: 10px;
    }
    .right-column {
        flex: 1;
        text-align: right;
        padding: 10px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="header-container">
        <div class="left-column">
            <div class="text-center">
                <div class="left-text">
                Đề tài khóa luận tốt nghiệp
                </div>
                <div class="text-sub">
                DỰ ĐOÁN KHẢ NĂNG PHỤC HỒI THẦN KINH Ở BỆNH NHÂN HÔN MÊ SAU NGỪNG TIM SỬ DỤNG CÁC MÔ HÌNH HỌC SÂU
                </div>
                </div>
        </div>
        <div class="middle-column">
                <div class="text-center">
                  <div class="left-text">
                  Nhóm sinh viên thực hiện
                  </div>
                  <div class="text-sub">
                  LƯU HIẾU NGÂN – 21520358
                  </div>
                <div class="text-sub">
                    PHẠM DUY KHÁNH - 21522211
                  </div>
                </div>
        </div>
        <div class="right-column">
            <div class="text-center">
                <div class="left-text">
                GIẢNG VIÊN HƯỚNG DẪN
                </div>
                <div class="text-sub">
                ThS. DƯƠNG PHI LONG
                </div>
        </div>
    </div>
    """, unsafe_allow_html=True)