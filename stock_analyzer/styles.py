def app_shell_css() -> str:
    return """
    <style>
    .stApp {
        background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
        color: #FAFAFA;
        font-family: 'Segoe UI', sans-serif;
    }
    [data-testid="stSidebar"] > div:first-child {
        background: linear-gradient(135deg, rgba(32, 50, 73, 0.6), rgba(57, 108, 139, 0.5));
        backdrop-filter: blur(16px);
        border-radius: 20px;
        padding: 25px 20px;
        margin: 10px;
    }
    .floating-btn {
        position: fixed;
        bottom: 30px;
        right: 30px;
        background-color: #4e9af1;
        color: white;
        border: none;
        padding: 15px 18px;
        border-radius: 50px;
        font-size: 20px;
        box-shadow: 0 5px 15px rgba(78, 154, 241, 0.4);
        cursor: pointer;
        z-index: 9999;
    }
    </style>
"""


def title_block_html() -> str:
    return """
    <style>
    .animated-title {
        font-size: 48px;
        text-align: center;
        margin-top: 30px;
        color: #FAFAFA;
        animation: fadeInSlide 2s ease-in-out forwards;
        opacity: 0;
    }

    @keyframes fadeInSlide {
        0% {
            opacity: 0;
            transform: translateY(-40px);
        }
        100% {
            opacity: 1;
            transform: translateY(0);
        }
    }
    </style>

    <h1 class="animated-title">📊 Stock Sentiment Analyzer</h1>
"""


def hide_streamlit_chrome_css() -> str:
    return """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
"""
