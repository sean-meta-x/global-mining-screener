"""
Simple whitelist authentication for Mining Screener.
Users are stored in auth_users.yaml with SHA-256 hashed passwords.

To add/change users, run:  python manage_users.py
"""
import hashlib
import os
import streamlit as st

try:
    import yaml
except ImportError:
    yaml = None

_AUTH_FILE = os.path.join(os.path.dirname(__file__), "auth_users.yaml")

_LOGIN_CSS = """
<style>
/* ── Page background ── */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0a0f1e 0%, #0f1f2e 50%, #0a1a0f 100%);
    min-height: 100vh;
}
[data-testid="stHeader"] { background: transparent; }

/* ── Card ── */
.login-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(212,175,55,0.25);
    border-radius: 16px;
    padding: 2.5rem 2rem 2rem;
    backdrop-filter: blur(12px);
    box-shadow: 0 8px 40px rgba(0,0,0,0.45), 0 0 0 1px rgba(212,175,55,0.08);
    margin-top: 1rem;
}

/* ── Icon / logo area ── */
.login-icon {
    font-size: 3rem;
    text-align: center;
    margin-bottom: 0.25rem;
    filter: drop-shadow(0 0 12px rgba(212,175,55,0.6));
}
.login-title {
    text-align: center;
    font-size: 1.45rem;
    font-weight: 700;
    color: #f0d060;
    letter-spacing: 0.02em;
    margin-bottom: 0.2rem;
    line-height: 1.3;
}
.login-sub {
    text-align: center;
    font-size: 0.82rem;
    color: #8a9bb0;
    margin-bottom: 1.6rem;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
.login-divider {
    border: none;
    border-top: 1px solid rgba(212,175,55,0.15);
    margin: 0 0 1.4rem;
}

/* ── Input fields ── */
[data-testid="stTextInput"] input,
[data-testid="stTextInput"] input:focus,
[data-testid="stTextInput"] input:active,
[data-testid="stTextInput"] input:-webkit-autofill,
[data-testid="stTextInput"] input:-webkit-autofill:focus {
    background: rgba(20,30,50,0.85) !important;
    border: 1px solid rgba(212,175,55,0.3) !important;
    border-radius: 8px !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    caret-color: #ffffff !important;
    font-size: 0.95rem !important;
    box-shadow: 0 0 0 100px rgba(20,30,50,0.85) inset !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: rgba(212,175,55,0.6) !important;
    outline: none !important;
}
[data-testid="stTextInput"] input::placeholder {
    color: #5a6a7a !important;
    -webkit-text-fill-color: #5a6a7a !important;
}
[data-testid="stTextInput"] label,
[data-testid="stTextInput"] label p {
    color: #c8d8e8 !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.03em;
}

/* ── Login button ── */
[data-testid="stButton"] button[kind="primary"] {
    background: linear-gradient(90deg, #b8860b 0%, #d4af37 50%, #b8860b 100%) !important;
    border: none !important;
    border-radius: 8px !important;
    color: #0a0f1e !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    padding: 0.6rem !important;
    margin-top: 0.4rem;
    transition: opacity 0.2s;
}
[data-testid="stButton"] button[kind="primary"]:hover {
    opacity: 0.88 !important;
}

/* ── Error box ── */
[data-testid="stAlert"] {
    background: rgba(239,68,68,0.12) !important;
    border: 1px solid rgba(239,68,68,0.35) !important;
    border-radius: 8px !important;
    color: #fca5a5 !important;
}

/* ── Footer note ── */
.login-footer {
    text-align: center;
    margin-top: 1.8rem;
}
.login-footer-company {
    font-size: 1rem;
    font-weight: 700;
    color: #d4af37;
    letter-spacing: 0.06em;
    margin-bottom: 0.3rem;
}
.login-footer-restricted {
    font-size: 0.82rem;
    color: #8a9bb0;
    letter-spacing: 0.04em;
}
</style>
"""


def _hash(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def _load_users() -> dict:
    if yaml is None or not os.path.exists(_AUTH_FILE):
        return {}
    with open(_AUTH_FILE, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("users", {})


def check_auth() -> None:
    """
    Gate the entire app behind a login page.
    Call at the very top of app.py before any other st.* calls.
    If auth_users.yaml does not exist, access is open (no login required).
    """
    users = _load_users()
    if not users:
        return

    if st.session_state.get("_auth_ok"):
        return

    # ── Login page ────────────────────────────────────────────────────
    st.set_page_config(
        page_title="Undervalued Mining Stock Screener",
        page_icon="⛏️",
        layout="centered",
    )
    st.markdown(_LOGIN_CSS, unsafe_allow_html=True)

    _, col, _ = st.columns([1, 1.5, 1])
    with col:
        st.markdown(
            '<div class="login-card">'
            '<div class="login-icon">⛏️</div>'
            '<div class="login-title">Undervalued<br>Mining Stock Screener</div>'
            '<div class="login-sub">Authorized Access</div>'
            '<hr class="login-divider">',
            unsafe_allow_html=True,
        )

        username = st.text_input("Username / 用户名", key="_login_user")
        password = st.text_input("Password / 密码", type="password", key="_login_pass")

        if st.button("Sign In  →", use_container_width=True, type="primary"):
            user = users.get(username)
            if user and user.get("password_hash") == _hash(password):
                st.session_state._auth_ok   = True
                st.session_state._auth_user = username
                st.session_state._auth_name = user.get("display_name", username)
                st.rerun()
            else:
                st.error("Invalid username or password / 用户名或密码错误")

        st.markdown(
            '</div>'
            '<div class="login-footer">'
            '<div class="login-footer-company">🔒 Lingbao Gold — Internal Use Only</div>'
            '<div class="login-footer-restricted">Access restricted to authorized users only</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    st.stop()


def current_user() -> str:
    return st.session_state.get("_auth_name", "")


def logout() -> None:
    for k in ("_auth_ok", "_auth_user", "_auth_name"):
        st.session_state.pop(k, None)
    st.rerun()
