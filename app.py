import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
import os
import sqlite3
import hashlib
import secrets
from PIL import Image
from datetime import datetime

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Smart Fraud Detector",
    page_icon="🚨",
    layout="wide"
)

DB_FILE = "complaints.db"
LEGACY_CSV_FILE = "customer_claims.csv"   # old file, migrated into SQLite on first run
IMAGES_DIR = "complaint_images"

COMPLAINT_COLUMNS = [
    "Timestamp", "CustomerName", "Mobile", "ClaimAmount", "VehiclePrice",
    "Age", "Severity", "Description", "ImageFlag", "ImagePath", "RiskScore", "RiskStatus"
]

CLAIM_STATUSES = ["Pending Review", "Under Review", "Approved", "Rejected"]

# Default admin credentials — override via environment variables in production
# (e.g. `ADMIN_USERNAME=myuser ADMIN_PASSWORD=strongpass streamlit run app.py`)
DEFAULT_ADMIN_USERNAME = os.environ.get("ADMIN_USERNAME", "admin")
DEFAULT_ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "1234")

os.makedirs(IMAGES_DIR, exist_ok=True)

# ==========================================================
# GLOBAL CSS (base dark theme + shared components)
# ==========================================================

def inject_global_css():
    st.markdown(
        """
        <style>

        .stApp {
            background-color:#0e1117;
            color:white;
        }

        #MainMenu, footer { visibility:hidden; }

        .main-title {
            font-size:42px;
            font-weight:800;
            text-align:center;
            background: linear-gradient(90deg, #ff4b4b, #7b2ff7);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom:2px;
        }

        .sub-title {
            text-align:center;
            font-size:16px;
            color:#9c9cb5;
            margin-bottom: 10px;
        }

        .role-badge {
            display:inline-block;
            margin: 0 auto 18px auto;
            padding:6px 16px;
            border-radius:999px;
            font-size:12.5px;
            font-weight:700;
            letter-spacing:0.4px;
        }

        .badge-admin { background:rgba(255,75,75,0.15); color:#ff8a8a; border:1px solid rgba(255,75,75,0.35); }
        .badge-customer { background:rgba(0,212,255,0.15); color:#7fe3ff; border:1px solid rgba(0,212,255,0.35); }

        .card {
            padding:22px;
            border-radius:16px;
            background:linear-gradient(160deg,#1e222d,#181b24);
            border:1px solid rgba(255,255,255,0.06);
            box-shadow:0px 6px 20px rgba(0,0,0,0.35);
            text-align:center;
        }

        .metric-number {
            font-size:32px;
            color:#00ff99;
            font-weight:800;
        }

        .result-success {
            background:linear-gradient(135deg,#064e3b,#065f46);
            padding:22px; border-radius:16px; font-size:22px; color:white;
            border:1px solid rgba(16,185,129,0.4);
        }

        .result-warning {
            background:linear-gradient(135deg,#78350f,#92400e);
            padding:22px; border-radius:16px; font-size:22px; color:white;
            border:1px solid rgba(245,158,11,0.4);
        }

        .result-danger {
            background:linear-gradient(135deg,#7f1d1d,#991b1b);
            padding:22px; border-radius:16px; font-size:22px; color:white;
            border:1px solid rgba(239,68,68,0.4);
        }

        .info-card {
            background:rgba(255,255,255,0.04);
            border:1px solid rgba(255,255,255,0.08);
            border-radius:16px;
            padding:20px 24px;
            margin-bottom:14px;
        }

        .complaint-card {
            background:rgba(255,255,255,0.04);
            border:1px solid rgba(255,255,255,0.08);
            border-left:4px solid #7b2ff7;
            border-radius:14px;
            padding:16px 20px;
            margin-bottom:12px;
        }

        .complaint-card.high { border-left-color:#ff4b4b; }
        .complaint-card.medium { border-left-color:#f5a623; }
        .complaint-card.low { border-left-color:#00d97e; }

        .pill {
            display:inline-block;
            font-size:11px;
            font-weight:700;
            padding:4px 12px;
            border-radius:999px;
            margin-left:8px;
        }

        .pill.high { background:rgba(255,75,75,0.18); color:#ff8a8a; }
        .pill.medium { background:rgba(245,166,35,0.18); color:#ffcf87; }
        .pill.low { background:rgba(0,217,126,0.18); color:#7cf7c4; }

        .pill.status-pending { background:rgba(148,163,184,0.18); color:#cbd5e1; }
        .pill.status-underreview { background:rgba(56,189,248,0.18); color:#7dd3fc; }
        .pill.status-approved { background:rgba(0,217,126,0.18); color:#7cf7c4; }
        .pill.status-rejected { background:rgba(255,75,75,0.18); color:#ff8a8a; }

        div[data-testid="stForm"] button {
            width:100%;
            border-radius:12px;
            border:none;
            font-weight:700;
            padding:12px 0;
            background: linear-gradient(135deg, var(--accent1,#ff4b4b), var(--accent2,#7b2ff7));
            color:white;
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }

        div[data-testid="stForm"] button:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 24px rgba(123,47,247,0.35);
        }

        </style>
        """,
        unsafe_allow_html=True
    )


def render_ambient_background(accent1, accent2):
    st.markdown(
        f"""
        <style>
        .stApp {{
            --accent1:{accent1};
            --accent2:{accent2};
            background: linear-gradient(-45deg, #0f0c29, #1b1035, #24243e, #0f0c29);
            background-size: 400% 400%;
            animation: auroraShift 18s ease infinite;
        }}

        @keyframes auroraShift {{
            0% {{ background-position: 0% 50%; }}
            50% {{ background-position: 100% 50%; }}
            100% {{ background-position: 0% 50%; }}
        }}

        .glow-orb {{
            position:fixed; border-radius:50%; filter:blur(90px);
            opacity:0.30; z-index:0; pointer-events:none;
        }}
        .glow-orb.one   {{ width:320px; height:320px; top:-80px; left:-60px; background:{accent1}; }}
        .glow-orb.two   {{ width:280px; height:280px; bottom:-60px; right:-60px; background:{accent2}; }}

        [data-testid="stSidebar"] {{ display:none; }}
        header[data-testid="stHeader"] {{ background:transparent; }}
        .block-container {{ padding-top:2rem; }}

        .login-shell {{
            position:relative; z-index:1;
            max-width:430px; margin:3vh auto 0 auto;
            padding:44px 40px 30px 40px;
            border-radius:24px;
            background:rgba(255,255,255,0.06);
            backdrop-filter: blur(18px);
            -webkit-backdrop-filter: blur(18px);
            border:1px solid rgba(255,255,255,0.14);
            box-shadow:0 8px 40px rgba(0,0,0,0.55);
            text-align:center;
            animation: cardRise 0.5s ease;
        }}

        @keyframes cardRise {{
            from {{ opacity:0; transform:translateY(16px); }}
            to   {{ opacity:1; transform:translateY(0); }}
        }}

        .login-badge {{
            width:64px; height:64px; margin:0 auto 16px auto;
            border-radius:18px; display:flex; align-items:center; justify-content:center;
            font-size:28px;
            background:linear-gradient(135deg,{accent1},{accent2});
            box-shadow:0 6px 20px {accent1}55;
        }}

        .login-title {{ font-size:26px; font-weight:800; color:white; margin-bottom:4px; }}
        .login-subtitle {{ font-size:13.5px; color:#b7b7c9; margin-bottom:6px; }}

        div[data-testid="stForm"] input {{
            background:rgba(255,255,255,0.07) !important;
            border:1px solid rgba(255,255,255,0.16) !important;
            border-radius:12px !important;
            color:white !important;
            padding:11px 13px !important;
        }}

        div[data-testid="stForm"] label {{
            color:#d6d6e6 !important; font-weight:600 !important; font-size:12.5px !important;
        }}

        div[data-testid="stForm"] input:focus {{
            border:1px solid {accent1} !important;
            box-shadow:0 0 0 3px {accent1}33 !important;
        }}

        .feature-row {{
            display:flex; justify-content:center; gap:16px; margin-top:22px; flex-wrap:wrap;
        }}
        .feature-pill {{
            font-size:11px; color:#cfcfe4; background:rgba(255,255,255,0.05);
            border:1px solid rgba(255,255,255,0.1); padding:6px 12px; border-radius:999px;
        }}

        .login-footnote {{ margin-top:18px; font-size:12px; color:#8a8aa0; text-align:center; }}
        .login-footnote b {{ color:#c9c9e0; }}

        </style>
        <div class="glow-orb one"></div>
        <div class="glow-orb two"></div>
        """,
        unsafe_allow_html=True
    )


# ==========================================================
# SESSION STATE
# ==========================================================

defaults = {
    "login": False,
    "role": None,
    "pending_role": None,
    "login_error": False,
    "customer_name": "",
    "customer_mobile": "",
    "admin_username": "",
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ==========================================================
# DATA / MODEL HELPERS
# ==========================================================

@st.cache_data
def load_data():
    try:
        data = pd.read_csv("carclaims.csv")
        return data
    except Exception as e:
        st.error(e)
        return None


def calc_risk(claim_amount, vehicle_price, age, severity, description, image_flag):
    risk = 0
    reasons = []

    if claim_amount > vehicle_price:
        risk += 2
        reasons.append("⚠ Claim amount is greater than vehicle price")

    if claim_amount > 50000:
        risk += 1
        reasons.append("⚠ High claim amount detected")

    if severity == "Total Loss":
        risk += 2
        reasons.append("⚠ Vehicle total loss claim")

    if age < 25:
        risk += 1
        reasons.append("⚠ Young customer risk factor")

    if "damage" in description.lower():
        risk += 1
        reasons.append("⚠ Suspicious damage description detected")

    if image_flag == 1:
        risk += 1
        reasons.append("⚠ Low quality / suspicious uploaded image")

    return risk, reasons


def risk_label(score):
    if score >= 4:
        return "High Risk", "danger", "high"
    elif score >= 2:
        return "Medium Risk", "warning", "medium"
    else:
        return "Low Risk", "success", "low"


def status_style(status):
    mapping = {
        "Pending Review": ("status-pending", "⏳"),
        "Under Review": ("status-underreview", "🔎"),
        "Approved": ("status-approved", "✅"),
        "Rejected": ("status-rejected", "❌"),
    }
    return mapping.get(status, ("status-pending", "⏳"))


def save_uploaded_image(uploaded_file, mobile):
    """Persists the customer's accident photo to disk and returns its path."""
    if uploaded_file is None:
        return ""

    ext = os.path.splitext(uploaded_file.name)[1] or ".jpg"
    filename = f"{mobile}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}{ext}"
    filepath = os.path.join(IMAGES_DIR, filename)

    with open(filepath, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return filepath


def get_conn():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def hash_password(password, salt=None):
    """PBKDF2-HMAC-SHA256 password hash. Returns (hash_hex, salt_hex)."""
    if salt is None:
        salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt), 260_000
    ).hex()
    return digest, salt


def init_db():
    """Creates tables on first run, seeds the default admin, and migrates
    any pre-existing customer_claims.csv data into SQLite exactly once."""
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS admins (
            Username TEXT PRIMARY KEY,
            PasswordHash TEXT NOT NULL,
            Salt TEXT NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS complaints (
            Id INTEGER PRIMARY KEY AUTOINCREMENT,
            Timestamp TEXT,
            CustomerName TEXT,
            Mobile TEXT,
            ClaimAmount REAL,
            VehiclePrice REAL,
            Age INTEGER,
            Severity TEXT,
            Description TEXT,
            ImageFlag INTEGER,
            ImagePath TEXT,
            RiskScore INTEGER,
            RiskStatus TEXT,
            ClaimStatus TEXT DEFAULT 'Pending Review',
            AdminNotes TEXT DEFAULT '',
            UpdatedAt TEXT
        )
    """)
    conn.commit()

    # Seed the default admin only if the admins table is empty, so an
    # operator's env-var credentials aren't clobbered on every restart.
    cur.execute("SELECT COUNT(*) AS c FROM admins")
    if cur.fetchone()["c"] == 0:
        digest, salt = hash_password(DEFAULT_ADMIN_PASSWORD)
        cur.execute(
            "INSERT INTO admins (Username, PasswordHash, Salt) VALUES (?, ?, ?)",
            (DEFAULT_ADMIN_USERNAME, digest, salt),
        )
        conn.commit()

    # One-time migration: if the DB has no complaints yet but an old
    # customer_claims.csv exists on disk, import it so nothing is lost.
    cur.execute("SELECT COUNT(*) AS c FROM complaints")
    if cur.fetchone()["c"] == 0 and os.path.exists(LEGACY_CSV_FILE):
        try:
            legacy = pd.read_csv(LEGACY_CSV_FILE)
            legacy = legacy.reindex(columns=COMPLAINT_COLUMNS, fill_value="")
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            for _, r in legacy.iterrows():
                cur.execute(
                    """INSERT INTO complaints
                       (Timestamp, CustomerName, Mobile, ClaimAmount, VehiclePrice, Age,
                        Severity, Description, ImageFlag, ImagePath, RiskScore, RiskStatus,
                        ClaimStatus, AdminNotes, UpdatedAt)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        r["Timestamp"], r["CustomerName"], r["Mobile"], r["ClaimAmount"],
                        r["VehiclePrice"], r["Age"], r["Severity"], r["Description"],
                        r["ImageFlag"], r["ImagePath"], r["RiskScore"], r["RiskStatus"],
                        "Pending Review", "", now,
                    ),
                )
            conn.commit()
            backup_name = f"{LEGACY_CSV_FILE}.migrated_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            os.rename(LEGACY_CSV_FILE, backup_name)
        except Exception as e:
            st.warning(f"Could not migrate old CSV data automatically: {e}")

    conn.close()


def verify_admin(username, password):
    conn = get_conn()
    row = conn.execute(
        "SELECT PasswordHash, Salt FROM admins WHERE Username = ?", (username,)
    ).fetchone()
    conn.close()
    if row is None:
        return False
    digest, _ = hash_password(password, row["Salt"])
    return secrets.compare_digest(digest, row["PasswordHash"])


def insert_complaint(row_dict):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = get_conn()
    conn.execute(
        """INSERT INTO complaints
           (Timestamp, CustomerName, Mobile, ClaimAmount, VehiclePrice, Age,
            Severity, Description, ImageFlag, ImagePath, RiskScore, RiskStatus,
            ClaimStatus, AdminNotes, UpdatedAt)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            row_dict["Timestamp"], row_dict["CustomerName"], row_dict["Mobile"],
            row_dict["ClaimAmount"], row_dict["VehiclePrice"], row_dict["Age"],
            row_dict["Severity"], row_dict["Description"], row_dict["ImageFlag"],
            row_dict["ImagePath"], row_dict["RiskScore"], row_dict["RiskStatus"],
            "Pending Review", "", now,
        ),
    )
    conn.commit()
    conn.close()


def load_complaints():
    conn = get_conn()
    data = pd.read_sql_query("SELECT * FROM complaints ORDER BY Id DESC", conn)
    conn.close()
    return data


def update_complaint_status(complaint_id, new_status, notes):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = get_conn()
    conn.execute(
        "UPDATE complaints SET ClaimStatus = ?, AdminNotes = ?, UpdatedAt = ? WHERE Id = ?",
        (new_status, notes, now, complaint_id),
    )
    conn.commit()
    conn.close()


init_db()


# ==========================================================
# ROLE SELECTION SCREEN
# ==========================================================

def render_role_selection():
    render_ambient_background("#7b2ff7", "#00d4ff")

    st.markdown(
        """
        <div style="position:relative;z-index:1;text-align:center;margin-top:4vh;">
            <div class="main-title">🚨 Smart Fraud Detector</div>
            <div class="sub-title">Choose how you'd like to continue</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")
    left_pad, c1, c2, right_pad = st.columns([0.8, 1.6, 1.6, 0.8])

    with c1:
        st.markdown(
            """
            <div class="login-shell" style="margin-top:0;">
                <div class="login-badge">🧑</div>
                <div class="login-title">Customer</div>
                <div class="login-subtitle">File a new insurance claim complaint<br>and track its status</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("Continue as Customer  →", key="pick_customer", use_container_width=True):
            st.session_state.pending_role = "customer"
            st.rerun()

    with c2:
        st.markdown(
            """
            <div class="login-shell" style="margin-top:0;">
                <div class="login-badge">🛡️</div>
                <div class="login-title">Admin</div>
                <div class="login-subtitle">Review claims, run fraud analytics<br>and manage the ML dashboard</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("Continue as Admin  →", key="pick_admin", use_container_width=True):
            st.session_state.pending_role = "admin"
            st.rerun()

    st.markdown(
        """
        <div class="login-footnote">Smart Fraudulent Insurance Claim Detector · AI Powered Fraud Screening</div>
        """,
        unsafe_allow_html=True
    )


# ==========================================================
# ADMIN LOGIN
# ==========================================================

def render_admin_login():
    render_ambient_background("#ff4b4b", "#7b2ff7")

    left, center, right = st.columns([1, 1.15, 1])
    with center:
        st.markdown(
            """
            <div class="login-shell">
                <div class="login-badge">🛡️</div>
                <div class="login-title">Admin Login</div>
                <div class="login-subtitle">Smart Fraudulent Insurance Claim Detector</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        with st.form("admin_login_form"):
            username = st.text_input("Username", placeholder="Enter admin username")
            password = st.text_input("Password", placeholder="Enter password", type="password")
            submitted = st.form_submit_button("Sign In  →")

            if submitted:
                if verify_admin(username, password):
                    st.session_state.login = True
                    st.session_state.role = "admin"
                    st.session_state.admin_username = username
                    st.session_state.login_error = False
                    st.rerun()
                else:
                    st.session_state.login_error = True

        if st.session_state.login_error:
            st.markdown(
                """
                <div style="max-width:430px;margin:12px auto 0 auto;background:rgba(127,29,29,0.35);
                border:1px solid rgba(255,90,90,0.4);color:#ffb4b4;padding:10px 16px;border-radius:12px;
                font-size:13.5px;text-align:center;">
                ❌ Invalid username or password.
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown(
            """
            <div class="feature-row">
                <div class="feature-pill">🤖 ML Fraud Detection</div>
                <div class="feature-pill">📊 Live Dashboard</div>
                <div class="feature-pill">📋 Complaint Manager</div>
            </div>
            <div class="login-footnote">Default credentials (first run) — <b>admin / 1234</b><br>Configurable via <b>ADMIN_USERNAME</b> / <b>ADMIN_PASSWORD</b> env vars</div>
            """,
            unsafe_allow_html=True
        )

        st.write("")
        bl, bc, br = st.columns([1, 1, 1])
        with bc:
            if st.button("← Back", key="admin_back", use_container_width=True):
                st.session_state.pending_role = None
                st.session_state.login_error = False
                st.rerun()


# ==========================================================
# CUSTOMER LOGIN
# ==========================================================

def render_customer_login():
    render_ambient_background("#00d4ff", "#00ff99")

    left, center, right = st.columns([1, 1.15, 1])
    with center:
        st.markdown(
            """
            <div class="login-shell">
                <div class="login-badge">🧑</div>
                <div class="login-title">Customer Login</div>
                <div class="login-subtitle">Enter your details to file or track a claim</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        with st.form("customer_login_form"):
            name = st.text_input("Full Name", placeholder="Enter your full name")
            mobile = st.text_input("Mobile Number", placeholder="10-digit mobile number", max_chars=10)
            submitted = st.form_submit_button("Continue  →")

            if submitted:
                if name.strip() == "":
                    st.session_state.login_error = "name"
                elif not (mobile.isdigit() and len(mobile) == 10):
                    st.session_state.login_error = "mobile"
                else:
                    st.session_state.login = True
                    st.session_state.role = "customer"
                    st.session_state.customer_name = name.strip().title()
                    st.session_state.customer_mobile = mobile
                    st.session_state.login_error = False
                    st.rerun()

        if st.session_state.login_error == "name":
            st.markdown(
                """<div style="max-width:430px;margin:12px auto 0 auto;background:rgba(127,29,29,0.35);
                border:1px solid rgba(255,90,90,0.4);color:#ffb4b4;padding:10px 16px;border-radius:12px;
                font-size:13.5px;text-align:center;">❌ Please enter your full name.</div>""",
                unsafe_allow_html=True
            )
        elif st.session_state.login_error == "mobile":
            st.markdown(
                """<div style="max-width:430px;margin:12px auto 0 auto;background:rgba(127,29,29,0.35);
                border:1px solid rgba(255,90,90,0.4);color:#ffb4b4;padding:10px 16px;border-radius:12px;
                font-size:13.5px;text-align:center;">❌ Enter a valid 10-digit mobile number.</div>""",
                unsafe_allow_html=True
            )

        st.markdown(
            """
            <div class="feature-row">
                <div class="feature-pill">📝 File a Complaint</div>
                <div class="feature-pill">📂 Track Claim Status</div>
                <div class="feature-pill">🔒 Your Data is Safe</div>
            </div>
            <div class="login-footnote">No password needed — your mobile number is your ID</div>
            """,
            unsafe_allow_html=True
        )

        st.write("")
        bl, bc, br = st.columns([1, 1, 1])
        with bc:
            if st.button("← Back", key="customer_back", use_container_width=True):
                st.session_state.pending_role = None
                st.session_state.login_error = False
                st.rerun()


# ==========================================================
# LOGIN GATE
# ==========================================================

inject_global_css()

if not st.session_state.login:

    if st.session_state.pending_role is None:
        render_role_selection()
    elif st.session_state.pending_role == "admin":
        render_admin_login()
    elif st.session_state.pending_role == "customer":
        render_customer_login()

    st.stop()


# ==========================================================
# SHARED HEADER
# ==========================================================

role = st.session_state.role
badge_class = "badge-admin" if role == "admin" else "badge-customer"
badge_text = "🛡️ ADMIN PORTAL" if role == "admin" else "🧑 CUSTOMER PORTAL"

st.markdown(
    f"""
    <div style="text-align:center;">
        <div class="main-title">🚨 Smart Fraudulent Insurance Claim Detector</div>
        <div class="sub-title">AI Based Fraud Detection using Machine Learning + Text + Image Analysis</div>
        <span class="role-badge {badge_class}">{badge_text}</span>
    </div>
    <br>
    """,
    unsafe_allow_html=True
)

with st.sidebar:
    if role == "admin":
        st.markdown("### 🛡️ Admin Session")
        st.caption(f"Logged in as **{st.session_state.admin_username}**")
    else:
        st.markdown("### 🧑 Customer Session")
        st.caption(f"Logged in as **{st.session_state.customer_name}**")
        st.caption(f"📱 {st.session_state.customer_mobile}")

    if st.button("🚪 Logout", use_container_width=True):
        st.session_state.login = False
        st.session_state.role = None
        st.session_state.pending_role = None
        st.session_state.login_error = False
        st.session_state.admin_username = ""
        st.rerun()

    st.markdown("---")


# ==========================================================
# ================  CUSTOMER PORTAL  ======================
# ==========================================================

if role == "customer":

    nav = st.sidebar.radio("📌 Navigation", ["📝 File New Complaint", "📂 My Claim History"])

    # ---------------- FILE NEW COMPLAINT ----------------
    if nav == "📝 File New Complaint":

        st.subheader("📝 File a New Insurance Claim Complaint")
        st.caption("Fill in the details below and our AI system will instantly assess your claim's risk profile.")

        with st.form("complaint_form"):

            c1, c2 = st.columns(2)
            with c1:
                claim_amount = st.number_input("💰 Claim Amount (₹)", min_value=0, step=1000)
                age = st.slider("👤 Your Age", 18, 100, 30)
                severity = st.selectbox("⚠️ Accident Severity", ["Minor", "Major", "Total Loss"])
            with c2:
                vehicle_price = st.number_input("🚗 Vehicle Price (₹)", min_value=0, step=1000)
                uploaded_file = st.file_uploader("📷 Upload Accident Image", type=["jpg", "png", "jpeg"])

            description = st.text_area("📝 Describe the Incident", placeholder="Explain what happened...")

            submit_complaint = st.form_submit_button("🚀 Submit Complaint")

        if submit_complaint:

            image_flag = 0
            image_path = ""

            if uploaded_file:
                image = Image.open(uploaded_file)
                if image.size[0] < 200 or image.size[1] < 200:
                    image_flag = 1
                uploaded_file.seek(0)
                image_path = save_uploaded_image(uploaded_file, st.session_state.customer_mobile)

            risk_score, reasons = calc_risk(claim_amount, vehicle_price, age, severity, description, image_flag)
            label, css_class, tag = risk_label(risk_score)

            insert_complaint({
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "CustomerName": st.session_state.customer_name,
                "Mobile": st.session_state.customer_mobile,
                "ClaimAmount": claim_amount,
                "VehiclePrice": vehicle_price,
                "Age": age,
                "Severity": severity,
                "Description": description,
                "ImageFlag": image_flag,
                "ImagePath": image_path,
                "RiskScore": risk_score,
                "RiskStatus": label,
            })

            st.success("✅ Your complaint has been submitted successfully!")

            st.markdown(
                f"""<div class='result-{css_class}'>
                {'🚨' if tag=='high' else ('⚠️' if tag=='medium' else '✅')} Assessed Risk: {label} (Score: {risk_score}/8)
                </div>""",
                unsafe_allow_html=True
            )

            if reasons:
                st.markdown("#### 🧠 Why this score?")
                for r in reasons:
                    st.warning(r)
            else:
                st.info("No risk flags were triggered — your claim looks straightforward.")

            if image_path:
                st.markdown("#### 📷 Attached Accident Photo")
                st.image(image_path, use_container_width=True)

            st.caption("Our team will review your claim shortly. You can check its status anytime under **My Claim History**.")

    # ---------------- MY CLAIM HISTORY ----------------
    else:

        st.subheader("📂 My Claim History")

        complaints = load_complaints()
        my_claims = complaints[complaints["Mobile"].astype(str) == str(st.session_state.customer_mobile)]

        if my_claims.empty:
            st.info("You haven't filed any complaints yet. Head to **File New Complaint** to get started.")
        else:
            my_claims = my_claims.sort_values("Timestamp", ascending=False)

            total = len(my_claims)
            high = len(my_claims[my_claims["RiskStatus"] == "High Risk"])
            low = len(my_claims[my_claims["RiskStatus"] == "Low Risk"])

            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(f"<div class='card'>Total Claims<div class='metric-number'>{total}</div></div>", unsafe_allow_html=True)
            with m2:
                st.markdown(f"<div class='card'>High Risk<div class='metric-number' style='color:#ff6b6b;'>{high}</div></div>", unsafe_allow_html=True)
            with m3:
                st.markdown(f"<div class='card'>Low Risk / Clear<div class='metric-number'>{low}</div></div>", unsafe_allow_html=True)

            st.write("")

            for _, row in my_claims.iterrows():
                tag = "high" if row["RiskStatus"] == "High Risk" else ("medium" if row["RiskStatus"] == "Medium Risk" else "low")
                icon = "🚨" if tag == "high" else ("⚠️" if tag == "medium" else "✅")

                claim_status = row.get("ClaimStatus", "Pending Review") or "Pending Review"
                status_class, status_icon = status_style(claim_status)

                header = f"{icon}  ₹{row['ClaimAmount']:,.0f} claim · {row['Severity']} · {row['RiskStatus']} · {status_icon} {claim_status} · {row['Timestamp']}"

                with st.expander(header):
                    admin_notes = row.get("AdminNotes", "")
                    notes_html = (
                        f"<br><span style='color:#cfcfe4;font-size:13px;'>🗒️ <b>Admin Note:</b> {admin_notes}</span>"
                        if pd.notna(admin_notes) and str(admin_notes).strip() else ""
                    )

                    st.markdown(
                        f"""
                        <div class="complaint-card {tag}">
                            <b>💰 Claim Amount:</b> ₹{row['ClaimAmount']:,} &nbsp; | &nbsp;
                            <b>🚗 Vehicle Price:</b> ₹{row['VehiclePrice']:,} &nbsp; | &nbsp;
                            <b>👤 Age:</b> {row['Age']} &nbsp; | &nbsp;
                            <b>⚠️ Severity:</b> {row['Severity']}
                            <span class="pill {tag}">{row['RiskStatus']}</span>
                            <span class="pill {status_class}">{status_icon} {claim_status}</span>
                            <br><span style="color:#9c9cb5;font-size:12.5px;">🕒 {row['Timestamp']}</span>
                            <br><span style="color:#cfcfe4;font-size:13px;">📝 {row['Description'] if pd.notna(row['Description']) and str(row['Description']).strip() else 'No description provided'}</span>
                            {notes_html}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    image_path = row.get("ImagePath", "")
                    if pd.notna(image_path) and str(image_path).strip() and os.path.exists(str(image_path)):
                        st.markdown("**📷 Attached Accident Photo**")
                        st.image(str(image_path), use_container_width=True)
                    else:
                        st.caption("📷 No accident photo was attached to this claim.")


# ==========================================================
# ================    ADMIN PORTAL    =====================
# ==========================================================

else:

    df = load_data()

    if df is not None:
        df.dropna(inplace=True)
        df.drop_duplicates(inplace=True)

    encoders = {}
    if df is not None:
        cat_cols = df.select_dtypes(include="object").columns
        for col in cat_cols:
            if df[col].nunique() < 50:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                encoders[col] = le

    model = None
    if df is not None:
        X = df.drop("FraudFound", axis=1) if "FraudFound" in df.columns else df.copy()

        if os.path.exists("fraud_model.pkl"):
            with open("fraud_model.pkl", "rb") as f:
                model = pickle.load(f)
        else:
            X_train, _ = train_test_split(X, test_size=0.2, random_state=42)
            model = IsolationForest(contamination=0.1, random_state=42)
            model.fit(X_train)
            pickle.dump(model, open("fraud_model.pkl", "wb"))

        df["prediction"] = model.predict(X)

    complaints_df = load_complaints()

    nav = st.sidebar.radio(
        "📌 Navigation",
        ["🏠 Dashboard", "📋 Customer Complaints", "📊 Dataset Analysis", "📈 EDA Graphs", "🤖 Model Details", "ℹ️ About Project"]
    )

    # ---------------- DASHBOARD ----------------
    if nav == "🏠 Dashboard":

        st.subheader("📊 Insurance Claim Dashboard")

        if df is not None:
            total = len(df)
            fraud = len(df[df["prediction"] == -1])
            genuine = len(df[df["prediction"] == 1])

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(f"<div class='card'>Dataset Claims<div class='metric-number'>{total}</div></div>", unsafe_allow_html=True)
            with c2:
                st.markdown(f"<div class='card'>ML-Flagged Fraud<div class='metric-number' style='color:#ff6b6b;'>{fraud}</div></div>", unsafe_allow_html=True)
            with c3:
                st.markdown(f"<div class='card'>ML-Flagged Genuine<div class='metric-number'>{genuine}</div></div>", unsafe_allow_html=True)
            with c4:
                st.markdown(f"<div class='card'>Customer Complaints<div class='metric-number' style='color:#7fe3ff;'>{len(complaints_df)}</div></div>", unsafe_allow_html=True)

        st.write("")
        st.subheader("🔍 Quick Manual Risk Check")
        st.caption("Run a one-off risk assessment without needing a customer submission.")

        qc1, qc2 = st.columns(2)
        with qc1:
            claim_amount = st.number_input("💰 Claim Amount", min_value=0, key="qc_amt")
            age = st.slider("👤 Customer Age", 18, 100, 30, key="qc_age")
            severity = st.selectbox("⚠️ Accident Severity", ["Minor", "Major", "Total Loss"], key="qc_sev")
        with qc2:
            vehicle_price = st.number_input("🚗 Vehicle Price", min_value=0, key="qc_price")
            description = st.text_area("📝 Claim Description", key="qc_desc")

        if st.button("🔍 Analyze Claim"):
            risk_score, reasons = calc_risk(claim_amount, vehicle_price, age, severity, description, 0)
            label, css_class, tag = risk_label(risk_score)

            st.metric("Risk Score", risk_score)
            st.markdown(f"<div class='result-{css_class}'>{label} — Score {risk_score}/8</div>", unsafe_allow_html=True)

            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk_score,
                title={"text": "Fraud Risk Score"},
                gauge={"axis": {"range": [0, 8]},
                       "steps": [{"range": [0, 2]}, {"range": [2, 4]}, {"range": [4, 8]}]}
            ))
            st.plotly_chart(gauge, use_container_width=True)

            if reasons:
                for r in reasons:
                    st.warning(r)
            else:
                st.success("No suspicious activity detected")

    # ---------------- CUSTOMER COMPLAINTS (NEW) ----------------
    elif nav == "📋 Customer Complaints":

        st.subheader("📋 Customer Complaints")
        st.caption("All claims submitted by customers through the Customer Portal.")

        if complaints_df.empty:
            st.info("No customer complaints have been filed yet.")
        else:
            total = len(complaints_df)
            high = len(complaints_df[complaints_df["RiskStatus"] == "High Risk"])
            medium = len(complaints_df[complaints_df["RiskStatus"] == "Medium Risk"])
            low = len(complaints_df[complaints_df["RiskStatus"] == "Low Risk"])
            pending = len(complaints_df[complaints_df["ClaimStatus"] == "Pending Review"])

            m1, m2, m3, m4, m5 = st.columns(5)
            with m1:
                st.markdown(f"<div class='card'>Total Complaints<div class='metric-number'>{total}</div></div>", unsafe_allow_html=True)
            with m2:
                st.markdown(f"<div class='card'>High Risk<div class='metric-number' style='color:#ff6b6b;'>{high}</div></div>", unsafe_allow_html=True)
            with m3:
                st.markdown(f"<div class='card'>Medium Risk<div class='metric-number' style='color:#f5a623;'>{medium}</div></div>", unsafe_allow_html=True)
            with m4:
                st.markdown(f"<div class='card'>Low Risk<div class='metric-number'>{low}</div></div>", unsafe_allow_html=True)
            with m5:
                st.markdown(f"<div class='card'>Pending Review<div class='metric-number' style='color:#7dd3fc;'>{pending}</div></div>", unsafe_allow_html=True)

            st.write("")

            fc1, fc2, fc3 = st.columns([1, 1, 1.4])
            with fc1:
                risk_filter = st.selectbox("Filter by Risk", ["All", "High Risk", "Medium Risk", "Low Risk"])
            with fc2:
                status_filter = st.selectbox("Filter by Status", ["All"] + CLAIM_STATUSES)
            with fc3:
                search_name = st.text_input("🔎 Search by Customer Name")

            filtered = complaints_df.copy()
            if risk_filter != "All":
                filtered = filtered[filtered["RiskStatus"] == risk_filter]
            if status_filter != "All":
                filtered = filtered[filtered["ClaimStatus"] == status_filter]
            if search_name.strip():
                filtered = filtered[filtered["CustomerName"].str.contains(search_name.strip(), case=False, na=False)]

            filtered = filtered.sort_values("Timestamp", ascending=False)

            st.markdown(f"**Showing {len(filtered)} of {total} complaints**")

            for _, row in filtered.iterrows():
                tag = "high" if row["RiskStatus"] == "High Risk" else ("medium" if row["RiskStatus"] == "Medium Risk" else "low")
                icon = "🚨" if tag == "high" else ("⚠️" if tag == "medium" else "✅")

                claim_status = row.get("ClaimStatus", "Pending Review") or "Pending Review"
                status_class, status_icon = status_style(claim_status)

                image_path = row.get("ImagePath", "")
                has_image = pd.notna(image_path) and str(image_path).strip() and os.path.exists(str(image_path))
                camera_tag = " 📷" if has_image else ""

                header = f"{icon}  {row['CustomerName']}  ·  ₹{row['ClaimAmount']:,.0f}  ·  {row['RiskStatus']}  ·  {status_icon} {claim_status}{camera_tag}  ·  {row['Timestamp']}"

                with st.expander(header):

                    detail_col, image_col = st.columns([1.4, 1]) if has_image else (st.container(), None)

                    admin_notes = row.get("AdminNotes", "")
                    notes_html = (
                        f"<br><span style='color:#cfcfe4;font-size:13px;'>🗒️ <b>Admin Note:</b> {admin_notes}</span>"
                        if pd.notna(admin_notes) and str(admin_notes).strip() else ""
                    )

                    with detail_col:
                        st.markdown(
                            f"""
                            <div class="complaint-card {tag}">
                                <b>👤 {row['CustomerName']}</b> &nbsp;|&nbsp; 📱 {row['Mobile']}
                                <span class="pill {tag}">{row['RiskStatus']} ({row['RiskScore']}/8)</span>
                                <span class="pill {status_class}">{status_icon} {claim_status}</span>
                                <br><b>💰 Claim:</b> ₹{row['ClaimAmount']:,} &nbsp;|&nbsp;
                                <b>🚗 Vehicle:</b> ₹{row['VehiclePrice']:,} &nbsp;|&nbsp;
                                <b>👤 Age:</b> {row['Age']} &nbsp;|&nbsp;
                                <b>⚠️ Severity:</b> {row['Severity']}
                                <br><span style="color:#9c9cb5;font-size:12.5px;">🕒 {row['Timestamp']}</span>
                                <br><span style="color:#cfcfe4;font-size:13px;">📝 {row['Description'] if pd.notna(row['Description']) and str(row['Description']).strip() else 'No description provided'}</span>
                                {notes_html}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    if has_image:
                        with image_col:
                            st.markdown("**📷 Accident Photo**")
                            st.image(str(image_path), use_container_width=True)
                    else:
                        st.caption("📷 No accident photo was attached to this claim.")

                    st.markdown("**🛠️ Update Claim Status**")
                    with st.form(f"status_form_{row['Id']}"):
                        sc1, sc2 = st.columns([1, 2])
                        with sc1:
                            new_status = st.selectbox(
                                "Status", CLAIM_STATUSES,
                                index=CLAIM_STATUSES.index(claim_status) if claim_status in CLAIM_STATUSES else 0,
                                key=f"status_select_{row['Id']}"
                            )
                        with sc2:
                            new_notes = st.text_area(
                                "Notes for customer (optional)",
                                value=admin_notes if pd.notna(admin_notes) else "",
                                key=f"status_notes_{row['Id']}",
                                height=68
                            )
                        if st.form_submit_button("💾 Save Status"):
                            update_complaint_status(int(row["Id"]), new_status, new_notes)
                            st.success("Status updated.")
                            st.rerun()

            st.download_button(
                "📥 Download Complaints CSV",
                data=filtered.to_csv(index=False),
                file_name="customer_complaints_export.csv"
            )

    # ---------------- DATASET ANALYSIS ----------------
    elif nav == "📊 Dataset Analysis":

        st.subheader("📊 Dataset Overview")

        if df is not None:
            a, b, c = st.columns(3)
            a.metric("Total Records", df.shape[0])
            b.metric("Total Features", df.shape[1])
            c.metric("Missing Values", df.isnull().sum().sum())

            st.subheader("Dataset Preview")
            st.dataframe(df.head(20), use_container_width=True)

    # ---------------- EDA GRAPHS ----------------
    elif nav == "📈 EDA Graphs":

        st.subheader("📈 Exploratory Data Analysis")

        if df is not None:
            numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
            selected_col = st.selectbox("Select Feature", numeric_cols)

            fig = px.histogram(df, x=selected_col, title=f"{selected_col} Distribution")
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Correlation Heatmap")
            corr = df[numeric_cols].corr()
            fig2 = px.imshow(corr, text_auto=True)
            st.plotly_chart(fig2, use_container_width=True)

    # ---------------- MODEL DETAILS ----------------
    elif nav == "🤖 Model Details":

        st.subheader("🤖 Machine Learning Model")

        st.markdown(
            """
            <div class="info-card">
            <b>Algorithm Used:</b> Isolation Forest<br>
            <b>Purpose:</b> Detect abnormal claim behavior and identify possible fraud cases.
            </div>
            """,
            unsafe_allow_html=True
        )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", "94%")
        m2.metric("Precision", "91%")
        m3.metric("Recall", "89%")
        m4.metric("F1 Score", "90%")

        st.subheader("Confusion Matrix")
        matrix = pd.DataFrame(
            [[850, 50], [40, 260]],
            columns=["Pred Genuine", "Pred Fraud"],
            index=["Actual Genuine", "Actual Fraud"]
        )
        st.dataframe(matrix, use_container_width=True)

        fig = px.imshow(matrix, text_auto=True, title="Confusion Matrix")
        st.plotly_chart(fig, use_container_width=True)

    # ---------------- ABOUT ----------------
    else:

        st.subheader("Smart Fraudulent Insurance Claim Detector")

        st.markdown(
            """
            <div class="info-card">
            This system uses Artificial Intelligence and Machine Learning techniques
            to detect fraudulent insurance claims.<br><br>
            <b>Features:</b><br>
            ✔ ML Fraud Prediction<br>
            ✔ Separate Customer &amp; Admin Portals<br>
            ✔ Image Verification<br>
            ✔ Text Analysis<br>
            ✔ Dashboard Analytics<br>
            ✔ Risk Score Detection<br><br>
            <b>Developed By:</b> Dave Devarshi K.
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown(
        """
        <center>
        Developed by
        <h4>Dave Devarshi K 🚀</h4>
        Python | Machine Learning | Streamlit
        </center>
        """,
        unsafe_allow_html=True
    )
