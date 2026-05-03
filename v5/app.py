# app.py
import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from model5 import (
    HierarchicalAgent,
    MultiAgentWorld,
    ActionBases,
    StateDynamicsCoeffs,
    EscalationCoeffs,
)

# ==============================================
# Tooltip texts (دو خطی و خیلی ساده)
# ==============================================
TOOLTIPS = {
    # اجرای شبیه‌سازی
    "scenario": "یک سناریوی آماده یا حالت سفارشی را انتخاب کن.\nدر حالت سفارشی می‌توانی همه پارامترها را تغییر بدهی.",
    "doctrine_update_every": "هر چند بار تکرار یک اقدام، دکترین کمی تغییر کند.\n۰ یعنی دکترین ثابت می‌ماند.",
    "test_mode": "برای تست و مقایسه، اجرا را تکرارپذیر می‌کند.\nبا Seed یکسان همیشه خروجی یکسان می‌گیری.",
    "seed": "عدد ثابت برای تصادفی‌سازی.\nSeed یکسان → نتیجه یکسان.",
    "steps": "تعداد گام‌های زمانی شبیه‌سازی.\nعدد بزرگ‌تر یعنی دوره طولانی‌تر.",
    "num_runs": "تعداد دفعات تکرار شبیه‌سازی.\nبرای رفع خطای تصادفی، نتایجِ چند اجرا با هم میانگین گرفته می‌شوند.",

    # سفارشی
    "custom_n": "تعداد کشورهای سناریوی دستی را مشخص می‌کند.\nبین ۲ تا ۵ کشور قابل انتخاب است.",
    "country_name": "نام نمایشی کشور در نمودارها و جدول‌ها.\nهر نامی می‌خواهی وارد کن.",

    # اقتصاد/مرز
    "res0": "منابع اولیه کشور در شروع شبیه‌سازی.\nبزرگ‌تر یعنی دست بازتر برای اقدام‌ها.",
    "income": "درآمد ثابت در هر گام زمانی.\nبزرگ‌تر یعنی منابع سریع‌تر برمی‌گردد.",
    "v_c": "آسیب‌پذیری مرزی (فشار مرزها).\nبالاتر یعنی با تنش سریع‌تر تحت فشار می‌رود.",
    "chi_c": "شدت خرج منابع برای اقدام‌ها.\nبالاتر یعنی هر اقدام منابع بیشتری مصرف می‌کند.",

    # دکترین
    "rho_c": "ریسک‌پذیری کشور.\nبالاتر یعنی احتمال اقدام تند بیشتر.",
    "d_c": "گرایش به نفوذ نرم نسبت به زور.\nبالاتر یعنی بیشتر سیگنال/نفوذ تا تقویت نظامی.",
    "f_c": "آستانه رفتن به زور.\nپایین‌تر یعنی زودتر به اقدام سخت می‌رسد.",

    # راهبرد
    "wsec": "وزن امنیت در تصمیم‌گیری.\nبالاتر یعنی امنیت مهم‌تر است.",
    "winf": "وزن نفوذ در تصمیم‌گیری.\nبالاتر یعنی اثرگذاری مهم‌تر است.",
    "wcost": "وزن هزینه در تصمیم‌گیری.\nبالاتر یعنی به خرج منابع حساس‌تر است.",

    # عملیاتی
    "lambda_op": "توان عملیاتی برای اجرای تصمیم‌ها.\nبالاتر یعنی ظرفیت عمل بیشتر.",
    "tau_c": "تمپو/سرعت ریتم اقدام‌ها.\nبالاتر یعنی کشور سریع‌تر عمل می‌کند.",
    "eps_c": "حد شروع واکنش شدید.\nپایین‌تر یعنی زودتر وارد حالت بسیج می‌شود.",

    # فنی
    "eta_c": "توان فنی و کیفیت اجرا.\nبالاتر یعنی اقدام‌ها مؤثرتر اجرا می‌شوند.",
    "kappa_c": "هزینه یادگیری و بهبود توان.\nبالاتر یعنی بهبود سخت‌تر/گران‌تر است.",
    "p_alpha": "باور اولیه از موفقیت اقدام‌ها (α).\nبالاتر یعنی خوش‌بینی بیشتر.",
    "p_beta": "باور اولیه از شکست اقدام‌ها (β).\nبالاتر یعنی بدبینی بیشتر.",
    "r_alpha": "باور اولیه از پایداری/اعتماد (α).\nبالاتر یعنی قابل‌اعتمادتر.",
    "r_beta": "باور اولیه از خرابی/بی‌ثباتی (β).\nبالاتر یعنی خرابی محتمل‌تر.",

    # تاکتیکی
    "beta_c": "حساسیت به تفاوت منفعت.\nبالاتر یعنی تغییرات کوچک سریع اثر می‌گذارد.",
    "prefP": "گرایش ذاتی به آگاهی وضعیتی (P).\nبالاتر یعنی P بیشتر انتخاب می‌شود.",
    "prefS": "گرایش ذاتی به سیگنال (S).\nبالاتر یعنی S بیشتر انتخاب می‌شود.",
    "prefR": "گرایش ذاتی به تقویت/زور (R).\nبالاتر یعنی R بیشتر انتخاب می‌شود.",

    # ماتریس تعاملات
    "W": "W[i,j] یعنی کشور i چقدر کشور j را هدف می‌گیرد.\nعدد بزرگ‌تر یعنی تعامل/هدف‌گیری بیشتر.",

    # نمایش‌ها
    "t_selected": "گام زمانی برای نمایش گراف تعاملات.\nیک لحظه از روند را انتخاب می‌کنی.",
    "edgeP": "یال‌های اقدام P را در گراف نشان/پنهان می‌کند.\nروی مدل اثر ندارد، فقط نمایش است.",
    "edgeS": "یال‌های اقدام S را در گراف نشان/پنهان می‌کند.\nروی مدل اثر ندارد، فقط نمایش است.",
    "edgeR": "یال‌های اقدام R را در گراف نشان/پنهان می‌کند.\nروی مدل اثر ندارد، فقط نمایش است.",
    "country_filter": "فقط همین کشورها در جدول تغییرات بیاید.\nبرای تمرکز و کاهش شلوغی.",
    "section_filter": "فقط این بخش‌ها در جدول تغییرات بیاید.\nبرای خلاصه‌تر شدن نمایش.",
    "run_btn": "شبیه‌سازی را با تنظیمات فعلی اجرا می‌کند.\nنتیجه‌ها پایین صفحه نمایش داده می‌شود.",
    "reset_btn": "مقادیر سفارشی را به حالت پیش‌فرض برمی‌گرداند.\nبرای شروع دوباره از این دکمه استفاده کن.",

}

def tip(key: str) -> str:
    return TOOLTIPS.get(key, "")

# ==========================================================
# Scenario-change reset
# ==========================================================
def _reset_on_scenario_change(new_choice: str):
    prev = st.session_state.get("_prev_scenario_choice", None)
    if prev is None:
        st.session_state["_prev_scenario_choice"] = new_choice
        return

    if prev == new_choice:
        return

    # --- keys to keep (minimal) ---
    keep = {"_prev_scenario_choice", "scenario_choice"}

    # Remove everything else
    for k in list(st.session_state.keys()):
        if k not in keep:
            try:
                del st.session_state[k]
            except Exception:
                pass

    st.session_state["_prev_scenario_choice"] = new_choice
    st.rerun()

# ==========================================================
# 0) Helpers
# ==========================================================
ACTION_LABEL_FA = {"P": "آگاهی وضعیتی (P)", "S": "سیگنال (S)", "R": "تقویت/زور (R)"}
SECTION_ORDER = ["دکترین", "راهبرد", "تکنیک", "تاکتیک", "وضعیت"]

def set_seed_if_needed(test_mode: bool, seed: int | None):
    if test_mode and seed is not None:
        np.random.seed(int(seed))

def normalize_weights(x1: float, x2: float, x3: float):
    s = max(1e-12, x1 + x2 + x3)
    return [x1 / s, x2 / s, x3 / s]

# ==========================================================
# 1) Scenario pack
# ==========================================================
def scenario_pack():
    scenarios = {}

    def _W01_to_signed(W01):
        W = np.array(W01, dtype=float)
        W = 1.0 - 2.0 * W  # 0..1 → +1..-1
        np.fill_diagonal(W, 0.0)
        return W.tolist()

    scenarios["scenario_1"] = {
        "title": "سناریو ۱: بحران دریایی و رقابت بازدارندگی",
        "countries": ["آرمینیا", "نرمان", "جلال"],
        "story": """
**داستان سناریو:**
سه کشور فرضی در یک آبراه راهبردی رقابت دارند.  
- **آرمینیا** امنیت مسیرهای تجاری را مهم می‌داند.  
- **نرمان** بیشتر با **سیگنال (S)** نفوذ می‌سازد.  
- **جلال** در برخی دوره‌ها به سمت **تقویت/زور (R)** می‌رود ولی هزینه‌اش را می‌پردازد.

**انتظار خروجی:**
- در نقشه اقدامات، **S** به‌خصوص برای نرمان دیده می‌شود.
- منابع همیشه افزایش پیدا نمی‌کند (هزینه اقدام واقعی است).
- تعاملات جهت‌دار هستند (هر کشور هدف مشخص دارد).
""",
        "W": _W01_to_signed([
            [0.0, 0.70, 0.30],
            [0.55, 0.0, 0.45],
            [0.35, 0.65, 0.0],
        ]),
        "agents": [
            dict(
                name="آرمینیا",
                res0=1100.0, v=0.55, rho=0.45, d=0.55, f=0.55, chi=1.05,
                wsec=3.2, winf=1.7, wcost=2.1, lambda_op=0.55, tau=5.0, eps=0.58, income=14.0,
                eta=1.10, kappa=1.00, pa=2.4, pb=2.2, ra=2.2, rb=2.2,
                beta=2.1, prefP=1.2, prefS=1.0, prefR=0.8,
            ),
            dict(
                name="نرمان",
                res0=1250.0, v=0.62, rho=0.35, d=0.78, f=0.45, chi=1.00,
                wsec=2.3, winf=3.2, wcost=1.8, lambda_op=0.45, tau=5.8, eps=0.55, income=16.0,
                eta=1.15, kappa=0.95, pa=2.6, pb=2.0, ra=2.5, rb=2.1,
                beta=1.9, prefP=0.9, prefS=1.4, prefR=0.7,
            ),
            dict(
                name="جلال",
                res0=1150.0, v=0.70, rho=0.60, d=0.48, f=0.62, chi=1.15,
                wsec=3.4, winf=1.4, wcost=2.2, lambda_op=0.60, tau=4.7, eps=0.60, income=13.0,
                eta=1.05, kappa=1.05, pa=2.2, pb=2.4, ra=2.0, rb=2.4,
                beta=2.2, prefP=1.0, prefS=0.8, prefR=1.2,
            ),
        ],
        "steps_default": 70,
    }

    scenarios["scenario_2"] = {
        "title": "سناریو ۲: اتحاد دفاعی در برابر تهدید مشترک",
        "countries": ["الفا", "بتا", "گاما"],
        "story": """
**داستان سناریو:**
الفا و بتا متحدند و بیشتر P/S انجام می‌دهند تا گاما را مهار کنند.  
گاما گاهی R می‌کند و منابعش سریع‌تر کم می‌شود.
""",
        "W": _W01_to_signed([
            [0.0, 0.20, 0.80],
            [0.20, 0.0, 0.80],
            [0.60, 0.40, 0.0],
        ]),
        "agents": [
            dict(
                name="الفا", res0=1300, v=0.52, rho=0.40, d=0.62, f=0.50, chi=1.00,
                wsec=3.0, winf=2.0, wcost=2.0, lambda_op=0.55, tau=5.4, eps=0.56, income=17.0,
                eta=1.15, kappa=0.95, pa=2.5, pb=2.2, ra=2.3, rb=2.3,
                beta=2.0, prefP=1.2, prefS=1.1, prefR=0.7
            ),
            dict(
                name="بتا", res0=1200, v=0.58, rho=0.35, d=0.74, f=0.45, chi=1.05,
                wsec=2.5, winf=3.0, wcost=1.8, lambda_op=0.50, tau=5.9, eps=0.54, income=16.0,
                eta=1.10, kappa=1.00, pa=2.6, pb=2.0, ra=2.4, rb=2.2,
                beta=1.9, prefP=0.9, prefS=1.4, prefR=0.7
            ),
            dict(
                name="گاما", res0=1400, v=0.76, rho=0.62, d=0.45, f=0.70, chi=1.20,
                wsec=3.6, winf=1.2, wcost=2.4, lambda_op=0.60, tau=4.6, eps=0.61, income=14.0,
                eta=1.00, kappa=1.10, pa=2.1, pb=2.6, ra=2.0, rb=2.6,
                beta=2.3, prefP=0.9, prefS=0.7, prefR=1.4
            ),
        ],
        "steps_default": 70,
    }

    scenarios["scenario_3"] = {
        "title": "سناریو ۳: رقابت نیابتی و جنگ روانی",
        "countries": ["دلتا", "اپسیلون", "زتا"],
        "story": """
**داستان سناریو:**
کشورها بیشتر با **سیگنال (S)** رقابت می‌کنند و R کمتر رخ می‌دهد مگر تنش بالا برود.
""",
        "W": _W01_to_signed([
            [0.0, 0.55, 0.45],
            [0.40, 0.0, 0.60],
            [0.60, 0.40, 0.0],
        ]),
        "agents": [
            dict(
                name="دلتا", res0=1150, v=0.60, rho=0.40, d=0.82, f=0.40, chi=0.95,
                wsec=2.2, winf=3.4, wcost=1.6, lambda_op=0.42, tau=6.2, eps=0.52, income=15.0,
                eta=1.20, kappa=0.90, pa=2.6, pb=2.0, ra=2.6, rb=2.0,
                beta=1.8, prefP=0.7, prefS=1.6, prefR=0.6
            ),
            dict(
                name="اپسیلون", res0=1250, v=0.64, rho=0.38, d=0.75, f=0.45, chi=1.00,
                wsec=2.5, winf=3.0, wcost=1.8, lambda_op=0.45, tau=6.0, eps=0.53, income=16.0,
                eta=1.15, kappa=0.95, pa=2.5, pb=2.1, ra=2.4, rb=2.2,
                beta=1.9, prefP=0.8, prefS=1.4, prefR=0.7
            ),
            dict(
                name="زتا", res0=1100, v=0.70, rho=0.50, d=0.62, f=0.55, chi=1.10,
                wsec=2.8, winf=2.2, wcost=2.0, lambda_op=0.50, tau=5.5, eps=0.57, income=14.0,
                eta=1.10, kappa=1.00, pa=2.2, pb=2.4, ra=2.2, rb=2.4,
                beta=2.0, prefP=0.9, prefS=1.1, prefR=1.0
            ),
        ],
        "steps_default": 70,
    }

    scenarios["scenario_4"] = {
        "title": "سناریو ۴: تنش مرزی بین دو کشور (دو بازیگر)",
        "countries": ["آتا", "بتا"],
        "story": """
**داستان سناریو:**
دو کشور هم‌مرز هستند.
آتا بیشتر با S بازدارندگی می‌سازد. بتا گاهی R می‌کند و منابعش سریع‌تر کم می‌شود.
""",
        "W": _W01_to_signed([
            [0.0, 1.0],
            [1.0, 0.0],
        ]),
        "agents": [
            dict(
                name="آتا", res0=1200, v=0.68, rho=0.38, d=0.80, f=0.45, chi=1.00,
                wsec=2.6, winf=3.2, wcost=1.8, lambda_op=0.45, tau=5.9, eps=0.54, income=16.0,
                eta=1.15, kappa=0.95, pa=2.6, pb=2.0, ra=2.5, rb=2.1,
                beta=1.9, prefP=0.8, prefS=1.5, prefR=0.7
            ),
            dict(
                name="بتا", res0=1350, v=0.76, rho=0.62, d=0.48, f=0.70, chi=1.20,
                wsec=3.5, winf=1.3, wcost=2.4, lambda_op=0.60, tau=4.6, eps=0.61, income=14.0,
                eta=1.00, kappa=1.10, pa=2.1, pb=2.6, ra=2.0, rb=2.6,
                beta=2.3, prefP=0.8, prefS=0.6, prefR=1.5
            ),
        ],
        "steps_default": 60,
    }

    scenarios["scenario_5"] = {
        "title": "سناریو ۵: بحران چندقطبی در تنگه تجاری (۵ کشور)",
        "countries": ["اوران", "سَحَر", "کایان", "مِهران", "وِستا"],
        "story": """
    **داستان سناریو (۵ کشور):**
    یک تنگه‌ی تجاری حیاتی وجود دارد که عبور انرژی و کالا از آن انجام می‌شود. پنج کشور فرضی درگیر رقابت و بازدارندگی‌اند.
    """,
        "W": _W01_to_signed([
            [0.0, 0.15, 0.45, 0.25, 0.15],
            [0.20, 0.0, 0.35, 0.10, 0.35],
            [0.35, 0.10, 0.0, 0.10, 0.45],
            [0.30, 0.15, 0.25, 0.0, 0.30],
            [0.25, 0.10, 0.45, 0.20, 0.0],
        ]),
        "agents": [
            dict(name="اوران", res0=1450.0, v=0.58, rho=0.38, d=0.55, f=0.58, chi=1.05, wsec=3.4, winf=1.8, wcost=2.2, lambda_op=0.60, tau=5.1, eps=0.58, income=17.0, eta=1.15, kappa=0.95, pa=2.6, pb=2.2, ra=2.4, rb=2.3, beta=2.1, prefP=1.35, prefS=0.95, prefR=0.70),
            dict(name="سَحَر", res0=1350.0, v=0.62, rho=0.34, d=0.86, f=0.48, chi=0.95, wsec=2.2, winf=3.6, wcost=1.6, lambda_op=0.45, tau=6.0, eps=0.54, income=16.0, eta=1.20, kappa=0.90, pa=2.7, pb=2.0, ra=2.6, rb=2.1, beta=1.9, prefP=0.80, prefS=1.55, prefR=0.65),
            dict(name="کایان", res0=1500.0, v=0.74, rho=0.62, d=0.42, f=0.62, chi=1.25, wsec=3.6, winf=1.3, wcost=2.6, lambda_op=0.70, tau=4.6, eps=0.62, income=15.0, eta=1.05, kappa=1.10, pa=2.2, pb=2.6, ra=2.0, rb=2.7, beta=2.3, prefP=0.85, prefS=0.70, prefR=1.45),
            dict(name="مِهران", res0=1400.0, v=0.56, rho=0.30, d=0.70, f=0.64, chi=1.00, wsec=2.8, winf=2.7, wcost=1.9, lambda_op=0.52, tau=5.8, eps=0.56, income=16.0, eta=1.18, kappa=0.95, pa=2.6, pb=2.1, ra=2.5, rb=2.2, beta=2.0, prefP=1.10, prefS=1.20, prefR=0.65),
            dict(name="وِستا", res0=1650.0, v=0.72, rho=0.50, d=0.55, f=0.55, chi=1.10, wsec=3.2, winf=1.9, wcost=2.3, lambda_op=0.62, tau=5.0, eps=0.60, income=18.0, eta=1.10, kappa=1.00, pa=2.4, pb=2.3, ra=2.3, rb=2.4, beta=2.2, prefP=1.05, prefS=0.90, prefR=1.10),
        ],
        "steps_default": 85,
    }

    scenarios["scenario_6"] = {
        "title": "سناریو ۶: رقابت قدرت‌های بزرگ",
        "countries": ["ایران", "اسرائیل", "آمریکا", "چین", "روسیه"],
        "story": """**داستان سناریو:** پنج بازیگر مهم با مجموعه‌ای از تقابل‌ها و همسویی‌ها همزمان در یک محیط پرتنش حضور دارند.""",
        "W": [
            [0.0, -0.90, -0.80, 0.45, 0.30],
            [-0.85, 0.0, 0.80, -0.20, -0.30],
            [-0.75, 0.85, 0.0, -0.70, -0.80],
            [0.35, -0.15, -0.65, 0.0, 0.60],
            [0.25, -0.25, -0.75, 0.55, 0.0],
        ],
        "agents": [
            dict(name="ایران", res0=1250.0, v=0.72, rho=0.58, d=0.55, f=0.58, chi=1.10, wsec=3.4, winf=2.0, wcost=2.2, lambda_op=0.58, tau=5.0, eps=0.60, income=15.0, eta=1.05, kappa=1.05, pa=2.3, pb=2.4, ra=2.2, rb=2.4, beta=2.2, prefP=1.0, prefS=1.0, prefR=1.2),
            dict(name="اسرائیل", res0=1150.0, v=0.60, rho=0.42, d=0.60, f=0.52, chi=1.05, wsec=3.8, winf=2.0, wcost=2.0, lambda_op=0.62, tau=5.2, eps=0.56, income=16.0, eta=1.18, kappa=0.95, pa=2.7, pb=2.0, ra=2.6, rb=2.1, beta=2.0, prefP=1.1, prefS=1.0, prefR=1.0),
            dict(name="آمریکا", res0=1800.0, v=0.50, rho=0.40, d=0.55, f=0.55, chi=1.00, wsec=3.6, winf=2.1, wcost=2.4, lambda_op=0.70, tau=5.4, eps=0.58, income=20.0, eta=1.25, kappa=0.90, pa=2.8, pb=1.9, ra=2.7, rb=2.0, beta=2.0, prefP=1.1, prefS=0.95, prefR=0.95),
            dict(name="چین", res0=1750.0, v=0.55, rho=0.38, d=0.78, f=0.50, chi=1.00, wsec=2.8, winf=3.2, wcost=2.0, lambda_op=0.62, tau=5.8, eps=0.56, income=19.0, eta=1.20, kappa=0.95, pa=2.7, pb=2.0, ra=2.6, rb=2.1, beta=1.9, prefP=0.95, prefS=1.25, prefR=0.80),
            dict(name="روسیه", res0=1550.0, v=0.62, rho=0.55, d=0.55, f=0.60, chi=1.15, wsec=3.3, winf=1.8, wcost=2.2, lambda_op=0.65, tau=4.9, eps=0.60, income=17.0, eta=1.10, kappa=1.05, pa=2.4, pb=2.3, ra=2.3, rb=2.4, beta=2.1, prefP=0.95, prefS=0.95, prefR=1.15),
        ],
        "steps_default": 90,
    }
    return scenarios

# ==========================================================
# 2) Custom UI
# ==========================================================
def build_custom_ui(prefill_agents=None, prefill_W=None, prefill_countries=None, lock_n: bool = False):
    st.subheader("🧰 تنظیمات سفارشی کشورها و تعاملات")

    if prefill_countries is not None:
        st.session_state.custom_country_names = list(prefill_countries)
    if prefill_agents is not None:
        st.session_state.custom_agents = [dict(a) for a in prefill_agents]
    if prefill_W is not None:
        W0 = np.array(prefill_W, dtype=float)
        np.fill_diagonal(W0, 0.0)
        st.session_state.custom_W = W0

    n_default = 3 if prefill_countries is None else len(prefill_countries)
    n_min = n_default if lock_n else 2
    n_max = n_default if lock_n else 5
    if lock_n:
        n = int(n_default)
        st.number_input("تعداد کشورها", min_value=n, max_value=n, value=n, step=1, key="custom_n_locked", disabled=True)
    else:
        n = st.slider("تعداد کشورها", min_value=int(n_min), max_value=int(n_max), value=int(n_default), step=1, key="custom_n")
    
    if "custom_country_names" not in st.session_state:
        st.session_state.custom_country_names = [f"کشور {chr(65 + i)}" for i in range(n)]
    else:
        cur = st.session_state.custom_country_names
        if len(cur) < n:
            cur = cur + [f"کشور {chr(65 + i)}" for i in range(len(cur), n)]
        elif len(cur) > n:
            cur = cur[:n]
        st.session_state.custom_country_names = cur

    cols = st.columns(n)
    for i in range(n):
        with cols[i]:
            st.session_state.custom_country_names[i] = st.text_input(f"نام کشور {i + 1}", value=st.session_state.custom_country_names[i], key=f"custom_name_{i}")
    countries = st.session_state.custom_country_names

    st.divider()

    def default_agent_cfg(name):
        return dict(
            name=name, res0=1200.0, v=0.60, rho=0.45, d=0.65, f=0.55, chi=1.05,
            wsec=3.0, winf=2.0, wcost=2.0, lambda_op=0.55, tau=5.5, eps=0.56, income=15.0,
            eta=1.10, kappa=1.00, pa=2.4, pb=2.3, ra=2.4, rb=2.4,
            beta=2.0, prefP=1.0, prefS=1.1, prefR=0.9,
        )

    if "custom_agents" not in st.session_state:
        st.session_state.custom_agents = [default_agent_cfg(c) for c in countries]
    else:
        agents = st.session_state.custom_agents
        if len(agents) < n:
            agents += [default_agent_cfg(countries[i]) for i in range(len(agents), n)]
        elif len(agents) > n:
            agents = agents[:n]
        for i in range(n):
            agents[i]["name"] = countries[i]
        st.session_state.custom_agents = agents

    tabs = st.tabs([f"کشور: {c}" for c in countries])
    for i, tab in enumerate(tabs):
        cfg = st.session_state.custom_agents[i]
        with tab:
            st.markdown("### 1) وضعیت و اقتصاد")
            c1, c2, c3, c4 = st.columns(4)
            cfg["res0"] = c1.number_input("منابع اولیه (res0)", 0.0, 10000.0, float(cfg["res0"]), 50.0, key=f"res0_{i}")
            cfg["income"] = c2.number_input("درآمد هر گام (μ_c)", 0.0, 200.0, float(cfg["income"]), 1.0, key=f"income_{i}")
            cfg["v"] = c3.slider("آسیب‌پذیری مرزی (v_c)", 0.0, 1.0, float(cfg["v"]), 0.01, key=f"v_{i}")
            cfg["chi"] = c4.slider("ضریب هزینه منابع (χ_c)", 0.1, 3.0, float(cfg["chi"]), 0.01, key=f"chi_{i}")

            st.markdown("### 2) دکترین (Doctrine)")
            d1, d2, d3 = st.columns(3)
            cfg["rho"] = d1.slider("ریسک‌پذیری (ρ_c)", 0.0, 1.0, float(cfg["rho"]), 0.01, key=f"rho_{i}")
            cfg["d"] = d2.slider("ترجیح بازدارندگی/نفوذ (d_c)", 0.0, 1.0, float(cfg["d"]), 0.01, key=f"d_{i}")
            cfg["f"] = d3.slider("آستانه زور (f_c)", 0.0, 1.0, float(cfg["f"]), 0.01, key=f"f_{i}")

            st.markdown("### 3) وزن‌های راهبردی (ω_S)")
            w1, w2, w3 = st.columns(3)
            cfg["wsec"] = w1.number_input("اهمیت امنیت (ω_sec)", 0.0, 10.0, float(cfg["wsec"]), 0.1, key=f"wsec_{i}")
            cfg["winf"] = w2.number_input("اهمیت نفوذ (ω_inf)", 0.0, 10.0, float(cfg["winf"]), 0.1, key=f"winf_{i}")
            cfg["wcost"] = w3.number_input("اهمیت هزینه (ω_cost)", 0.0, 10.0, float(cfg["wcost"]), 0.1, key=f"wcost_{i}")

            st.markdown("### 4) عملیاتی (Operational)")
            o1, o2, o3 = st.columns(3)
            cfg["lambda_op"] = o1.slider("توان تخصیص عملیات (λ_op)", 0.0, 1.0, float(cfg["lambda_op"]), 0.01, key=f"lambdaop_{i}")
            cfg["tau"] = o2.slider("سرعت/Tempo (τ_c)", 1.0, 15.0, float(cfg["tau"]), 0.1, key=f"tau_{i}")
            cfg["eps"] = o3.slider("حد شروع واکنش شدید (ε_c)", 0.0, 1.0, float(cfg["eps"]), 0.01, key=f"eps_{i}")

            st.markdown("### 5) تاکتیک (Tactical)")
            b1, _ = st.columns(2)
            cfg["beta"] = b1.slider("حساسیت به سود (β_c)", 0.1, 10.0, float(cfg["beta"]), 0.1, key=f"beta_{i}")
            p1, p2, p3 = st.columns(3)
            cfg["prefP"] = p1.number_input("ترجیح آگاهی وضعیتی (ω_P)", 0.1, 10.0, float(cfg["prefP"]), 0.1, key=f"prefP_{i}")
            cfg["prefS"] = p2.number_input("ترجیح سیگنال (ω_S)", 0.1, 10.0, float(cfg["prefS"]), 0.1, key=f"prefS_{i}")
            cfg["prefR"] = p3.number_input("ترجیح تقویت/زور (ω_R)", 0.1, 10.0, float(cfg["prefR"]), 0.1, key=f"prefR_{i}")

            st.markdown("### 6) تکنیک (Technical)")
            t1, t2, t3, t4 = st.columns(4)
            cfg["eta"] = t1.slider("توان فنی (η_c)", 0.2, 3.0, float(cfg["eta"]), 0.01, key=f"eta_{i}")
            cfg["kappa"] = t2.slider("هزینه یادگیری (κ_c)", 0.2, 3.0, float(cfg["kappa"]), 0.01, key=f"kappa_{i}")
            cfg["pa"] = t3.number_input("موفقیت فرضی (p:α)", 1.0, 10.0, float(cfg["pa"]), 0.1, key=f"pa_{i}")
            cfg["pb"] = t4.number_input("شکست فرضی (p:β)", 1.0, 10.0, float(cfg["pb"]), 0.1, key=f"pb_{i}")
            r1, r2 = st.columns(2)
            cfg["ra"] = r1.number_input("پایداری/اطمینان (r:α)", 1.0, 10.0, float(cfg["ra"]), 0.1, key=f"ra_{i}")
            cfg["rb"] = r2.number_input("خرابی/بی‌ثباتی (r:β)", 1.0, 10.0, float(cfg["rb"]), 0.1, key=f"rb_{i}")

        st.session_state.custom_agents[i] = cfg

    st.divider()
    st.markdown("### 🔁 ماتریس تعاملات کشورها (W)")
    st.caption("W[i,j] رابطه‌ی کشور i با کشور j است. بازه بین -۱ تا +۱")

    if "custom_W" not in st.session_state:
        W0 = np.zeros((n, n), dtype=float)
        np.fill_diagonal(W0, 0.0)
        st.session_state.custom_W = W0

    W = st.session_state.custom_W
    if W.shape != (n, n):
        W2 = np.zeros((n, n), dtype=float)
        np.fill_diagonal(W2, 0.0)
        st.session_state.custom_W = W2
        W = W2

    W_df = pd.DataFrame(W, index=countries, columns=countries)
    edited = st.data_editor(W_df, use_container_width=True, key="W_editor")
    edited = edited.apply(pd.to_numeric, errors="coerce").fillna(0.0).clip(lower=-1.0, upper=1.0)
    
    # FIX: Explicitly create a copy of the array so it's not read-only
    W_array = edited.to_numpy(dtype=float, copy=True)
    np.fill_diagonal(W_array, 0.0)
    st.session_state.custom_W = W_array

    if st.button("↩️ بازنشانی مقادیر", use_container_width=True):
        st.session_state.custom_agents = [default_agent_cfg(c) for c in countries]
        W2 = np.zeros((n, n), dtype=float)
        np.fill_diagonal(W2, 0.0)
        st.session_state.custom_W = W2
        st.rerun()

    return st.session_state.custom_agents, st.session_state.custom_W.tolist(), countries

# ==========================================================
# 3) Build agents + run simulation
# ==========================================================
def build_agents_from_configs(agent_cfgs):
    action_bases = ActionBases()
    dyn = StateDynamicsCoeffs()

    agents = []
    for c in agent_cfgs:
        omega_S = normalize_weights(c["wsec"], c["winf"], c["wcost"])
        omega_a = normalize_weights(c["prefP"], c["prefS"], c["prefR"])
        agents.append(
            HierarchicalAgent(
                name=c["name"], initial_resource=float(c["res0"]), v_c=float(c["v"]),
                rho_c=float(c["rho"]), d_c=float(c["d"]), f_c=float(c["f"]), chi_c=float(c["chi"]),
                omega_S=np.array(omega_S, dtype=float), lambda_op=float(c["lambda_op"]),
                tau_c=float(c["tau"]), eps_c=float(c["eps"]), income_c=float(c["income"]),
                eta_c=float(c["eta"]), p_params=[float(c["pa"]), float(c["pb"])],
                r_params=[float(c["ra"]), float(c["rb"])], kappa_c=float(c["kappa"]),
                beta_c=float(c["beta"]), omega_a=np.array(omega_a, dtype=float),
                action_bases=action_bases, dyn_coeffs=dyn,
            )
        )
    return agents

def run_simulation(agent_cfgs, W, steps, test_mode, seed, doctrine_update_every: int):
    set_seed_if_needed(test_mode, seed)
    agents = build_agents_from_configs(agent_cfgs)
    meta = {
        "initial": {ag.name: ag.snapshot() for ag in agents},
        "final": {},
        "doctrine_update_every": int(doctrine_update_every),
    }

    world = MultiAgentWorld(
        agents=agents, interaction_W=W, esc_coeffs=EscalationCoeffs(),
        doctrine_update_every=int(doctrine_update_every),
    )
    for t in range(int(steps)):
        world.step(t)

    meta["final"] = {ag.name: ag.snapshot() for ag in agents}
    df = pd.DataFrame(world.history)
    return df, meta

def run_multiple_simulations(agent_cfgs, W, steps, test_mode, seed, doctrine_update_every, num_runs):
    dfs = []
    metas = []
    for i in range(num_runs):
        run_seed = seed + i if (test_mode and seed is not None) else None
        df, meta = run_simulation(agent_cfgs, W, steps, test_mode, run_seed, doctrine_update_every)
        dfs.append(df)
        metas.append(meta)

    if num_runs == 1:
        return dfs[0], metas[0], dfs

    # Average DFs
    df_concat = pd.concat(dfs)
    numeric_cols = df_concat.select_dtypes(include=[np.number]).columns.tolist()
    if 'Time' in numeric_cols:
        numeric_cols.remove('Time')

    df_num = df_concat.groupby('Time')[numeric_cols].mean().reset_index()

    cat_cols = df_concat.select_dtypes(exclude=[np.number]).columns.tolist()
    if 'Time' in cat_cols:
        cat_cols.remove('Time')

    if cat_cols:
        # Get Mode for Categorical
        df_cat = df_concat.groupby('Time')[cat_cols].agg(
            lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0]
        ).reset_index()
        df_avg = pd.merge(df_num, df_cat, on='Time')
    else:
        df_avg = df_num

    # Average Meta
    avg_meta = {
        "initial": {},
        "final": {},
        "doctrine_update_every": doctrine_update_every,
    }
    for state in ["initial", "final"]:
        for c in metas[0][state].keys():
            avg_meta[state][c] = {}
            for k, v in metas[0][state][c].items():
                if isinstance(v, (int, float)):
                    avg_meta[state][c][k] = float(np.mean([m[state][c][k] for m in metas]))
                elif isinstance(v, np.ndarray):
                    avg_meta[state][c][k] = np.mean([m[state][c][k] for m in metas], axis=0)
                else:
                    avg_meta[state][c][k] = v

    return df_avg, avg_meta, dfs

# ==========================================================
# 4) Tables + charts
# ==========================================================
def df_action_counts(dfs, countries):
    if isinstance(dfs, pd.DataFrame):
        dfs = [dfs]
    out = []
    n_runs = len(dfs)
    for c in countries:
        col = f"Action_{c}"
        counts = {"P": 0, "S": 0, "R": 0}
        for df in dfs:
            if col in df.columns:
                vc = df[col].value_counts().to_dict()
                for k in counts:
                    counts[k] += vc.get(k, 0)
        out.append({
            "کشور": c,
            "آگاهی وضعیتی (P)": round(counts["P"] / n_runs, 1),
            "سیگنال (S)": round(counts["S"] / n_runs, 1),
            "تقویت/زور (R)": round(counts["R"] / n_runs, 1),
        })
    return pd.DataFrame(out)

def _resource_norm(x: float) -> float:
    x = float(x)
    return x / (x + 1000.0)

def plot_three_indices_heatmaps(df: pd.DataFrame, countries: list[str], window: int = 10):
    if df is None or len(df) == 0 or len(countries) == 0: return
    if "Time" not in df.columns: return

    dfx = df.copy()
    dfx["Time"] = pd.to_numeric(dfx["Time"], errors="coerce").fillna(0).astype(int)
    dfx = dfx.sort_values("Time").drop_duplicates(subset=["Time"]).set_index("Time")
    times = dfx.index.astype(int).tolist()
    if not times: return

    valid, sec_z, res_z, inf_z = [], [], [], []

    for c in countries:
        tcol, rcol, pcol, acol = f"Tension_{c}", f"Resource_{c}", f"Psi_{c}", f"Action_{c}"
        if not all(col in dfx.columns for col in [tcol, rcol, pcol, acol]): continue

        tension, resource, psi, action = dfx[tcol].astype(float), dfx[rcol].astype(float), dfx[pcol].astype(float), dfx[acol]
        rnorm = resource.apply(_resource_norm)
        trend = resource - resource.shift(5)
        trend_norm = 1.0 / (1.0 + np.exp(-(trend.fillna(0.0) / 100.0)))
        is_S = (action == "S").astype(int)
        sshare = is_S.rolling(window=window, min_periods=1).mean()

        security = (0.45 * (1.0 - tension) + 0.35 * rnorm + 0.20 * (1.0 - psi)).clip(0, 1)
        resilience = (0.55 * rnorm + 0.25 * (1.0 - psi) + 0.20 * trend_norm).clip(0, 1)
        influence = (0.60 * sshare + 0.25 * (1.0 - tension) + 0.15 * (1.0 - psi)).clip(0, 1)

        valid.append(c)
        sec_z.append(security.reindex(times).fillna(0.0).tolist())
        res_z.append(resilience.reindex(times).fillna(0.0).tolist())
        inf_z.append(influence.reindex(times).fillna(0.0).tolist())

    if not valid: return
    st.subheader("۳ شاخص کلیدی در طول زمان (نقشه حرارتی)")
    tabs = st.tabs(["قدرت امنیت", "تاب‌آوری", "نفوذ/بازدارندگی"])

    def _heat(z, title):
        fig = go.Figure(data=go.Heatmap(z=z, x=times, y=valid, zmin=0, zmax=1, colorscale="RdYlGn", hovertemplate="%{y}<br>زمان: %{x}<br>امتیاز: %{z:.3f}<extra></extra>"))
        fig.update_layout(title=title, xaxis_title="گام زمانی", yaxis_title="کشور", margin=dict(l=30, r=30, t=60, b=30))
        st.plotly_chart(fig, use_container_width=True)

    with tabs[0]: _heat(sec_z, "قدرت امنیت")
    with tabs[1]: _heat(res_z, "تاب‌آوری")
    with tabs[2]: _heat(inf_z, "نفوذ/بازدارندگی")

def build_transition_df(meta, countries):
    if not meta or "initial" not in meta or "final" not in meta: return pd.DataFrame()
    rows = []
    for c in countries:
        ini, fin = meta["initial"].get(c, {}), meta["final"].get(c, {})
        if not ini or not fin: continue

        for k, label in [("rho_c", "ریسک‌پذیری (ρ_c)"), ("d_c", "ترجیح بازدارندگی/نفوذ (d_c)"), ("f_c", "آستانه زور (f_c)"), ("chi_c", "ضریب هزینه منابع (χ_c)")]:
            rows.append({"کشور": c, "بخش": "دکترین", "پارامتر": label, "ابتدا": float(ini.get(k, np.nan)), "انتها": float(fin.get(k, np.nan))})
        for idx, label in enumerate(["ω_sec", "ω_inf", "ω_cost"]):
            rows.append({"کشور": c, "بخش": "راهبرد", "پارامتر": f"وزن ({label})", "ابتدا": float(ini.get("omega_S", [np.nan]*3)[idx]), "انتها": float(fin.get("omega_S", [np.nan]*3)[idx])})
        for k, label in [("p_c", "موفقیت (p_c)"), ("r_c", "قابلیت اطمینان (r_c)"), ("eta_c", "توان فنی (η_c)"), ("kappa_c", "هزینه یادگیری (κ_c)")]:
            rows.append({"کشور": c, "بخش": "تکنیک", "پارامتر": label, "ابتدا": float(ini.get(k, np.nan)), "انتها": float(fin.get(k, np.nan))})
        rows.append({"کشور": c, "بخش": "تاکتیک", "پارامتر": "حساسیت به سود (β_c)", "ابتدا": float(ini.get("beta_c", np.nan)), "انتها": float(fin.get("beta_c", np.nan))})
        for idx, label in enumerate(["ω_P", "ω_S", "ω_R"]):
            rows.append({"کشور": c, "بخش": "تاکتیک", "پارامتر": f"ترجیح ({label})", "ابتدا": float(ini.get("omega_a", [np.nan]*3)[idx]), "انتها": float(fin.get("omega_a", [np.nan]*3)[idx])})
        for k, label in [("tension", "تنش"), ("resource", "منابع")]:
            rows.append({"کشور": c, "بخش": "وضعیت", "پارامتر": label, "ابتدا": float(ini.get(k, np.nan)), "انتها": float(fin.get(k, np.nan))})
    return pd.DataFrame(rows)

def plot_global_escalation(df):
    if "Global_Escalation" not in df.columns: return
    fig = px.area(df, x="Time", y="Global_Escalation", labels={"Time": "گام زمانی", "Global_Escalation": "احتمال/وضعیت بحران کلی"}, title="وضعیت بحران کلی", color_discrete_sequence=["red"])
    fig.update_yaxes(range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)

def plot_lines_by_country(df, countries, prefix, title_fa, y_label_fa):
    cols = [f"{prefix}_{c}" for c in countries if f"{prefix}_{c}" in df.columns]
    if not cols: return
    dfl = df.melt(id_vars=["Time"], value_vars=cols, var_name="variable", value_name="value")
    dfl["کشور"] = dfl["variable"].str.replace(f"{prefix}_", "", regex=False)
    fig = px.line(dfl, x="Time", y="value", color="کشور", title=title_fa, labels={"Time": "گام زمانی", "value": y_label_fa, "کشور": "کشور"})
    st.plotly_chart(fig, use_container_width=True)

def plot_actions_map(df, countries):
    act_cols = [f"Action_{c}" for c in countries if f"Action_{c}" in df.columns]
    if not act_cols: return
    df_m = df.melt(id_vars=["Time"], value_vars=act_cols, var_name="Country", value_name="Action")
    df_m["Country"] = df_m["Country"].str.replace("Action_", "", regex=False)
    uniq = df_m["Country"].unique().tolist()
    to_y = {c: i for i, c in enumerate(uniq)}
    df_m["y_base"] = df_m["Country"].map(to_y)
    offset_map = {"P": 0.00, "S": 0.12, "R": 0.24}
    df_m["y"] = df_m["y_base"] + df_m["Action"].map(offset_map).fillna(0.0)
    df_m["اقدام"] = df_m["Action"].map(ACTION_LABEL_FA).fillna(df_m["Action"])
    fig = px.scatter(df_m, x="Time", y="y", color="اقدام", symbol="اقدام", title="نقشه اقدامات (حالت پرتکرار)", labels={"Time": "گام زمانی", "y": "کشور"})
    fig.update_yaxes(tickmode="array", tickvals=[to_y[c] + 0.12 for c in uniq], ticktext=uniq, title="کشور")
    st.plotly_chart(fig, use_container_width=True)

def plot_dyad_tension_heatmap(df: pd.DataFrame, countries: list[str]):
    if df is None or len(df) == 0 or len(countries) < 2 or "Time" not in df.columns: return
    df = df.copy()
    df["Time"] = pd.to_numeric(df["Time"], errors="coerce").fillna(0).astype(int)
    df = df.sort_values("Time")
    times = df["Time"].unique().tolist()
    st.subheader("ماتریس حرارتی تنش بین کشورها در طول زمان")

    pairs = [(src, dst) for src in countries for dst in countries if src != dst]
    y_labels = [f"{src} - {dst}" for src, dst in pairs]
    df_idx = df.set_index("Time")
    z = []
    for src, dst in pairs:
        col = f"DyadTension_{src}_{dst}"
        series = pd.to_numeric(df_idx[col], errors="coerce") if col in df_idx.columns else pd.Series([], dtype=float)
        z.append(series.reindex(times).fillna(0.0).astype(float).tolist())

    fig = go.Figure(data=go.Heatmap(z=z, x=times, y=y_labels, zmin=0, zmax=1, colorscale="RdYlGn_r", hovertemplate="زمان: %{x}<br>%{y}<br>تنش: %{z:.3f}<extra></extra>"))
    fig.update_layout(title="تنش دوتایی (کشورِ کنش‌گر - کشورِ هدف)", xaxis_title="گام زمانی", yaxis_title="زوج کشورها", yaxis_autorange="reversed", height=min(900, 120 + 22 * len(y_labels)))
    st.plotly_chart(fig, use_container_width=True)

def plot_dyad_crisis_heatmap(df: pd.DataFrame, countries: list[str]):
    if df is None or len(df) == 0 or len(countries) < 2 or "Time" not in df.columns: return
    df = df.copy()
    df["Time"] = pd.to_numeric(df["Time"], errors="coerce").fillna(0).astype(int)
    df = df.sort_values("Time")
    times = df["Time"].unique().tolist()
    st.subheader("ماتریس حرارتی عاملِ بحران در طول زمان")
    view = st.radio("نمایش بر اساس", options=["رخداد واقعی (احتمال تجمیع‌شده)", "احتمال تئوریک"], horizontal=True, key="crisis_heatmap_view")

    pairs = [(src, dst) for src in countries for dst in countries if src != dst]
    y_labels = [f"{src} - {dst}" for src, dst in pairs]
    df_idx = df.set_index("Time")
    z = []
    for src, dst in pairs:
        col = f"Crisis_{src}_{dst}" if view.startswith("رخداد") else f"CrisisProb_{src}_{dst}"
        series = pd.to_numeric(df_idx[col], errors="coerce") if col in df_idx.columns else pd.Series([], dtype=float)
        z.append(series.reindex(times).fillna(0.0).astype(float).tolist())

    colorscale = [[0.0, "green"], [1.0, "red"]] if view.startswith("رخداد") else "RdYlGn_r"
    fig = go.Figure(data=go.Heatmap(z=z, x=times, y=y_labels, zmin=0, zmax=1, colorscale=colorscale, hovertemplate="زمان: %{x}<br>%{y}<br>ارزش: %{z:.3f}<extra></extra>"))
    fig.update_layout(title="عامل بحران (کشورِ کنش‌گر - کشورِ هدف)", xaxis_title="گام زمانی", yaxis_title="زوج کشورها", yaxis_autorange="reversed", height=min(900, 120 + 22 * len(y_labels)))
    st.plotly_chart(fig, use_container_width=True)

def _layout_positions_circle(names):
    n = len(names)
    pos = {}
    for i, name in enumerate(names):
        ang = 2 * math.pi * i / max(1, n)
        pos[name] = (math.cos(ang), math.sin(ang))
    return pos

def plot_interaction_graph_directed(df, countries):
    if df is None or len(df) == 0 or len(countries) < 2 or "Time" not in df.columns: return
    df = df.copy()
    df["Time"] = pd.to_numeric(df["Time"], errors="coerce").fillna(0).astype(int)
    max_t = int(df["Time"].max())

    st.subheader("گراف تعاملات جهت‌دار")
    c1, c2, c3 = st.columns(3)
    with c1: show_edges_P = st.checkbox("نمایش یال‌های آگاهی وضعیتی (P)", value=True, key="edgeP_action")
    with c2: show_edges_S = st.checkbox("نمایش یال‌های سیگنال (S)", value=True, key="edgeS_action")
    with c3: show_edges_R = st.checkbox("نمایش یال‌های تقویت/زور (R)", value=True, key="edgeR_action")
    visible_map = {"P": show_edges_P, "S": show_edges_S, "R": show_edges_R}

    if "graph_time" not in st.session_state: st.session_state["graph_time"] = 0
    t_selected = st.slider("زمان نمایش (t)", 0, max_t, int(st.session_state["graph_time"]), key="graph_time")
    pos = _layout_positions_circle(countries)
    act_color = {"P": "green", "S": "orange", "R": "red"}
    palette = px.colors.qualitative.Set2
    country_color = {c: palette[i % len(palette)] for i, c in enumerate(countries)}

    def _get_row_safe(t: int):
        sub = df.loc[df["Time"] == int(t)]
        if not sub.empty: return sub.iloc[0]
        return df.loc[(df["Time"] - int(t)).abs().idxmin()]

    def _shrink_segment(x0, y0, x1, y1, shrink=0.22):
        L = ((x1-x0)**2 + (y1-y0)**2)**0.5 + 1e-9
        return x0 + (x1-x0)*(shrink/L), y0 + (y1-y0)*(shrink/L), x0 + (x1-x0)*(1-shrink/L), y0 + (y1-y0)*(1-shrink/L)

    def build_frame(t: int):
        row = _get_row_safe(t)
        xs, ys, texts, fills, borders, hovers = [], [], [], [], [], []
        for c in countries:
            a = row.get(f"Action_{c}", "P")
            x, y = pos[c]
            xs.append(x); ys.append(y); texts.append(c); fills.append(country_color[c]); borders.append(act_color.get(a, "white"))
            hovers.append(f"{c}<br>اقدام: {a}")
        nodes_trace = go.Scatter(x=xs, y=ys, mode="markers+text", text=texts, textposition="bottom center", marker=dict(size=26, color=fills, line=dict(width=4, color=borders)), hovertext=hovers, hoverinfo="text", showlegend=False)

        seg_on, seg_off, anns = {"P": [], "S": [], "R": []}, {"P": [], "S": [], "R": []}, []
        for src in countries:
            a = row.get(f"Action_{src}", "P")
            if a not in ("P", "S", "R"): a = "P"
            tgt = row.get(f"Target_{src}", None)
            if tgt is None or tgt not in countries or tgt == src: tgt = countries[(countries.index(src) + 1) % len(countries)]
            
            y_val = float(row.get(f"Y_{src}_{tgt}", 0.0))
            x0, y0 = pos[src]; x1, y1 = pos[tgt]
            x0s, y0s, x1s, y1s = _shrink_segment(x0, y0, x1, y1)
            
            if y_val >= 0.5: seg_on[a].append((x0s, y0s, x1s, y1s))
            else: seg_off[a].append((x0s, y0s, x1s, y1s))
            anns.append(dict(x=x1s, y=y1s, ax=x0s, ay=y0s, xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=3, arrowsize=1.15, arrowwidth=2, arrowcolor=act_color[a], opacity=0.25 + 0.70 * y_val))

        traces = []
        for act in ["P", "S", "R"]:
            x_list, y_list = [], []
            for (x0s, y0s, x1s, y1s) in seg_off[act]: x_list += [x0s, x1s, None]; y_list += [y0s, y1s, None]
            traces.append(go.Scatter(x=x_list, y=y_list, mode="lines", line=dict(width=2, color=act_color[act], dash="dot"), hoverinfo="skip", showlegend=False, opacity=0.25, visible=True if visible_map[act] else "legendonly"))

            x_list2, y_list2 = [], []
            for (x0s, y0s, x1s, y1s) in seg_on[act]: x_list2 += [x0s, x1s, None]; y_list2 += [y0s, y1s, None]
            traces.append(go.Scatter(x=x_list2, y=y_list2, mode="lines", line=dict(width=4, color=act_color[act]), name=f"یال‌های {act}", hoverinfo="skip", showlegend=True, opacity=0.95, visible=True if visible_map[act] else "legendonly"))
        return (traces + [nodes_trace]), anns

    frames = []
    for t in range(max_t + 1):
        d, ann = build_frame(t)
        frames.append(go.Frame(data=d, name=str(t), layout=dict(annotations=ann)))

    base_data, base_ann = build_frame(t_selected)
    fig = go.Figure(data=base_data, frames=frames)
    fig.update_layout(title=f"گراف اقدام — زمان {t_selected}", xaxis=dict(visible=False, range=[-1.4, 1.4]), yaxis=dict(visible=False, range=[-1.3, 1.3]), margin=dict(l=20, r=20, t=60, b=150), annotations=base_ann, legend=dict(orientation="h", y=1.02, x=10.0), uirevision="1", updatemenus=[dict(type="buttons", direction="left", x=0.5, y=-0.20, buttons=[dict(label="▶ پخش", method="animate", args=[None, dict(frame=dict(duration=450, redraw=True), transition=dict(duration=150), fromcurrent=True, mode="immediate")]), dict(label="⏸ توقف", method="animate", args=[[None], dict(frame=dict(duration=0, redraw=False), transition=dict(duration=0), mode="immediate")])])])
    st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# 5) Streamlit UI
# ==========================================================
def main():
    st.set_page_config(page_title="شبیه‌ساز ژئوپلیتیک", layout="wide")
    st.title("🌍 شبیه‌ساز ژئوپلیتیک")

    scenarios = scenario_pack()
    st.sidebar.header("🧩 سناریوهای آماده")
    scenario_keys = ["custom"] + list(scenarios.keys())
    key_to_title = dict(zip(scenario_keys, ["سفارشی (دستی)"] + [scenarios[k]["title"] for k in scenarios.keys()]))

    chosen = st.sidebar.radio("انتخاب سناریو آماده", options=scenario_keys, format_func=lambda k: key_to_title[k], index=1, key="scenario_choice")
    _reset_on_scenario_change(chosen)

    if "last_scenario" not in st.session_state: st.session_state["last_scenario"] = chosen
    elif st.session_state["last_scenario"] != chosen:
        for k in list(st.session_state.keys()):
            if k not in {"scenario_choice", "last_scenario"}: del st.session_state[k]
        st.session_state["last_scenario"] = chosen
        st.rerun()

    st.sidebar.divider()
    st.sidebar.header("⚙️ تنظیمات اجرا")

    doctrine_update_every = st.sidebar.number_input("تغییر دکترین بعد از چند بار انجام اقدام؟", 0, 200, 0, 5)
    
    num_runs = st.sidebar.number_input("تعداد تکرار (میانگین‌گیری Monte Carlo)", min_value=1, max_value=200, value=1, step=1, help=tip("num_runs"))

    test_mode = st.sidebar.toggle("حالت تست (Test Mode)", value=False)
    seed = st.sidebar.number_input("عدد بذر تصادفی (Seed)", 0, 10_000_000, 42) if test_mode else None
    
    steps = st.sidebar.number_input("تعداد گام‌های زمانی", 10, 200, scenarios.get(chosen, {}).get("steps_default", 70), 5)
    run_btn = st.sidebar.button("🚀 اجرای شبیه‌سازی", type="primary", use_container_width=True)

    if chosen != "custom":
        st.sidebar.divider()
        st.sidebar.subheader("📘 توضیح سناریو")
        st.sidebar.markdown(scenarios[chosen]["story"])
        agent_cfgs, W, countries = build_custom_ui(prefill_agents=scenarios[chosen]["agents"], prefill_W=scenarios[chosen]["W"], prefill_countries=scenarios[chosen]["countries"], lock_n=True)
    else:
        st.sidebar.info("حالت سفارشی فعال است: پارامترها را تنظیم کنید.")
        agent_cfgs, W, countries = build_custom_ui()

    if "sim_df" not in st.session_state: st.session_state.sim_df = None
    if "sim_meta" not in st.session_state: st.session_state.sim_meta = None
    if "has_run" not in st.session_state: st.session_state.has_run = False

    if run_btn:
        with st.spinner(f"در حال اجرای شبیه‌سازی ({num_runs} بار)..."):
            df_avg, avg_meta, all_dfs = run_multiple_simulations(agent_cfgs, W, steps, test_mode, seed, doctrine_update_every, num_runs)
            st.session_state.sim_df = df_avg
            st.session_state.sim_meta = avg_meta
            st.session_state.all_dfs = all_dfs
            st.session_state.has_run = True

    if not st.session_state.has_run: st.stop()

    df = st.session_state.sim_df
    meta = st.session_state.sim_meta
    all_dfs = st.session_state.get("all_dfs", [df])

    if df is None or df.empty: return

    st.divider()
    st.subheader("خلاصه اقدامات (میانگین دفعات)")
    st.dataframe(df_action_counts(all_dfs, countries), use_container_width=True)
    
    st.divider()
    plot_three_indices_heatmaps(df, countries, window=10)

    st.divider()
    plot_global_escalation(df)

    st.divider()
    plot_dyad_crisis_heatmap(df, countries)

    st.divider()
    plot_lines_by_country(df, countries, prefix="Tension", title_fa="روند تنش کشورها (Tension)", y_label_fa="تنش (Tension)")

    st.divider()
    plot_lines_by_country(df, countries, prefix="Resource", title_fa="روند منابع کشورها (Resources)", y_label_fa="منابع (Resources)")

    st.divider()
    plot_lines_by_country(df, countries, prefix="Psi", title_fa="خروجی تشدید کشور (ψ_c)", y_label_fa="ψ_c")

    st.divider()
    plot_actions_map(df, countries)

    st.divider()
    plot_dyad_tension_heatmap(df, countries)

    st.divider()
    plot_interaction_graph_directed(df, countries)

    st.divider()
    st.subheader("تغییر پارامترها (ابتدا → انتها)")
    trans_df = build_transition_df(meta, countries)
    if trans_df is not None and not trans_df.empty:
        country_filter = st.multiselect("فیلتر کشورها", trans_df["کشور"].unique(), trans_df["کشور"].unique())
        section_filter = st.multiselect("فیلتر بخش‌ها", SECTION_ORDER, [s for s in trans_df["بخش"].unique() if s in SECTION_ORDER])
        show_df = trans_df[trans_df["کشور"].isin(country_filter) & trans_df["بخش"].isin(section_filter)].copy()
        show_df["Δ (تغییر)"] = show_df["انتها"] - show_df["ابتدا"]
        st.dataframe(show_df, use_container_width=True)
        st.download_button("⬇️ دانلود جدول تغییرات (CSV)", show_df.to_csv(index=False).encode("utf-8-sig"), "parameter_transitions.csv", "text/csv", use_container_width=True)

if __name__ == "__main__":
    main()
