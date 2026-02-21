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
# Scenario-change reset (wipe session_state keys on scenario switch)
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

    # Remove everything else (widgets + caches) so UI fully refreshes
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
# 1) Scenario pack (همان نسخه‌ای که دادی)
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
                res0=1100.0,
                v=0.55,
                rho=0.45,
                d=0.55,
                f=0.55,
                chi=1.05,
                wsec=3.2, winf=1.7, wcost=2.1,
                lambda_op=0.55,
                tau=5.0,
                eps=0.58,
                income=14.0,
                eta=1.10, kappa=1.00, pa=2.4, pb=2.2, ra=2.2, rb=2.2,
                beta=2.1,
                prefP=1.2, prefS=1.0, prefR=0.8,
            ),
            dict(
                name="نرمان",
                res0=1250.0,
                v=0.62,
                rho=0.35,
                d=0.78,
                f=0.45,
                chi=1.00,
                wsec=2.3, winf=3.2, wcost=1.8,
                lambda_op=0.45,
                tau=5.8,
                eps=0.55,
                income=16.0,
                eta=1.15, kappa=0.95, pa=2.6, pb=2.0, ra=2.5, rb=2.1,
                beta=1.9,
                prefP=0.9, prefS=1.4, prefR=0.7,
            ),
            dict(
                name="جلال",
                res0=1150.0,
                v=0.70,
                rho=0.60,
                d=0.48,
                f=0.62,
                chi=1.15,
                wsec=3.4, winf=1.4, wcost=2.2,
                lambda_op=0.60,
                tau=4.7,
                eps=0.60,
                income=13.0,
                eta=1.05, kappa=1.05, pa=2.2, pb=2.4, ra=2.0, rb=2.4,
                beta=2.2,
                prefP=1.0, prefS=0.8, prefR=1.2,
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

    # سناریو ۵ (پنج کشوری)
    scenarios["scenario_5"] = {
        "title": "سناریو ۵: بحران چندقطبی در تنگه تجاری (۵ کشور)",
        "countries": ["اوران", "سَحَر", "کایان", "مِهران", "وِستا"],
        "story": """
    **داستان سناریو (۵ کشور):**
    یک تنگه‌ی تجاری حیاتی وجود دارد که عبور انرژی و کالا از آن انجام می‌شود. پنج کشور فرضی درگیر رقابت و بازدارندگی‌اند:

    - **اوران**: قدرت دریایی و امنیت‌محور است؛ بیشتر به **آگاهی وضعیتی (P)** تکیه می‌کند تا مسیرها را امن نگه دارد.
    - **سَحَر**: جنگ روانی و رسانه‌ای قوی دارد؛ بیشتر **سیگنال (S)** می‌دهد تا نفوذ بسازد.
    - **کایان**: کشوری پرریسک‌تر و تهاجمی‌تر است؛ گاهی سراغ **تقویت/زور (R)** می‌رود و هزینه‌اش را می‌پردازد.
    - **مِهران**: میانجی‌گر و محافظه‌کار است؛ ترکیبی از **P و S** انجام می‌دهد و کمتر دنبال R می‌رود.
    - **وِستا**: اقتصاد قوی‌تر دارد ولی مرزهای حساس‌تری هم دارد؛ بسته به فشارها بین P و R نوسان می‌کند.

    **ساختار تعاملات (W):**
    - اوران و مهران تا حدی همسو هستند.
    - سحر بیشتر روی عملیات نفوذ و سیگنال علیه کایان و وستا کار می‌کند.
    - کایان بیشتر اوران و وستا را هدف می‌گیرد (فشار مستقیم).
    - وستا نسبت به کایان حساس است و گاهی پاسخ سنگین‌تر می‌دهد.

    **انتظار خروجی:**
    - در نقشه اقدامات، برای **سحر** سیگنال‌ها زیاد دیده می‌شود.
    - برای **کایان** در دوره‌هایی R دیده می‌شود و منابعش می‌تواند افت کند.
    - در گراف تعاملات، در هر زمان برای هر کشور یک یال (action→target) وجود دارد:
      - **اگر Y=1** باشد یال پررنگ است،
      - **اگر Y=0** باشد یال کم‌رنگ و خط‌چین است.
    """,
        # W[i][j] = شدت تعامل/احتمال هدف‌گیری j توسط i (قطر اصلی باید 0 باشد)
        "W": _W01_to_signed([
            #          اوران  سحر   کایان  مهران  وستا
            [0.0, 0.15, 0.45, 0.25, 0.15],  # اوران بیشتر کایان و مهران را زیرنظر دارد
            [0.20, 0.0, 0.35, 0.10, 0.35],  # سحر بیشتر روی کایان و وستا سیگنال می‌دهد
            [0.35, 0.10, 0.0, 0.10, 0.45],  # کایان بیشتر اوران و وستا را هدف می‌گیرد
            [0.30, 0.15, 0.25, 0.0, 0.30],  # مهران متعادل و میانجی: با همه تعامل دارد
            [0.25, 0.10, 0.45, 0.20, 0.0],  # وستا حساس به کایان و تا حدی اوران
        ]),
        "agents": [
            dict(
                name="اوران",
                res0=1450.0,
                v=0.58,
                rho=0.38, d=0.55, f=0.58, chi=1.05,
                wsec=3.4, winf=1.8, wcost=2.2,
                lambda_op=0.60, tau=5.1, eps=0.58,
                income=17.0,
                eta=1.15, kappa=0.95, pa=2.6, pb=2.2, ra=2.4, rb=2.3,
                beta=2.1,
                prefP=1.35, prefS=0.95, prefR=0.70,
            ),
            dict(
                name="سَحَر",
                res0=1350.0,
                v=0.62,
                rho=0.34, d=0.86, f=0.48, chi=0.95,
                wsec=2.2, winf=3.6, wcost=1.6,
                lambda_op=0.45, tau=6.0, eps=0.54,
                income=16.0,
                eta=1.20, kappa=0.90, pa=2.7, pb=2.0, ra=2.6, rb=2.1,
                beta=1.9,
                prefP=0.80, prefS=1.55, prefR=0.65,
            ),
            dict(
                name="کایان",
                res0=1500.0,
                v=0.74,
                rho=0.62, d=0.42, f=0.62, chi=1.25,
                wsec=3.6, winf=1.3, wcost=2.6,
                lambda_op=0.70, tau=4.6, eps=0.62,
                income=15.0,
                eta=1.05, kappa=1.10, pa=2.2, pb=2.6, ra=2.0, rb=2.7,
                beta=2.3,
                prefP=0.85, prefS=0.70, prefR=1.45,
            ),
            dict(
                name="مِهران",
                res0=1400.0,
                v=0.56,
                rho=0.30, d=0.70, f=0.64, chi=1.00,
                wsec=2.8, winf=2.7, wcost=1.9,
                lambda_op=0.52, tau=5.8, eps=0.56,
                income=16.0,
                eta=1.18, kappa=0.95, pa=2.6, pb=2.1, ra=2.5, rb=2.2,
                beta=2.0,
                prefP=1.10, prefS=1.20, prefR=0.65,
            ),
            dict(
                name="وِستا",
                res0=1650.0,
                v=0.72,
                rho=0.50, d=0.55, f=0.55, chi=1.10,
                wsec=3.2, winf=1.9, wcost=2.3,
                lambda_op=0.62, tau=5.0, eps=0.60,
                income=18.0,
                eta=1.10, kappa=1.00, pa=2.4, pb=2.3, ra=2.3, rb=2.4,
                beta=2.2,
                prefP=1.05, prefS=0.90, prefR=1.10,
            ),
        ],
        "steps_default": 85,
    }

    scenarios["scenario_6"] = {
        "title": "سناریو ۶: رقابت و بازدارندگی قدرت‌های بزرگ (ایران/اسرائیل/آمریکا/چین/روسیه)",
        "countries": ["ایران", "اسرائیل", "آمریکا", "چین", "روسیه"],
        "story": """
**داستان سناریو:**
پنج بازیگر مهم با مجموعه‌ای از تقابل‌ها و همسویی‌ها همزمان در یک محیط پرتنش حضور دارند.

- **ایران** بیشترین تقابل را با **اسرائیل** و **آمریکا** دارد و در عین حال با **چین** و **روسیه** همسویی نسبی دارد.
- **اسرائیل** همسویی بالایی با **آمریکا** دارد و با **ایران** در تقابل شدید است.
- **آمریکا** با **چین** و **روسیه** رقابت جدی دارد و در مقابل **اسرائیل** همسو است.
- **چین** و **روسیه** تا حدی همسو هستند ولی هر کدام رقابت خود را با آمریکا دارند.

**معنای W:**
اعداد بین **-۱ تا +۱** هستند: **-۱ یعنی بیشترین تقابل** و **+۱ یعنی بیشترین همسویی**.
""",
        "W": [
            #            ایران  اسرائیل  آمریکا   چین   روسیه
            [0.0, -0.90, -0.80, 0.45, 0.30],  # ایران
            [-0.85, 0.0, 0.80, -0.20, -0.30],  # اسرائیل
            [-0.75, 0.85, 0.0, -0.70, -0.80],  # آمریکا
            [0.35, -0.15, -0.65, 0.0, 0.60],  # چین
            [0.25, -0.25, -0.75, 0.55, 0.0],  # روسیه
        ],
        "agents": [
            dict(
                name="ایران",
                res0=1250.0, v=0.72,
                rho=0.58, d=0.55, f=0.58, chi=1.10,
                wsec=3.4, winf=2.0, wcost=2.2,
                lambda_op=0.58, tau=5.0, eps=0.60, income=15.0,
                eta=1.05, kappa=1.05, pa=2.3, pb=2.4, ra=2.2, rb=2.4,
                beta=2.2, prefP=1.0, prefS=1.0, prefR=1.2,
            ),
            dict(
                name="اسرائیل",
                res0=1150.0, v=0.60,
                rho=0.42, d=0.60, f=0.52, chi=1.05,
                wsec=3.8, winf=2.0, wcost=2.0,
                lambda_op=0.62, tau=5.2, eps=0.56, income=16.0,
                eta=1.18, kappa=0.95, pa=2.7, pb=2.0, ra=2.6, rb=2.1,
                beta=2.0, prefP=1.1, prefS=1.0, prefR=1.0,
            ),
            dict(
                name="آمریکا",
                res0=1800.0, v=0.50,
                rho=0.40, d=0.55, f=0.55, chi=1.00,
                wsec=3.6, winf=2.1, wcost=2.4,
                lambda_op=0.70, tau=5.4, eps=0.58, income=20.0,
                eta=1.25, kappa=0.90, pa=2.8, pb=1.9, ra=2.7, rb=2.0,
                beta=2.0, prefP=1.1, prefS=0.95, prefR=0.95,
            ),
            dict(
                name="چین",
                res0=1750.0, v=0.55,
                rho=0.38, d=0.78, f=0.50, chi=1.00,
                wsec=2.8, winf=3.2, wcost=2.0,
                lambda_op=0.62, tau=5.8, eps=0.56, income=19.0,
                eta=1.20, kappa=0.95, pa=2.7, pb=2.0, ra=2.6, rb=2.1,
                beta=1.9, prefP=0.95, prefS=1.25, prefR=0.80,
            ),
            dict(
                name="روسیه",
                res0=1550.0, v=0.62,
                rho=0.55, d=0.55, f=0.60, chi=1.15,
                wsec=3.3, winf=1.8, wcost=2.2,
                lambda_op=0.65, tau=4.9, eps=0.60, income=17.0,
                eta=1.10, kappa=1.05, pa=2.4, pb=2.3, ra=2.3, rb=2.4,
                beta=2.1, prefP=0.95, prefS=0.95, prefR=1.15,
            ),
        ],
        "steps_default": 90,
    }

    return scenarios


# ==========================================================
# 2) Custom UI (کاملاً قابل تنظیم)
# ==========================================================
def build_custom_ui(prefill_agents=None, prefill_W=None, prefill_countries=None, lock_n: bool = False):
    st.subheader("🧰 تنظیمات سفارشی کشورها و تعاملات")

    # --- prefill (برای سناریوهای آماده) ---
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
        st.number_input(
            "تعداد کشورها",
            min_value=n,
            max_value=n,
            value=n,
            step=1,
            key="custom_n_locked",
            disabled=True,
            help=tip("custom_n"),
        )
    else:
        n = st.slider(
            "تعداد کشورها",
            min_value=int(n_min),
            max_value=int(n_max),
            value=int(n_default),
            step=1,
            key="custom_n",
            help=tip("custom_n"),
        )
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
            st.session_state.custom_country_names[i] = st.text_input(
                f"نام کشور {i + 1}",
                value=st.session_state.custom_country_names[i],
                key=f"custom_name_{i}",
                help=tip("country_name"),
            )
    countries = st.session_state.custom_country_names

    st.divider()

    def default_agent_cfg(name):
        return dict(
            name=name,
            res0=1200.0, v=0.60,
            rho=0.45, d=0.65, f=0.55, chi=1.05,
            wsec=3.0, winf=2.0, wcost=2.0,
            lambda_op=0.55, tau=5.5, eps=0.56, income=15.0,
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
            cfg["res0"] = c1.number_input("منابع اولیه (res0)", 0.0, 10000.0, float(cfg["res0"]), 50.0, key=f"res0_{i}",
                                          help=tip("res0"))
            cfg["income"] = c2.number_input("درآمد هر گام (μ_c)", 0.0, 200.0, float(cfg["income"]), 1.0,
                                            key=f"income_{i}", help=tip("income"))
            cfg["v"] = c3.slider("آسیب‌پذیری مرزی (v_c)", 0.0, 1.0, float(cfg["v"]), 0.01, key=f"v_{i}",
                                 help=tip("v_c"))
            cfg["chi"] = c4.slider("ضریب هزینه منابع (χ_c)", 0.1, 3.0, float(cfg["chi"]), 0.01, key=f"chi_{i}",
                                   help=tip("chi_c"))

            st.markdown("### 2) دکترین (Doctrine)")
            d1, d2, d3 = st.columns(3)
            cfg["rho"] = d1.slider("ریسک‌پذیری (ρ_c)", 0.0, 1.0, float(cfg["rho"]), 0.01, key=f"rho_{i}",
                                   help=tip("rho_c"))
            cfg["d"] = d2.slider("ترجیح بازدارندگی/نفوذ (d_c)", 0.0, 1.0, float(cfg["d"]), 0.01, key=f"d_{i}",
                                 help=tip("d_c"))
            cfg["f"] = d3.slider("آستانه زور (f_c)", 0.0, 1.0, float(cfg["f"]), 0.01, key=f"f_{i}", help=tip("f_c"))

            st.markdown("### 3) وزن‌های راهبردی (ω_S)")

            w1, w2, w3 = st.columns(3)

            cfg["wsec"] = w1.number_input(
                "اهمیت امنیت (ω_sec)",
                0.0, 10.0,
                float(cfg["wsec"]),
                0.1,
                key=f"wsec_{i}",
                help=tip("wsec"),
            )

            cfg["winf"] = w2.number_input(
                "اهمیت نفوذ (ω_inf)",
                0.0, 10.0,
                float(cfg["winf"]),
                0.1,
                key=f"winf_{i}",
                help=tip("winf"),
            )

            cfg["wcost"] = w3.number_input(
                "اهمیت هزینه (ω_cost)",
                0.0, 10.0,
                float(cfg["wcost"]),
                0.1,
                key=f"wcost_{i}",
                help=tip("wcost"),
            )

            total = cfg["wsec"] + cfg["winf"] + cfg["wcost"]

            if total > 0:
                p_sec = cfg["wsec"] / total
                p_inf = cfg["winf"] / total
                p_cost = cfg["wcost"] / total
            else:
                p_sec = p_inf = p_cost = 0

            st.markdown("#### 📊 سهم واقعی هر وزن در تصمیم‌گیری:")

            c1, c2, c3 = st.columns(3)
            c1.metric("سهم امنیت", f"{p_sec * 100:.1f}%")
            c2.metric("سهم نفوذ", f"{p_inf * 100:.1f}%")
            c3.metric("سهم هزینه", f"{p_cost * 100:.1f}%")

            st.caption(f"جمع کل وزن‌های وارد شده = {round(total, 2)}")

            st.markdown("### 4) عملیاتی (Operational)")
            o1, o2, o3 = st.columns(3)
            cfg["lambda_op"] = o1.slider("توان تخصیص عملیات (λ_op)", 0.0, 1.0, float(cfg["lambda_op"]), 0.01,
                                         key=f"lambdaop_{i}", help=tip("lambda_op"))
            cfg["tau"] = o2.slider("سرعت/Tempo (τ_c)", 1.0, 15.0, float(cfg["tau"]), 0.1, key=f"tau_{i}",
                                   help=tip("tau_c"))
            cfg["eps"] = o3.slider("حد شروع واکنش شدید (ε_c)", 0.0, 1.0, float(cfg["eps"]), 0.01, key=f"eps_{i}",
                                   help=tip("eps_c"))

            st.markdown("### 5) تاکتیک (Tactical)")
            b1, _ = st.columns(2)
            cfg["beta"] = b1.slider("حساسیت به سود (β_c)", 0.1, 10.0, float(cfg["beta"]), 0.1, key=f"beta_{i}",
                                    help=tip("beta_c"))
            st.caption("ترجیحات ذاتی اقدامات (ω_a): هرچه بزرگ‌تر باشد، آن اقدام بیشتر انتخاب می‌شود.")
            p1, p2, p3 = st.columns(3)
            cfg["prefP"] = p1.number_input("ترجیح آگاهی وضعیتی (ω_P)", 0.1, 10.0, float(cfg["prefP"]), 0.1,
                                           key=f"prefP_{i}", help=tip("prefP"))
            cfg["prefS"] = p2.number_input("ترجیح سیگنال (ω_S)", 0.1, 10.0, float(cfg["prefS"]), 0.1, key=f"prefS_{i}",
                                           help=tip("prefS"))
            cfg["prefR"] = p3.number_input("ترجیح تقویت/زور (ω_R)", 0.1, 10.0, float(cfg["prefR"]), 0.1,
                                           key=f"prefR_{i}", help=tip("prefR"))

            st.markdown("### 6) تکنیک (Technical)")
            t1, t2, t3, t4 = st.columns(4)
            cfg["eta"] = t1.slider("توان فنی (η_c)", 0.2, 3.0, float(cfg["eta"]), 0.01, key=f"eta_{i}",
                                   help=tip("eta_c"))
            cfg["kappa"] = t2.slider("هزینه یادگیری (κ_c)", 0.2, 3.0, float(cfg["kappa"]), 0.01, key=f"kappa_{i}",
                                     help=tip("kappa_c"))
            cfg["pa"] = t3.number_input("موفقیت فرضی (p:α)", 1.0, 10.0, float(cfg["pa"]), 0.1, key=f"pa_{i}",
                                        help=tip("p_alpha"))
            cfg["pb"] = t4.number_input("شکست فرضی (p:β)", 1.0, 10.0, float(cfg["pb"]), 0.1, key=f"pb_{i}",
                                        help=tip("p_beta"))
            r1, r2 = st.columns(2)
            cfg["ra"] = r1.number_input("پایداری/اطمینان (r:α)", 1.0, 10.0, float(cfg["ra"]), 0.1, key=f"ra_{i}",
                                        help=tip("r_alpha"))
            cfg["rb"] = r2.number_input("خرابی/بی‌ثباتی (r:β)", 1.0, 10.0, float(cfg["rb"]), 0.1, key=f"rb_{i}",
                                        help=tip("r_beta"))

        st.session_state.custom_agents[i] = cfg

    st.divider()
    st.markdown("### 🔁 ماتریس تعاملات کشورها (W)")
    st.caption(
        "W[i,j] رابطه‌ی کشور i با کشور j است. بازه بین -۱ تا +۱: -۱=بیشترین تقابل، +۱=بیشترین همسویی. قطر ماتریس همیشه ۰ است.")
    # راهنما به صورت پاپ‌آپ
    if hasattr(st, "popover"):
        with st.popover("راهنما: W"):
            st.write(tip("W"))
    else:
        with st.expander("راهنما: W", expanded=False):
            st.write(tip("W"))

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
    edited = st.data_editor(W_df, use_container_width=True, key="W_editor", disabled=False)

    edited = edited.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    edited = edited.clip(lower=-1.0, upper=1.0)
    np.fill_diagonal(edited.values, 0.0)
    st.session_state.custom_W = edited.values

    if st.button("↩️ بازنشانی مقادیر سفارشی به پیش‌فرض", use_container_width=True, help=tip("reset_btn")):
        st.session_state.custom_agents = [default_agent_cfg(c) for c in countries]
        W2 = np.zeros((n, n), dtype=float)
        np.fill_diagonal(W2, 0.0)
        st.session_state.custom_W = W2
        st.rerun()

    return st.session_state.custom_agents, st.session_state.custom_W.tolist(), countries


# ==========================================================
# 3) Build agents + run simulation (با meta)
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
                name=c["name"],
                initial_resource=float(c["res0"]),
                v_c=float(c["v"]),
                rho_c=float(c["rho"]),
                d_c=float(c["d"]),
                f_c=float(c["f"]),
                chi_c=float(c["chi"]),
                omega_S=np.array(omega_S, dtype=float),
                lambda_op=float(c["lambda_op"]),
                tau_c=float(c["tau"]),
                eps_c=float(c["eps"]),
                income_c=float(c["income"]),
                eta_c=float(c["eta"]),
                p_params=[float(c["pa"]), float(c["pb"])],
                r_params=[float(c["ra"]), float(c["rb"])],
                kappa_c=float(c["kappa"]),
                beta_c=float(c["beta"]),
                omega_a=np.array(omega_a, dtype=float),
                action_bases=action_bases,
                dyn_coeffs=dyn,
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
        agents=agents,
        interaction_W=W,
        esc_coeffs=EscalationCoeffs(),
        doctrine_update_every=int(doctrine_update_every),
    )
    for t in range(int(steps)):
        world.step(t)

    meta["final"] = {ag.name: ag.snapshot() for ag in agents}

    df = pd.DataFrame(world.history)
    return df, meta


# ==========================================================
# 4) Tables + charts
# ==========================================================
def df_action_counts(df, countries):
    out = []
    for c in countries:
        col = f"Action_{c}"
        if col in df.columns:
            vc = df[col].value_counts().to_dict()
            out.append({
                "کشور": c,
                "آگاهی وضعیتی (P)": int(vc.get("P", 0)),
                "سیگنال (S)": int(vc.get("S", 0)),
                "تقویت/زور (R)": int(vc.get("R", 0)),
            })
    return pd.DataFrame(out)


def _resource_norm(x: float) -> float:
    x = float(x)
    return x / (x + 1000.0)


def compute_three_indices(df: pd.DataFrame, countries: list[str], window: int = 10):
    df = df.copy()
    max_t = int(df["Time"].max())

    rows = []
    for c in countries:
        tcol = f"Tension_{c}"
        rcol = f"Resource_{c}"
        pcol = f"Psi_{c}"
        acol = f"Action_{c}"

        if not all(col in df.columns for col in [tcol, rcol, pcol, acol]):
            continue

        rnorm = df[rcol].apply(_resource_norm)

        trend = df[rcol] - df[rcol].shift(5)
        trend_norm = 1.0 / (1.0 + np.exp(-(trend.fillna(0.0) / 100.0)))

        is_S = (df[acol] == "S").astype(int)
        sshare = is_S.rolling(window=window, min_periods=1).mean()

        security = (
                0.45 * (1.0 - df[tcol].astype(float))
                + 0.35 * rnorm.astype(float)
                + 0.20 * (1.0 - df[pcol].astype(float))
        ).clip(0, 1)

        resilience = (
                0.55 * rnorm.astype(float)
                + 0.25 * (1.0 - df[pcol].astype(float))
                + 0.20 * trend_norm.astype(float)
        ).clip(0, 1)

        influence = (
                0.60 * sshare.astype(float)
                + 0.25 * (1.0 - df[tcol].astype(float))
                + 0.15 * (1.0 - df[pcol].astype(float))
        ).clip(0, 1)

        start_idx = df.index[df["Time"] == 0][0]
        end_idx = df.index[df["Time"] == max_t][0]

        def pack(series):
            s0 = float(series.loc[start_idx])
            s1 = float(series.loc[end_idx])
            delta = s1 - s0
            pct = (delta / (s0 + 1e-12)) * 100.0
            return s0, s1, delta, pct

        sec0, sec1, secD, secP = pack(security)
        res0, res1, resD, resP = pack(resilience)
        inf0, inf1, infD, infP = pack(influence)

        rows.append({
            "کشور": c,

            "قدرت امنیت (ابتدا)": round(sec0, 3),
            "قدرت امنیت (انتها)": round(sec1, 3),
            "تغییر امنیت": round(secD, 3),
            "٪ تغییر امنیت": round(secP, 1),

            "تاب‌آوری (ابتدا)": round(res0, 3),
            "تاب‌آوری (انتها)": round(res1, 3),
            "تغییر تاب‌آوری": round(resD, 3),
            "٪ تغییر تاب‌آوری": round(resP, 1),

            "نفوذ/بازدارندگی (ابتدا)": round(inf0, 3),
            "نفوذ/بازدارندگی (انتها)": round(inf1, 3),
            "تغییر نفوذ": round(infD, 3),
            "٪ تغییر نفوذ": round(infP, 1),
        })

    return pd.DataFrame(rows)


def plot_three_indices_heatmaps(df: pd.DataFrame, countries: list[str], window: int = 10):
    """
    نمایش سه شاخص کلیدی به صورت نقشه حرارتی در طول زمان.
    سبز = بهتر/قوی‌تر ، قرمز = ضعیف‌تر
    """
    if df is None or len(df) == 0 or len(countries) == 0:
        return
    if "Time" not in df.columns:
        return

    dfx = df.copy()
    dfx["Time"] = pd.to_numeric(dfx["Time"], errors="coerce").fillna(0).astype(int)
    # یک ردیف برای هر زمان (اگر تکراری شد)
    dfx = dfx.sort_values("Time").drop_duplicates(subset=["Time"]).set_index("Time")
    times = dfx.index.astype(int).tolist()
    if not times:
        return

    # محاسبه شاخص‌ها برای هر کشور
    valid = []
    sec_z, res_z, inf_z = [], [], []

    for c in countries:
        tcol = f"Tension_{c}"
        rcol = f"Resource_{c}"
        pcol = f"Psi_{c}"
        acol = f"Action_{c}"
        if not all(col in dfx.columns for col in [tcol, rcol, pcol, acol]):
            continue

        tension = dfx[tcol].astype(float)
        resource = dfx[rcol].astype(float)
        psi = dfx[pcol].astype(float)
        action = dfx[acol]

        rnorm = resource.apply(_resource_norm)

        trend = resource - resource.shift(5)
        trend_norm = 1.0 / (1.0 + np.exp(-(trend.fillna(0.0) / 100.0)))

        is_S = (action == "S").astype(int)
        sshare = is_S.rolling(window=window, min_periods=1).mean()

        security = (
                0.45 * (1.0 - tension)
                + 0.35 * rnorm
                + 0.20 * (1.0 - psi)
        ).clip(0, 1)

        resilience = (
                0.55 * rnorm
                + 0.25 * (1.0 - psi)
                + 0.20 * trend_norm
        ).clip(0, 1)

        influence = (
                0.60 * sshare
                + 0.25 * (1.0 - tension)
                + 0.15 * (1.0 - psi)
        ).clip(0, 1)

        valid.append(c)
        sec_z.append(security.reindex(times).fillna(0.0).tolist())
        res_z.append(resilience.reindex(times).fillna(0.0).tolist())
        inf_z.append(influence.reindex(times).fillna(0.0).tolist())

    if not valid:
        st.info("برای نمایش شاخص‌ها داده کافی وجود ندارد.")
        return

    st.subheader("۳ شاخص کلیدی در طول زمان (نقشه حرارتی)")
    with st.expander("📌 راهنما", expanded=False):
        st.markdown(
            """
- **هر سطر یک کشور** است و **هر ستون یک گام زمانی**.  
- هرچه رنگ **سبزتر** باشد یعنی وضعیت **بهتر/قوی‌تر** است.  
- هرچه رنگ **قرمزتر** باشد یعنی وضعیت **ضعیف‌تر** است.
"""
        )

    tabs = st.tabs(["قدرت امنیت", "تاب‌آوری", "نفوذ/بازدارندگی"])

    def _heat(z, title):
        fig = go.Figure(
            data=go.Heatmap(
                z=z,
                x=times,
                y=valid,
                zmin=0,
                zmax=1,
                colorscale="RdYlGn",
                hovertemplate="%{y}<br>زمان: %{x}<br>امتیاز: %{z:.3f}<extra></extra>",
            )
        )
        fig.update_layout(
            title=title,
            xaxis_title="گام زمانی",
            yaxis_title="کشور",
            margin=dict(l=30, r=30, t=60, b=30),
        )
        st.plotly_chart(fig, use_container_width=True)

    with tabs[0]:
        _heat(sec_z, "قدرت امنیت در طول زمان")
    with tabs[1]:
        _heat(res_z, "تاب‌آوری در طول زمان")
    with tabs[2]:
        _heat(inf_z, "نفوذ/بازدارندگی در طول زمان")


def build_transition_df(meta, countries):
    if not meta or "initial" not in meta or "final" not in meta:
        return pd.DataFrame()

    rows = []
    for c in countries:
        ini = meta["initial"].get(c, {})
        fin = meta["final"].get(c, {})
        if not ini or not fin:
            continue

        # ---- Doctrine ----
        for k, label in [
            ("rho_c", "ریسک‌پذیری (ρ_c)"),
            ("d_c", "ترجیح بازدارندگی/نفوذ (d_c)"),
            ("f_c", "آستانه زور (f_c)"),
            ("chi_c", "ضریب هزینه منابع (χ_c)"),
        ]:
            rows.append({"کشور": c, "بخش": "دکترین", "پارامتر": label, "ابتدا": float(ini.get(k, np.nan)),
                         "انتها": float(fin.get(k, np.nan))})

        # ---- Strategic ----
        for idx, label in enumerate(["ω_sec", "ω_inf", "ω_cost"]):
            a0 = float(ini.get("omega_S", [np.nan, np.nan, np.nan])[idx])
            a1 = float(fin.get("omega_S", [np.nan, np.nan, np.nan])[idx])
            rows.append({"کشور": c, "بخش": "راهبرد", "پارامتر": f"وزن راهبردی ({label})", "ابتدا": a0, "انتها": a1})

        # ---- Technical ----
        for k, label in [
            ("p_c", "میانگین موفقیت (p_c)"),
            ("r_c", "میانگین قابلیت اطمینان (r_c)"),
            ("eta_c", "توان فنی (η_c)"),
            ("kappa_c", "هزینه یادگیری (κ_c)"),
        ]:
            rows.append({"کشور": c, "بخش": "تکنیک", "پارامتر": label, "ابتدا": float(ini.get(k, np.nan)),
                         "انتها": float(fin.get(k, np.nan))})

        # ---- Tactical ----
        rows.append(
            {"کشور": c, "بخش": "تاکتیک", "پارامتر": "حساسیت به سود (β_c)", "ابتدا": float(ini.get("beta_c", np.nan)),
             "انتها": float(fin.get("beta_c", np.nan))})
        for idx, label in enumerate(["ω_P", "ω_S", "ω_R"]):
            b0 = float(ini.get("omega_a", [np.nan, np.nan, np.nan])[idx])
            b1 = float(fin.get("omega_a", [np.nan, np.nan, np.nan])[idx])
            rows.append({"کشور": c, "بخش": "تاکتیک", "پارامتر": f"ترجیح تاکتیکی ({label})", "ابتدا": b0, "انتها": b1})

        # ---- State ----
        for k, label in [
            ("tension", "تنش (Tension)"),
            ("resource", "منابع (Resources)"),
        ]:
            rows.append({"کشور": c, "بخش": "وضعیت", "پارامتر": label, "ابتدا": float(ini.get(k, np.nan)),
                         "انتها": float(fin.get(k, np.nan))})

    return pd.DataFrame(rows)


def plot_global_escalation(df):
    if "Global_Escalation" not in df.columns:
        return
    fig = px.area(
        df,
        x="Time",
        y="Global_Escalation",
        labels={"Time": "گام زمانی", "Global_Escalation": "وضعیت بحران کلی"},
        title="وضعیت بحران کلی",
        color_discrete_sequence=["red"],  # 👈 رنگ قرمز
    )
    fig.update_yaxes(range=[0, 1])
    st.plotly_chart(fig, use_container_width=True, )


def plot_lines_by_country(df, countries, prefix, title_fa, y_label_fa):
    cols = [f"{prefix}_{c}" for c in countries if f"{prefix}_{c}" in df.columns]
    if not cols:
        return
    dfl = df.melt(id_vars=["Time"], value_vars=cols, var_name="variable", value_name="value")
    dfl["کشور"] = dfl["variable"].str.replace(f"{prefix}_", "", regex=False)

    fig = px.line(
        dfl,
        x="Time",
        y="value",
        color="کشور",
        title=title_fa,
        labels={"Time": "گام زمانی", "value": y_label_fa, "کشور": "کشور"},
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_actions_map(df, countries):
    act_cols = [f"Action_{c}" for c in countries if f"Action_{c}" in df.columns]
    if not act_cols:
        return

    df_m = df.melt(id_vars=["Time"], value_vars=act_cols, var_name="Country", value_name="Action")
    df_m["Country"] = df_m["Country"].str.replace("Action_", "", regex=False)

    uniq = df_m["Country"].unique().tolist()
    to_y = {c: i for i, c in enumerate(uniq)}
    df_m["y_base"] = df_m["Country"].map(to_y)

    offset_map = {"P": 0.00, "S": 0.12, "R": 0.24}
    df_m["y"] = df_m["y_base"] + df_m["Action"].map(offset_map).fillna(0.0)
    df_m["اقدام"] = df_m["Action"].map(ACTION_LABEL_FA).fillna(df_m["Action"])

    fig = px.scatter(
        df_m,
        x="Time",
        y="y",
        color="اقدام",
        symbol="اقدام",
        title="نقشه اقدامات",
        labels={"Time": "گام زمانی", "y": "کشور"},
    )

    tickvals = [to_y[c] + 0.12 for c in uniq]
    fig.update_yaxes(
        tickmode="array",
        tickvals=tickvals,
        ticktext=uniq,
        title="کشور",
    )

    st.plotly_chart(fig, use_container_width=True)


def plot_dyad_tension_heatmap(df: pd.DataFrame, countries: list[str]):
    """Heatmap of directed dyadic tension over all time.
    Rows: (src - dst) ، Columns: Time ، Values in [0,1].
    """
    if df is None or len(df) == 0 or len(countries) < 2:
        return
    if "Time" not in df.columns:
        return

    # quick existence check
    sample_src = countries[0]
    sample_dst = countries[1]
    if f"DyadTension_{sample_src}_{sample_dst}" not in df.columns:
        return

    df = df.copy()
    df["Time"] = pd.to_numeric(df["Time"], errors="coerce").fillna(0).astype(int)
    df = df.sort_values("Time")
    times = df["Time"].unique().tolist()

    st.subheader("ماتریس حرارتی تنش بین کشورها در طول زمان")
    with st.expander("📌 راهنما", expanded=False):
        st.markdown(
            """
- هر **ردیف** یک زوج کشور است: **کشورِ کنش‌گر - کشورِ هدف**.
- هر **ستون** یک گام زمانی است.
- **سبز = تنش کم** و **قرمز = تنش زیاد**.
"""
        )

    # build pair list (directed)
    pairs = [(src, dst) for src in countries for dst in countries if src != dst]
    y_labels = [f"{src} - {dst}" for src, dst in pairs]

    # build z matrix: rows=pairs, cols=times
    df_idx = df.set_index("Time")
    z = []
    for src, dst in pairs:
        col = f"DyadTension_{src}_{dst}"
        series = pd.to_numeric(df_idx[col], errors="coerce") if col in df_idx.columns else pd.Series([], dtype=float)
        series = series.reindex(times).fillna(0.0)
        z.append(series.astype(float).tolist())

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=times,
            y=y_labels,
            zmin=0,
            zmax=1,
            colorscale="RdYlGn_r",  # 0 green, 1 red
            hovertemplate="زمان: %{x}<br>%{y}<br>تنش: %{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="تنش دوتایی (کشورِ کنش‌گر - کشورِ هدف) در طول زمان",
        xaxis_title="گام زمانی",
        yaxis_title="زوج کشورها",
        yaxis_autorange="reversed",
        margin=dict(l=30, r=30, t=60, b=30),
        height=min(900, 120 + 22 * len(y_labels)),
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_dyad_crisis_heatmap(df: pd.DataFrame, countries: list[str]):
    """Heatmap of directed dyadic crisis attribution over all time.

    - رخداد واقعی: 0/1 (آیا در آن لحظه بحران واقعاً اتفاق افتاده؟)
    - احتمال: عدد بین 0 تا 1 (چقدر روی لبه‌ی بحران بوده؟)
    """
    if df is None or len(df) == 0 or len(countries) < 2:
        return
    if "Time" not in df.columns:
        return

    sample_src = countries[0]
    sample_dst = countries[1]
    if f"Crisis_{sample_src}_{sample_dst}" not in df.columns:
        return

    df = df.copy()
    df["Time"] = pd.to_numeric(df["Time"], errors="coerce").fillna(0).astype(int)
    df = df.sort_values("Time")
    times = df["Time"].unique().tolist()

    st.subheader("ماتریس حرارتی عاملِ بحران در طول زمان")
    with st.expander("📌 راهنما", expanded=False):
        st.markdown(
            """
این نمودار کمک می‌کند بفهمیم **بحران دقیقاً بین کدام دو کشور** شکل گرفته است.

**دو حالت نمایش:**
- **رخداد واقعی (0/1):** در همین لحظه *واقعاً* بحران بین این دو کشور رخ داده یا نه؟  
  - **0 (سبز)** یعنی رخ نداده، **1 (قرمز)** یعنی رخ داده.
- **احتمال:** نشان می‌دهد *چقدر احتمال داشته* بحران رخ بدهد (بین 0 تا 1).  
  - عدد بزرگ‌تر یعنی «ریسک بالاتر» (قرمزتر).
"""
        )

    view = st.radio(
        "نمایش بر اساس",
        options=["رخداد واقعی (0/1)", "احتمال"],
        horizontal=True,
        index=0,
        key="crisis_heatmap_view",
    )

    # build pair list (directed)
    pairs = [(src, dst) for src in countries for dst in countries if src != dst]
    y_labels = [f"{src} - {dst}" for src, dst in pairs]

    df_idx = df.set_index("Time")
    z = []
    for src, dst in pairs:
        if view.startswith("رخداد"):
            col = f"Crisis_{src}_{dst}"
        else:
            col = f"CrisisProb_{src}_{dst}"

        series = pd.to_numeric(df_idx[col], errors="coerce") if col in df_idx.columns else pd.Series([], dtype=float)
        series = series.reindex(times).fillna(0.0)
        z.append(series.astype(float).tolist())

    if view.startswith("رخداد"):
        colorscale = [[0.0, "green"], [1.0, "red"]]
        hover_lbl = "بحران"
    else:
        colorscale = "RdYlGn_r"  # 0 green, 1 red
        hover_lbl = "ریسک"

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=times,
            y=y_labels,
            zmin=0,
            zmax=1,
            colorscale=colorscale,
            hovertemplate=f"زمان: %{{x}}<br>%{{y}}<br>{hover_lbl}: %{{z:.3f}}<extra></extra>",
        )
    )
    fig.update_layout(
        title="عامل بحران (کشورِ کنش‌گر - کشورِ هدف) در طول زمان",
        xaxis_title="گام زمانی",
        yaxis_title="زوج کشورها",
        yaxis_autorange="reversed",
        margin=dict(l=30, r=30, t=60, b=30),
        height=min(900, 120 + 22 * len(y_labels)),
    )
    st.plotly_chart(fig, use_container_width=True)


def _layout_positions_circle(names):
    n = len(names)
    pos = {}
    for i, name in enumerate(names):
        ang = 2 * math.pi * i / max(1, n)
        pos[name] = (math.cos(ang), math.sin(ang))
    return pos


def plot_interaction_graph_directed(df, countries):
    """
    گراف اقدام مدل‌محور:
    ✅ هر کشور در هر زمان Target دارد (از مدل) => پس یال همیشه رسم می‌شود
    ✅ Y فقط سبک یال را تعیین می‌کند: Y=1 پررنگ، Y=0 کم‌رنگ و خط‌چین
    ✅ خاموش/روشن فقط یال‌ها بر اساس اقدام مبدأ (P/S/R)
    ✅ زمان با تیک‌زدن ریست نمی‌شود (session_state)
    ✅ رنگ داخل کشور ثابت، حاشیه نشان‌دهنده اقدام همان لحظه
    """

    if df is None or len(df) == 0 or len(countries) < 2:
        return
    if "Time" not in df.columns:
        st.warning("ستون Time در داده‌ها وجود ندارد.")
        return

    df = df.copy()
    df["Time"] = pd.to_numeric(df["Time"], errors="coerce").fillna(0).astype(int)
    max_t = int(df["Time"].max())

    st.subheader("گراف تعاملات جهت‌دار")

    with st.expander("📌 راهنمای گراف تعاملات", expanded=False):
        st.markdown(
            """
    - هر کشور در هر زمان **Action** و **Target** دارد ⇒ پس یال همیشه وجود دارد.  
    - **Y=1** ⇒ یال پررنگ و پیوسته، **Y=0** ⇒ یال کم‌رنگ و خط‌چین.  
    - رنگ یال/حاشیه = نوع اقدام (P سبز، S نارنجی، R قرمز).  
    """
        )

    c1, c2, c3 = st.columns(3)
    with c1:
        show_edges_P = st.checkbox("نمایش یال‌های آگاهی وضعیتی (P)", value=True, key="edgeP_action", help=tip("edgeP"))
    with c2:
        show_edges_S = st.checkbox("نمایش یال‌های سیگنال (S)", value=True, key="edgeS_action", help=tip("edgeS"))
    with c3:
        show_edges_R = st.checkbox("نمایش یال‌های تقویت/زور (R)", value=True, key="edgeR_action", help=tip("edgeR"))

    visible_map = {"P": show_edges_P, "S": show_edges_S, "R": show_edges_R}
    if not (show_edges_P or show_edges_S or show_edges_R):
        st.info("هیچ دسته‌ای از یال‌ها برای نمایش انتخاب نشده است.")
        return

    if "graph_time" not in st.session_state:
        st.session_state["graph_time"] = 0

    t_selected = st.slider(
        "زمان نمایش (t)",
        min_value=0,
        max_value=max_t,
        value=int(st.session_state["graph_time"]),
        key="graph_time",
        help=tip("t_selected"),
    )

    pos = _layout_positions_circle(countries)

    act_color = {"P": "green", "S": "orange", "R": "red"}
    act_name = {"P": "آگاهی وضعیتی (P)", "S": "سیگنال (S)", "R": "تقویت/زور (R)"}

    palette = px.colors.qualitative.Set2
    country_color = {c: palette[i % len(palette)] for i, c in enumerate(countries)}

    def _get_row_safe(t: int):
        sub = df.loc[df["Time"] == int(t)]
        if not sub.empty:
            return sub.iloc[0]
        idx = (df["Time"] - int(t)).abs().idxmin()
        return df.loc[idx]

    def _shrink_segment(x0, y0, x1, y1, shrink=0.22):
        dx, dy = x1 - x0, y1 - y0
        L = (dx * dx + dy * dy) ** 0.5 + 1e-9
        x0s = x0 + dx * (shrink / L)
        y0s = y0 + dy * (shrink / L)
        x1s = x0 + dx * (1 - shrink / L)
        y1s = y0 + dy * (1 - shrink / L)
        return x0s, y0s, x1s, y1s

    def build_frame(t: int):
        row = _get_row_safe(t)

        xs, ys, texts = [], [], []
        fills, borders, hovers = [], [], []

        for c in countries:
            a = row.get(f"Action_{c}", "P")
            x, y = pos[c]
            xs.append(x);
            ys.append(y);
            texts.append(c)
            fills.append(country_color[c])
            borders.append(act_color.get(a, "white"))
            hovers.append(f"{c}<br>اقدام: {act_name.get(a, a)}")

        nodes_trace = go.Scatter(
            x=xs, y=ys,
            mode="markers+text",
            text=texts,
            textposition="bottom center",
            marker=dict(size=26, color=fills, line=dict(width=4, color=borders)),
            hovertext=hovers,
            hoverinfo="text",
            showlegend=False,
            name="کشورها",
        )

        seg_on = {"P": [], "S": [], "R": []}
        seg_off = {"P": [], "S": [], "R": []}
        anns = []

        for src in countries:
            a = row.get(f"Action_{src}", "P")
            if a not in ("P", "S", "R"):
                a = "P"

            tgt = row.get(f"Target_{src}", None)
            # ✅ اگر target نبود، برای اینکه همیشه یال داشته باشیم:
            # یک هدف جایگزین انتخاب می‌کنیم (کشور بعدی در لیست)
            if tgt is None or tgt not in countries or tgt == src:
                idx_src = countries.index(src)
                tgt = countries[(idx_src + 1) % len(countries)]

            y = int(row.get(f"Y_{src}_{tgt}", 0))

            x0, y0 = pos[src]
            x1, y1 = pos[tgt]
            x0s, y0s, x1s, y1s = _shrink_segment(x0, y0, x1, y1)

            if y == 1:
                seg_on[a].append((x0s, y0s, x1s, y1s))
            else:
                seg_off[a].append((x0s, y0s, x1s, y1s))

            anns.append(
                dict(
                    x=x1s, y=y1s, ax=x0s, ay=y0s,
                    xref="x", yref="y", axref="x", ayref="y",
                    showarrow=True,
                    arrowhead=3,
                    arrowsize=1.15,
                    arrowwidth=2,
                    arrowcolor=act_color[a],
                    opacity=0.9 if y == 1 else 0.25,
                )
            )

        traces = []
        for act in ["P", "S", "R"]:
            x_list, y_list = [], []
            for (x0s, y0s, x1s, y1s) in seg_off[act]:
                x_list += [x0s, x1s, None]
                y_list += [y0s, y1s, None]
            traces.append(
                go.Scatter(
                    x=x_list, y=y_list,
                    mode="lines",
                    line=dict(width=2, color=act_color[act], dash="dot"),
                    hoverinfo="skip",
                    showlegend=False,
                    opacity=0.25,
                    visible=True if visible_map[act] else "legendonly",
                )
            )

            x_list2, y_list2 = [], []
            for (x0s, y0s, x1s, y1s) in seg_on[act]:
                x_list2 += [x0s, x1s, None]
                y_list2 += [y0s, y1s, None]
            traces.append(
                go.Scatter(
                    x=x_list2, y=y_list2,
                    mode="lines",
                    line=dict(width=4, color=act_color[act]),
                    name=f"یال‌های {act_name[act]}",
                    hoverinfo="skip",
                    showlegend=True,
                    opacity=0.95,
                    visible=True if visible_map[act] else "legendonly",
                )
            )

        return (traces + [nodes_trace]), anns

    frames = []
    for t in range(max_t + 1):
        d, ann = build_frame(t)
        frames.append(go.Frame(data=d, name=str(t), layout=dict(annotations=ann)))

    base_data, base_ann = build_frame(t_selected)

    fig = go.Figure(data=base_data, frames=frames)
    fig.update_layout(
        title=f"گراف اقدام — زمان {t_selected}",
        xaxis=dict(visible=False, range=[-1.4, 1.4]),
        yaxis=dict(visible=False, range=[-1.3, 1.3]),
        margin=dict(l=20, r=20, t=60, b=150),
        annotations=base_ann,
        legend=dict(
            title="یال‌ها بر اساس اقدام مبدأ",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=10.0,
        ),
        uirevision="action_graph_keep_state_v1",
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                x=0.5,
                y=-0.20,
                xanchor="center",
                yanchor="top",
                buttons=[
                    dict(
                        label="▶ پخش",
                        method="animate",
                        args=[None, dict(
                            frame=dict(duration=450, redraw=True),
                            transition=dict(duration=150),
                            fromcurrent=True,
                            mode="immediate"
                        )],
                    ),
                    dict(
                        label="⏸ توقف",
                        method="animate",
                        args=[[None], dict(
                            frame=dict(duration=0, redraw=False),
                            transition=dict(duration=0),
                            mode="immediate"
                        )],
                    ),
                ],
            )
        ],
    )

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
    scenario_titles = ["سفارشی (دستی)"] + [scenarios[k]["title"] for k in scenarios.keys()]
    key_to_title = dict(zip(scenario_keys, scenario_titles))

    chosen = st.sidebar.radio(
        "انتخاب سناریو آماده",
        options=scenario_keys,
        format_func=lambda k: key_to_title[k],
        index=1,
        help=tip("scenario"),
        key="scenario_choice",
    )
    _reset_on_scenario_change(chosen)

    # --- Reset all inputs/outputs when switching scenarios ---
    if "last_scenario" not in st.session_state:
        st.session_state["last_scenario"] = chosen
    elif st.session_state["last_scenario"] != chosen:
        # Keep only the scenario selector state
        _keep = {"scenario_choice", "last_scenario"}
        for _k in list(st.session_state.keys()):
            if _k not in _keep:
                del st.session_state[_k]
        st.session_state["last_scenario"] = chosen
        st.rerun()

    # -----------------------------
    # ✅ اگر سناریو عوض شد، state ورودی‌ها را ریست کن تا نام کشورها/فیلدها درست آپدیت شوند
    # (Streamlit مقدار ویجت‌ها را با key نگه می‌دارد؛ پس باید کلیدهای قبلی پاک شوند)
    if "last_scenario" not in st.session_state:
        st.session_state["last_scenario"] = chosen
    elif st.session_state["last_scenario"] != chosen:
        prefixes = [
            "custom_name_",
            "res0_", "income_", "v_", "chi_",
            "rho_", "d_", "f_",
            "wsec_", "winf_", "wcost_",
            "lambdaop_", "tau_", "eps_",
            "eta_", "kappa_", "pa_", "pb_", "ra_", "rb_",
            "beta_", "prefP_", "prefS_", "prefR_",
        ]
        explicit = {
            "custom_n",
            "custom_country_names",
            "custom_agents",
            "custom_W",
            "W_editor",
            "sim_df",
            "sim_meta",
            "has_run",
        }
        for k in list(st.session_state.keys()):
            if (k in explicit) or any(k.startswith(p) for p in prefixes):
                try:
                    del st.session_state[k]
                except Exception:
                    pass
        st.session_state["last_scenario"] = chosen
        st.rerun()

    st.sidebar.divider()
    st.sidebar.header("⚙️ تنظیمات اجرا")

    # ✅ ورودی تغییر دکترین
    doctrine_update_every = st.sidebar.number_input(
        "تغییر دکترین بعد از چند بار انجام یک اقدام؟",
        min_value=0,
        max_value=200,
        value=0,
        step=5,
        help=tip("doctrine_update_every"),
    )

    test_mode = st.sidebar.toggle("حالت تست (Test Mode)", value=False, help=tip("test_mode"))
    seed = None
    if test_mode:
        seed = st.sidebar.number_input("عدد بذر تصادفی (Seed)", min_value=0, max_value=10_000_000, value=42, step=1,
                                       help=tip("seed"))

    if chosen != "custom":
        default_steps = scenarios[chosen]["steps_default"]
    else:
        default_steps = 70

    steps = st.sidebar.number_input("تعداد گام‌های زمانی", min_value=10, max_value=200, value=int(default_steps),
                                    step=5, help=tip("steps"))
    run_btn = st.sidebar.button("🚀 اجرای شبیه‌سازی", type="primary", use_container_width=True, help=tip("run_btn"))

    # انتخاب سناریو/سفارشی
    if chosen != "custom":
        sc = scenarios[chosen]
        st.sidebar.divider()
        st.sidebar.subheader("📘 توضیح سناریو")
        st.sidebar.markdown(sc["story"])

        # ✅ سناریوهای آماده هم مثل حالت سفارشی قابل مشاهده/ویرایش هستند
        agent_cfgs, W, countries = build_custom_ui(
            prefill_agents=sc["agents"],
            prefill_W=sc["W"],
            prefill_countries=sc["countries"],
            lock_n=True,
        )
    else:
        st.sidebar.info("حالت سفارشی فعال است: پارامترها را تنظیم کنید.")
        agent_cfgs, W, countries = build_custom_ui()

    # --- session state ---
    if "sim_df" not in st.session_state:
        st.session_state.sim_df = None
    if "sim_meta" not in st.session_state:
        st.session_state.sim_meta = None
    if "has_run" not in st.session_state:
        st.session_state.has_run = False

    # ✅ فقط وقتی Run زده شد اجرا کن
    if run_btn:
        with st.spinner("در حال اجرای شبیه‌سازی..."):
            df, meta = run_simulation(
                agent_cfgs=agent_cfgs,
                W=W,
                steps=int(steps),
                test_mode=test_mode,
                seed=seed,
                doctrine_update_every=int(doctrine_update_every),
            )
            st.session_state.sim_df = df
            st.session_state.sim_meta = meta
            st.session_state.has_run = True

    # ✅ تا قبل از اجرا هیچ خروجی‌ای نشان نده
    if not st.session_state.has_run:
        st.stop()

    df = st.session_state.sim_df
    meta = st.session_state.sim_meta

    if df is None or df.empty:
        st.warning("هیچ داده‌ای برای نمایش وجود ندارد.")
        return

    st.divider()
    st.subheader("خلاصه اقدامات")
    st.dataframe(df_action_counts(df, countries), use_container_width=True)
    st.divider()
    plot_three_indices_heatmaps(df, countries, window=10)

    st.divider()
    plot_global_escalation(df)

    st.divider()
    plot_dyad_crisis_heatmap(df, countries)

    st.divider()
    plot_lines_by_country(df, countries, prefix="Tension", title_fa="روند تنش کشورها (Tension)",
                          y_label_fa="تنش (Tension)")

    st.divider()
    plot_lines_by_country(df, countries, prefix="Resource", title_fa="روند منابع کشورها (Resources)",
                          y_label_fa="منابع (Resources)")

    st.divider()
    plot_lines_by_country(df, countries, prefix="Psi", title_fa="خروجی تشدید کشور (ψ_c)", y_label_fa="ψ_c")

    st.divider()
    plot_actions_map(df, countries)

    st.divider()
    plot_dyad_tension_heatmap(df, countries)

    st.divider()
    plot_interaction_graph_directed(df, countries)

    # -----------------------------
    # ✅ جدول تغییر پارامترها + فیلتر بخش
    # -----------------------------
    st.divider()
    st.subheader("تغییر پارامترها (ابتدا → انتها)")
    trans_df = build_transition_df(meta, countries)

    if trans_df is None or trans_df.empty:
        st.info("جدول تغییر پارامترها خالی است (meta کامل تولید نشده).")
    else:
        country_options = sorted(trans_df["کشور"].dropna().unique().tolist())
        country_filter = st.multiselect(
            "فیلتر کشورها",
            options=country_options,
            default=country_options,
            key="filter_countries_params",
            help=tip("country_filter"),
        )

        section_options = SECTION_ORDER
        existing_sections = sorted([s for s in trans_df["بخش"].dropna().unique().tolist() if s in section_options])
        if not existing_sections:
            existing_sections = section_options

        section_filter = st.multiselect(
            "فیلتر بخش‌ها",
            options=section_options,
            default=existing_sections,
            key="filter_sections_params",
            help="می‌تونی مشخص کنی فقط دکترین یا فقط تکنیک و ... نمایش داده شود.",
        )

        show_df = trans_df.copy()
        if country_filter:
            show_df = show_df[show_df["کشور"].isin(country_filter)]
        if section_filter:
            show_df = show_df[show_df["بخش"].isin(section_filter)]

        show_df = show_df.copy()
        show_df["Δ (تغییر)"] = show_df["انتها"] - show_df["ابتدا"]

        show_df["ترتیب_بخش"] = show_df["بخش"].map({v: i for i, v in enumerate(section_options)}).fillna(999).astype(int)
        show_df = show_df.sort_values(["ترتیب_بخش", "کشور", "پارامتر"]).drop(columns=["ترتیب_بخش"])

        st.dataframe(show_df, use_container_width=True)

        csv_bytes = show_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "⬇️ دانلود جدول تغییرات (CSV)",
            data=csv_bytes,
            file_name="parameter_transitions.csv",
            mime="text/csv",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()