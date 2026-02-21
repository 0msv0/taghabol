# model2.py
# -------------------------------------------------------------------
# این فایل «موتور محاسباتی» مدل است (بدون UI).
# هدف: پیاده‌سازی عامل سلسله‌مراتبی + جهان چندعاملی + تعاملات جهت‌دار
# مطابق فرمول‌های کتابچه (Utility/Logit/ψ_c/ψ_ij/State Dynamics) و
# چند اصلاح مهندسی برای پایداری/واقع‌گرایی خروجی‌ها.
# -------------------------------------------------------------------

import numpy as np
from dataclasses import dataclass, field


# ==========================================================
# 0) Helpers + constants
# ==========================================================
# این بخش ابزارهای پایه‌ای است که در همه جای مدل استفاده می‌شوند.

def sigmoid(x: float) -> float:
    # تابع سیگموید σ(x) که در کتابچه بارها استفاده شده:
    # - تبدیل ترکیب خطی به احتمال/شدت بین 0 و 1
    # - نمونه‌ها: ψ_c(t)=σ(...) ، ψ_ij(t)=σ(...) ، tension_{t+1}=σ(...)
    return 1.0 / (1.0 + np.exp(-x))


def fit_logistic_map(X, y, w0=None, l2=1.0, lr=0.1, iters=200):
    """Fit logistic regression via MAP (Bernoulli-Logit likelihood + Normal(0,1/l2) prior).

    This implements the booklet's idea:
        Posterior ∝ Likelihood(Y | θ) × Prior(θ)
    where Likelihood is Bernoulli-Logit and Prior is Normal ⇒ L2 penalty in MAP.

    Parameters
    ----------
    X : array-like, shape (n, d)
    y : array-like, shape (n,)
    w0 : optional initial weights, shape (d,)
    l2 : float, L2 strength (acts like 1/σ² for a zero-mean normal prior)
    lr : float, learning rate
    iters : int, number of gradient steps
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    if X.ndim != 2:
        raise ValueError("X must be 2D (n,d)")
    if y.ndim != 1 or y.shape[0] != X.shape[0]:
        raise ValueError("y must be 1D with length n")

    n, d = X.shape
    w = np.zeros(d, dtype=float) if w0 is None else np.asarray(w0, dtype=float).copy()

    # simple batch gradient ascent on log-posterior
    for _ in range(int(iters)):
        z = X @ w
        p = 1.0 / (1.0 + np.exp(-z))
        grad = X.T @ (y - p) - l2 * w
        w += (lr / max(1.0, float(n))) * grad

    return w
    # فرمول استاندارد: 1/(1+e^-x)


ACTIONS = ["Patrol (P)", "Signal (S)", "Reinforce (R)"]
# تعریف «فضای اقدام» (Action Space) با سه اقدام:
# P: گشت‌زنی، S: سیگنال/مانور، R: تقویت/زور
# این ایده در کتابچه در بخش معرفی فضای اقدام (مثلاً حوالی صفحه 10) آمده.

ACT_MAP = {0: "P", 1: "S", 2: "R"}


# در محاسبات راحت‌تر است با اندیس 0/1/2 کار کنیم،
# اما در خروجی‌ها خواناتر است با P/S/R نشان دهیم؛ این نگاشت همین کار را می‌کند.

# ==========================================================
# 1) Coefficients (booklet-style) with sane scales
# ==========================================================
# اینجا ضرایب جهانیِ فرمول‌های کتابچه را نگه می‌داریم.
# «جهانی» یعنی برای همه کشورها مشترک‌اند (مگر بعداً بخواهی کشورمحورشان کنی).

@dataclass
class EscalationCoeffs:
    """
    ψ_c(t)=σ(αS^T S + αO^T O + αT^T T + δ^T Z)

    نکته مهم:
    - T (ریسک/اتلاف/هزینه یادگیری) باید احتمال تشدید را کم کند
      پس alpha_T باید منفی باشد (ریسک ↑ => ψ ↓)
    - برای جلوگیری از اشباع، یک bias/scale هم می‌گذاریم.
    """
    # این کلاس دقیقاً برای فرمول ψ_c(t) است (کتابچه: بخش احتمال تشدید/صفحه 17).
    # ایده: احتمال اینکه «یک کشور» در لحظه t وارد وضعیت تشدید شود.

    alpha_S: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.8, -1.2], dtype=float))  # [sec, inf, cost]
    # αS ضرایب بخش راهبردی S در ψ_c:
    # - sec: امنیت، inf: نفوذ، cost: هزینه
    # اصلاح نسبت به نسخه ساده: cost باید اثر منفی داشته باشد ⇒ ضریب cost منفی گذاشته شده
    # تا هزینه بیشتر ⇒ ψ کمتر (منطقی و مطابق نیت مدل).

    alpha_O: np.ndarray = field(
        default_factory=lambda: np.array([0.7, 0.6, 0.6], dtype=float))  # [alloc, tempo, mobilize]
    # αO ضرایب بخش عملیاتی O در ψ_c:
    # alloc/tempo/mobilize که در gC تولید می‌شوند.
    # این ضرایب کمی کوچکتر از 1 انتخاب شدند تا ψ خیلی سریع به 0/1 اشباع نشود.

    alpha_T: np.ndarray = field(
        default_factory=lambda: np.array([-1.0, -1.0, -0.8], dtype=float))  # [eff_loss, fail_risk, learning_cost]
    # αT ضرایب بخش فنی/ریسکی T در ψ_c (کتابچه: نگاشت فنی/صفحه 13 + نقش آن در ψ/صفحه 17).
    # چون T شامل «اتلاف/ریسک شکست/هزینه یادگیری» است، افزایشش باید تشدید را کم کند ⇒ ضرایب منفی.

    delta: np.ndarray = field(default_factory=lambda: np.array([0.6, 0.3], dtype=float))  # [v_c, resource_norm]
    # δ ضرایب بردار زمینه‌ای Z:
    # - v_c: آسیب‌پذیری مرز (هرچه بالاتر ⇒ فشار بیشتر ⇒ احتمال تشدید بالاتر)
    # - resource_norm: منابع نرمال‌شده (کشور تواناتر ⇒ ظرفیت تشدید بیشتر)
    # ضرایب مثبت اما کنترل‌شده انتخاب شده‌اند.

    # ψ_ij(t)=σ(η1 ψ_i + η2 ψ_j + η3 ψ_i ψ_j)
    # این فرمول در کتابچه برای «تشدید دوتایی»/تعامل بین دو کشور آمده (کتابچه: صفحه 17).
    eta1: float = 0.85
    # η1 وزن اثر کشور i روی ψ_ij (یعنی اگر i خودش مستعد تشدید باشد، رابطه i→j هم مستعدتر می‌شود).

    eta2: float = 0.85
    # η2 وزن اثر کشور j (هدف) روی ψ_ij (یعنی اگر j هم مستعد تشدید باشد، تعامل حساس‌تر می‌شود).

    eta3: float = 0.25
    # η3 وزن اثر هم‌افزایی (ψ_i ψ_j):
    # اگر هر دو کشور همزمان مستعد باشند، احتمال تشدید رابطه بیش از جمع ساده می‌شود.

    # calibration to avoid saturation

    eta_bias: float = 0.0
    # بایاس ثابت برای ψ_ij (داخل سیگموید)؛ در نسخه بیزی/MAP قابل یادگیری است.

    eta_W: float = 0.6
    # ضریب اثر وزن تعامل W_ij در ψ_ij (همان چیزی که در کد قبلاً به شکل ثابت اضافه می‌شد).
    psi_bias: float = 1.8
    # این پارامتر در کتابچه صراحتاً نیست.
    # دلیل اضافه‌شدن: در عمل ممکن است αS·S + αO·O + αT·T + δ·Z
    # عدد بزرگی شود و سیگموید خیلی سریع نزدیک 1 بچسبد ⇒ همه چیز دائماً جنگی/یکنواخت.
    # با bias، ورودی سیگموید را «جا‌به‌جا» می‌کنیم تا در محدوده میانی بماند.

    psi_scale: float = 0.75
    # این هم در کتابچه نیست و برای «مقیاس کردن» ورودی ψ اضافه شده
    # تا دامنه ورودی سیگموید کنترل شود و تنوع خروجی‌ها بیشتر شود.


@dataclass
class StateDynamicsCoeffs:
    """
    tension_{t+1}=σ(α0 + αv v_c + αψ ψ_c + αa E[U] - αr resource_norm)

    نکته:
    - به جای کم کردن resource خام، resource_norm استفاده می‌کنیم تا مقیاس‌ها نخوابند.
    """
    # این کلاس ضرایب دینامیک «تنش داخلی» را نگه می‌دارد.
    # فرمول tension_{t+1}=σ(...) در کتابچه در بخش «State Dynamics / latent state update»
    # آمده (معمولاً حوالی صفحه 15 تا 21 که درباره تنش/حالت نهفته صحبت می‌کند).

    alpha0: float = -0.35
    # α0 بایاس پایه: منفی گذاشته شده تا در حالت خنثی، تنش به طور طبیعی کم باشد.

    alpha_v: float = 1.1
    # αv وزن آسیب‌پذیری مرز v_c: مرز آسیب‌پذیرتر ⇒ تنش بیشتر (منطقی و مطابق مدل کتابچه).

    alpha_psi: float = 0.9
    # αψ وزن ψ_c: اگر کشور مستعد تشدید باشد، تنش هم بالا می‌رود.

    alpha_a: float = 0.35
    # αa وزن مطلوبیت مورد انتظار E[U] (کتابچه: E[U] در دینامیک حالت آمده).
    # اگر تصمیم‌های کشور «به سمت سود بالاتر» یا «حالات افراطی» برود، تنش هم تغییر می‌کند.

    alpha_r: float = 0.9  # روی resource_norm اعمال می‌شود (نه resource خام)
    # αr وزن منابع، با علامت منفی در فرمول.
    # تغییر مهندسی نسبت به نسخه خام:
    # - به جای کم کردن resource خام (که ممکن است 1000+ باشد) از resource_norm استفاده می‌کنیم
    # تا مقیاس درست شود و مجبور به ضرایب خیلی ریز/خیلی بزرگ نشویم.

    lambda_v: float = 2.0  # tension_0 = σ(lambda_v * v_c)
    # مقداردهی اولیه تنش طبق یک نگاشت ساده از v_c.
    # اگر کشور از ابتدا مرزش آسیب‌پذیرتر باشد ⇒ تنش اولیه بالاتر.


# ==========================================================
# 2) Action feature bases
# ==========================================================
# این کلاس «پایه ویژگی‌ها» برای اقدامات است:
# یعنی هر اقدام P/S/R چه اثر پایه‌ای روی امنیت/نفوذ/هزینه و ریسک‌ها دارد.
# این‌ها همان gS/gC/gR را تغذیه می‌کنند (کتابچه: صفحات 12 و 13).

@dataclass
class ActionBases:
    """
    sec_gain / inf_gain / cost: پایه‌های راهبردی
    eff_loss / fail_risk / learning_cost: پایه‌های فنی (ریسک/هزینه)
    """
    # این docstring می‌گوید این کلاس چه چیزهایی را نگه می‌دارد:
    # - بخش راهبردی: sec_gain, inf_gain, cost (کتابچه: نگاشت راهبردی/صفحه 12)
    # - بخش فنی: eff_loss, fail_risk, learning_cost (کتابچه: نگاشت فنی/صفحه 13)

    sec_gain: dict = None
    # دیکشنری: sec_gain[action] مقدار پایه‌ی «بهبود امنیت» برای آن اقدام.

    inf_gain: dict = None
    # دیکشنری: inf_gain[action] مقدار پایه‌ی «کسب نفوذ» برای آن اقدام.

    cost: dict = None
    # دیکشنری: cost[action] هزینه پایه برای آن اقدام.

    eff_loss: dict = None
    # دیکشنری: eff_loss[action] اتلاف کارایی (هزینه فنی/عملیاتی) برای آن اقدام.

    fail_risk: dict = None
    # دیکشنری: fail_risk[action] ریسک شکست پایه برای آن اقدام.

    learning_cost: dict = None
    # دیکشنری: learning_cost[action] هزینه یادگیری/پیچیدگی فنی برای آن اقدام.

    gamma_e: float = 10.0  # γ_e در σ(γ_e(ε - tension))

    # γ_e ضریب تیزکننده برای بسیج mobilize در gC.
    # در کتابچه mobilize معمولاً یک سیگموید از (ε - tension) است (نگاشت عملیاتی/صفحه 12).

    def __post_init__(self):
        # __post_init__ بعد از ساخته شدن dataclass اجرا می‌شود.
        # کارش: اگر کاربر هیچ مقدار نداده باشد، مقادیر پیش‌فرض منطقی ست کند.

        if self.sec_gain is None:
            # اگر پایه امنیت تعریف نشده بود:
            self.sec_gain = {0: 0.85, 1: 0.50, 2: 0.95}
            # معنی:
            # - P امنیت خوب اما نه حداکثر (0.85)
            # - S امنیت متوسط (0.50)
            # - R امنیت بسیار بالا (0.95) چون زور/تقویت امنیت فوری می‌دهد.

        if self.inf_gain is None:
            # اگر پایه نفوذ تعریف نشده بود:
            self.inf_gain = {0: 0.25, 1: 0.85, 2: 0.90}  # S نفوذ/سیگنال را بالا دارد
            # معنی:
            # - S (سیگنال) برای نفوذ بسیار مهم است ⇒ 0.85
            # - R هم نفوذ ایجاد می‌کند (ترس/قدرت) ⇒ 0.90
            # - P نفوذ کمتر ⇒ 0.25

        if self.cost is None:
            # اگر پایه هزینه تعریف نشده بود:
            self.cost = {0: 0.25, 1: 0.12, 2: 0.90}  # R پرهزینه‌تر است
            # معنی:
            # - R شدیداً پرهزینه ⇒ 0.90
            # - P هزینه متوسط ⇒ 0.25
            # - S ارزان‌تر ⇒ 0.12

        if self.eff_loss is None:
            # اگر اتلاف کارایی تعریف نشده بود:
            self.eff_loss = {0: 0.14, 1: 0.10, 2: 0.36}
            # R اتلاف بیشتری دارد چون عملیات سنگین‌تر است.

        if self.fail_risk is None:
            # اگر ریسک شکست تعریف نشده بود:
            self.fail_risk = {0: 0.20, 1: 0.22, 2: 0.55}
            # R سخت‌تر و پرریسک‌تر ⇒ ریسک شکست پایه بالاتر.

        if self.learning_cost is None:
            # اگر هزینه یادگیری تعریف نشده بود:
            self.learning_cost = {0: 0.05, 1: 0.07, 2: 0.18}
            # R پیچیدگی/آموزش/هماهنگی بیشتر ⇒ هزینه یادگیری بالاتر.


# ==========================================================
# 3) Agent (clean separation: income vs operational allocation)
# ==========================================================
# این کلاس «مغز هر کشور» است:
# - با دیدن وضعیت خودش و ضرایبش، utility هر اقدام را حساب می‌کند
# - سپس با قانون لاجیت (MNL) اقدام را انتخاب می‌کند
# - ψ_c را می‌سازد (احتمال تشدید کشور)
# - یادگیری و تغییرات حالت (تنش/منابع) را انجام می‌دهد
# مطابق ساختار سلسله‌مراتبی کتابچه.

class HierarchicalAgent:
    """
    U = ωS·gS + ωC·gC − ωR·gR
    انتخاب: MNL با β_c و ω_{a,c}

    اصلاحات منطقی:
    1) income_c (μ_c) برای افزایش منابع جدا شد.
    2) lambda_op فقط برای gC (توان تخصیص/عملیات) استفاده می‌شود.
    3) شمارنده اقدامات داریم تا "هر N بار انجام یک اقدام" دکترین تغییر کند.
    """

    # این docstring خلاصه فرمول‌های کلیدی کتابچه را می‌گوید:
    # - Utility: کتابچه بخش تابع مطلوبیت (معمولاً صفحه 11 تا 13)
    # - MNL/Logit choice: کتابچه بخش انتخاب اقدام (معمولاً صفحه 14-15)
    # و همچنین اصلاحات مهندسی‌ای که اضافه کردیم را فهرست می‌کند.

    def __init__(
            self,
            name: str,
            # name نام کشور/عامل است و در گزارش‌ها و ستون‌های دیتا استفاده می‌شود.

            # state
            initial_resource: float,
            # منابع اولیه کشور (Resources) در زمان شروع شبیه‌سازی.

            v_c: float,  # border vulnerability ∈ [0,1]
            # v_c آسیب‌پذیری مرزی: پارامتر زمینه‌ای که روی تنش و ψ اثر دارد.

            # doctrine
            rho_c: float,
            # rho_c تحمل ریسک/ریسک‌پذیری (کتابچه سطح دکترین/صفحه 10).

            d_c: float,
            # d_c ترجیح بازدارندگی/نفوذ (کتابچه سطح دکترین/صفحه 10).

            f_c: float,
            # f_c آستانه استفاده از زور (کتابچه سطح دکترین/صفحه 10).

            chi_c: float,
            # chi_c شدت «خرج کردن منابع» برای اقدام‌ها.
            # این در کتابچه ممکن است صراحتاً با همین نماد نیاید،
            # اما به عنوان پارامتر هزینه/مصرف منابع برای واقعی‌سازی dynamics منابع اضافه شده.

            # strategic weights ω_S = [sec, inf, cost]
            omega_S: np.ndarray,
            # ω_S وزن‌های راهبردی روی امنیت/نفوذ/هزینه (کتابچه: وزن‌دهی راهبردی/صفحه 12).

            # operational parameters (clean)
            lambda_op: float,  # λ_op,c  (ONLY in gC)
            # λ_op فقط در gC استفاده می‌شود (توان تخصیص عملیات).
            # نکته: این λ_op را از «درآمد» جدا کرده‌ایم تا منابع بی‌دلیل رشد نکند.

            tau_c: float,
            # τ_c پارامتر tempo در gC (کتابچه: نگاشت عملیاتی/صفحه 12).

            eps_c: float,
            # ε_c آستانه بسیج (mobilization threshold) در gC (کتابچه: نگاشت عملیاتی/صفحه 12).

            # economic income (new clean parameter)
            income_c: float,  # μ_c   (ONLY in resource update)
            # μ_c درآمد/ورودی منابع در هر گام زمانی (اصلاح مهندسی).
            # هدف: جدا کردن «درآمد طبیعی» از «تخصیص عملیاتی» (حل مشکل رشد عجیب منابع).

            # technical
            eta_c: float,
            # η_c توان فنی (هرچه بالاتر، اتلاف/ریسک کمتر) (کتابچه: نگاشت فنی/صفحه 13).

            p_params: list,  # [alpha,beta]
            # پارامترهای بتا برای p (احتمال موفقیت) (کتابچه: Beta prior/پیوست/صفحات مربوط).

            r_params: list,  # [alpha,beta]
            # پارامترهای بتا برای r (قابلیت اطمینان/پایداری) (کتابچه: Beta prior/پیوست).

            kappa_c: float,
            # κ_c ضریب هزینه یادگیری (کتابچه: نگاشت فنی/صفحه 13).

            # tactical
            beta_c: float,
            # β_c پارامتر عقلانیت/حساسیت به سود در قانون لاجیت (کتابچه: صفحه 14).

            omega_a: np.ndarray,  # [prefP, prefS, prefR]
            # ω_{a,c} ترجیحات تاکتیکی برای P/S/R (کتابچه: بخش ترجیحات تاکتیکی/عادت‌ها).

            # shared configs
            action_bases: ActionBases,
            # شیء ActionBases که پایه ویژگی‌های اقدامات را می‌دهد.

            dyn_coeffs: StateDynamicsCoeffs,
            # ضرایب دینامیک تنش (State dynamics coefficients).
    ):
        self.name = name
        # ذخیره نام کشور برای استفاده در ستون‌های دیتا و گزارش‌ها.

        self.action_bases = action_bases
        # اتصال پایگاه ویژگی‌های اقدام به عامل، تا gS/gC/gR بتوانند از آن بخوانند.

        self.dyn = dyn_coeffs
        # اتصال ضرایب دینامیک حالت به عامل، تا update_state بتواند از آن استفاده کند.

        # ---------- state ----------
        self.resource = float(initial_resource)
        # مقدار اولیه منابع کشور (حالت فیزیکی/اقتصادی).

        self.v_c = float(v_c)
        # ذخیره آسیب‌پذیری مرز.

        self.tension = sigmoid(self.dyn.lambda_v * self.v_c)
        # مقداردهی اولیه تنش:
        # tension_0 = σ(lambda_v * v_c)
        # کتابچه: ایده «تنش اولیه تابعی از شرایط زمینه‌ای مثل آسیب‌پذیری» (حوالی صفحه 21).
        # دلیل انتخاب سیگموید: خروجی نرمال بین 0 و 1.

        # ---------- doctrine ----------
        self.rho_c = float(rho_c)
        # ذخیره ریسک‌پذیری کشور (دکترین).

        self.d_c = float(d_c)
        # ذخیره پارامتر بازدارندگی/نفوذ (دکترین).

        self.f_c = float(f_c)
        # ذخیره آستانه زور (دکترین).

        self.chi_c = float(chi_c)
        # ذخیره شدت خرج کردن/مصرف منابع (برای منابع واقع‌گرایانه‌تر).

        # ---------- strategic weights ----------
        self.omega_S = np.array(omega_S, dtype=float)
        # تبدیل ورودی وزن‌ها به آرایه عددی قابل محاسبه.

        self.omega_S = self.omega_S / (self.omega_S.sum() + 1e-12)
        # نرمال‌سازی ω_S تا جمعشان 1 شود:
        # این باعث می‌شود وزن‌دهی قابل مقایسه باشد و scale بهم نریزد.

        # ---------- operational (clean) ----------
        self.lambda_op = float(lambda_op)
        # ذخیره λ_op برای gC.

        self.tau_c = float(tau_c)
        # ذخیره τ برای gC.

        self.eps_c = float(eps_c)
        # ذخیره ε برای gC.

        # ---------- economic income ----------
        self.income_c = float(income_c)
        # ذخیره درآمد μ_c برای آپدیت منابع.

        # ---------- technical ----------
        self.p_ab = [float(p_params[0]), float(p_params[1])]
        # ذخیره پارامترهای بتا برای p در قالب [α, β].

        self.r_ab = [float(r_params[0]), float(r_params[1])]
        # ذخیره پارامترهای بتا برای r در قالب [α, β].

        self.eta_c = float(eta_c)
        # ذخیره توان فنی η_c.

        self.kappa_c = float(kappa_c)
        # ذخیره κ_c برای هزینه یادگیری.

        # ---------- tactical ----------
        self.beta_c = float(beta_c)
        # ذخیره β_c (حساسیت انتخاب لاجیت).

        self.omega_a = np.array(omega_a, dtype=float)
        # تبدیل ترجیحات تاکتیکی به آرایه.

        self.omega_a = self.omega_a / (self.omega_a.sum() + 1e-12)
        # نرمال‌سازی ω_a تا جمعشان 1 شود:
        # این کار باعث می‌شود ω_a مثل «یک توزیع ترجیح» رفتار کند.

        # if not provided in UI: keep equal weights for ωC and ωR
        self.omega_C = np.array([1.0, 1.0, 1.0], dtype=float)
        # ω_C وزن‌های نگاشت عملیاتی gC در utility.
        # در کتابچه ω_C برداری است؛ چون در UI نداده‌ایم، برابر فرض می‌کنیم.

        self.omega_R = np.array([1.0, 1.0, 1.0], dtype=float)
        # ω_R وزن‌های نگاشت فنی gR در utility (با علامت منفی).
        # در UI نداده‌ایم، برابر فرض می‌کنیم تا مدل کامل باشد.

        # ---- action counters for doctrine update ----
        self.action_counts = np.zeros(3, dtype=int)
        # شمارنده تعداد دفعاتی که هر اقدام انجام شده:
        # action_counts[0]=تعداد P ، [1]=تعداد S ، [2]=تعداد R
        # این جزء کتابچه نیست؛ برای نیاز تو اضافه شد تا «هر N بار» دکترین تغییر کند.

    # ---------- snapshots (برای "ابتدا→انتها") ----------
    def snapshot(self) -> dict:
        # این تابع یک «عکس لحظه‌ای» از پارامترهای مهم عامل می‌گیرد
        # تا بتوانیم در UI تغییرات ابتدا→انتها را نشان دهیم.

        return dict(
            # doctrine
            rho_c=self.rho_c,
            # ثبت ریسک‌پذیری برای مقایسه بعدی.

            d_c=self.d_c,
            # ثبت دکترین بازدارندگی/نفوذ.

            f_c=self.f_c,
            # ثبت آستانه زور.

            chi_c=self.chi_c,
            # ثبت شدت هزینه‌کرد.

            # strategic weights
            omega_S=self.omega_S.copy(),
            # ثبت وزن‌های راهبردی (کپی برای اینکه بعداً تغییر کرد روی snapshot اثر نگذارد).

            # operational
            lambda_op=self.lambda_op,
            # ثبت λ_op عملیاتی.

            tau_c=self.tau_c,
            # ثبت τ عملیاتی.

            eps_c=self.eps_c,
            # ثبت ε عملیاتی.

            # economic
            income_c=self.income_c,
            # ثبت درآمد.

            # technical (means)
            p_c=self.p_c,
            # ثبت میانگین فعلی p (از بتا).

            r_c=self.r_c,
            # ثبت میانگین فعلی r (از بتا).

            eta_c=self.eta_c,
            # ثبت توان فنی.

            kappa_c=self.kappa_c,
            # ثبت هزینه یادگیری.

            # tactical
            beta_c=self.beta_c,
            # ثبت β انتخاب.

            omega_a=self.omega_a.copy(),
            # ثبت ترجیحات تاکتیکی.

            # state
            tension=self.tension,
            # ثبت تنش فعلی.

            resource=self.resource,
            # ثبت منابع فعلی.

            v_c=self.v_c,
            # ثبت آسیب‌پذیری مرز.

        )
        # خروجی یک dict است که راحت در pandas/streamlit نمایش داده می‌شود.

    # ---------- Beta means ----------
    @staticmethod
    def _mean_beta(ab):
        # این تابع میانگین توزیع Beta(α,β) را می‌دهد:
        # E[x] = α / (α + β)
        # این دقیقاً در کتابچه در بخش Beta update/پیوست آمده.

        a, b = ab
        # جدا کردن α و β از لیست/تاپل ورودی.

        return float(a / (a + b))
        # میانگین بتا را به float تبدیل می‌کنیم برای راحتی.

    @property
    def p_c(self) -> float:
        # p_c احتمال موفقیت (mean of Beta) برای کشور.
        # کتابچه: پارامتر موفقیت فنی با بتا مدل می‌شود.

        return self._mean_beta(self.p_ab)
        # میانگین بتای p از پارامترهای p_ab

    @property
    def r_c(self) -> float:
        # r_c قابلیت اطمینان/پایداری (mean of Beta) برای کشور.

        return self._mean_beta(self.r_ab)
        # میانگین بتای r از پارامترهای r_ab

    # ---------- feature maps ----------
    def gS(self, a_idx: int) -> np.ndarray:
        """
        S = [security, influence, cost]
        - امنیت با تنش داخلی کاهش می‌یابد.
        - نفوذ با d_c تقویت می‌شود.
        - هزینه با ریسک‌پذیری (rho) کمتر حس می‌شود.
        """
        # gS همان «نگاشت راهبردی» در کتابچه است (کتابچه: صفحه 12).
        # خروجی یک بردار 3تایی است: [امنیت, نفوذ, هزینه]

        sec = self.action_bases.sec_gain[a_idx] * (1.0 - self.tension)
        # امنیت:
        # - پایه امنیت اقدام از ActionBases گرفته می‌شود.
        # - با (1 - tension) ضرب می‌شود تا اگر تنش بالا بود، «اثر امنیتی واقعی» کمتر حس شود.
        # این ایده از کتابچه می‌آید که تنش (حالت نهفته) روی برداشت امنیت اثر دارد.

        inf = self.action_bases.inf_gain[a_idx] * self.d_c
        # نفوذ:
        # - پایه نفوذ اقدام از ActionBases گرفته می‌شود.
        # - با d_c (ترجیح بازدارندگی/نفوذ) تعدیل می‌شود:
        #   کشورهایی که d بالاتر دارند از سیگنال/نمایش قدرت نفوذ بیشتری می‌گیرند.

        cst = self.action_bases.cost[a_idx] * (1.0 - self.rho_c)
        # هزینه:
        # - پایه هزینه اقدام
        # - ضرب در (1 - rho): کشور ریسک‌پذیرتر (rho بزرگتر) هزینه را کمتر «احساس» می‌کند.
        # این ایده مطابق کتابچه است که ریسک‌پذیری روی ادراک هزینه اثر دارد (سطح دکترین).

        return np.array([sec, inf, cst], dtype=float)
        # برگرداندن بردار S برای استفاده در Utility و ψ_c.

    def gC(self, a_idx: int) -> np.ndarray:
        """
        O = [alloc, tempo, mobilize]
        - alloc: λ_op * 1[a∈{P,R}]
        - tempo: (1/τ) * 1[a=P]
        - mobilize: σ(γ(ε - tension))
        """
        # gC همان «نگاشت عملیاتی/Operational mapping» در کتابچه است (کتابچه: صفحه 12).
        # خروجی یک بردار 3تایی است: [alloc, tempo, mobilize]

        alloc = self.lambda_op * (1.0 if a_idx in (0, 2) else 0.0)
        # alloc:
        # - اگر اقدام P یا R باشد، تخصیص عملیات فعال می‌شود ⇒ 1
        # - اگر S باشد، تخصیص مستقیم کمتر است ⇒ 0
        # - λ_op شدت آن را تعیین می‌کند.
        # این دقیقاً مطابق فرم 1[ a∈{...} ] در کتابچه است.

        tempo = (1.0 / (self.tau_c + 1e-12)) * (1.0 if a_idx == 0 else 0.0)
        # tempo:
        # - فقط برای P معنا دارد (گشت‌زنی یعنی حضور مستمر/ریتم)
        # - (1/τ) یعنی هرچه τ کوچکتر ⇒ سرعت/ریتم بیشتر.
        # +1e-12 برای جلوگیری از تقسیم بر صفر (اصلاح مهندسی).

        mobilize = sigmoid(self.action_bases.gamma_e * (self.eps_c - self.tension))
        # mobilize:
        # - یک سیگموید از (ε - tension)
        # - اگر tension بالاتر از ε شود، (ε - tension) منفی می‌شود ⇒ mobilize کاهش می‌یابد
        # - اگر tension پایین‌تر از ε باشد، mobilize بالاتر می‌رود
        # γ_e شیب سیگموید را تند می‌کند.
        # این ایده مطابق کتابچه برای بسیج/آستانه بسیج است.

        return np.array([alloc, tempo, mobilize], dtype=float)
        # بازگرداندن بردار O

    def gR(self, a_idx: int) -> np.ndarray:
        """
        T = [eff_loss, fail_risk, learning_cost]
        - eff_loss با توان فنی eta_c کاهش می‌یابد.
        - fail_risk با احتمال موفقیت p و قابلیت اطمینان r کم می‌شود.
        - learning_cost با kappa بیشتر می‌شود.
        """
        # gR همان «نگاشت فنی/Technical mapping» در کتابچه است (کتابچه: صفحه 13).
        # خروجی یک بردار 3تایی است: [اتلاف کارایی, ریسک شکست, هزینه یادگیری]

        eff = self.action_bases.eff_loss[a_idx] / (self.eta_c + 1e-12)
        # eff_loss:
        # - پایه اتلاف اقدام
        # - تقسیم بر η_c: اگر توان فنی بیشتر باشد، اتلاف کمتر می‌شود (منطقی).
        # +1e-12 برای جلوگیری از تقسیم بر صفر.

        fail = self.action_bases.fail_risk[a_idx] * (1.0 - (self.p_c * self.r_c))
        # fail_risk:
        # - پایه ریسک شکست اقدام
        # - ضرب در (1 - p*r)
        # اگر p و r بالا باشند، (p*r) بزرگ می‌شود ⇒ (1 - p*r) کوچک ⇒ ریسک کمتر.
        # این همان ایده کتابچه درباره موفقیت/قابلیت اطمینان است.

        learn = self.action_bases.learning_cost[a_idx] * self.kappa_c
        # learning_cost:
        # - پایه هزینه یادگیری اقدام
        # - ضرب در κ: اگر κ بالاتر باشد، هزینه یادگیری بیشتر حس می‌شود.

        return np.array([eff, fail, learn], dtype=float)
        # بازگرداندن بردار T

    # ---------- utility ----------
    def utilities(self) -> np.ndarray:
        # این تابع Utility هر اقدام را حساب می‌کند.
        # کتابچه: U = ωS·gS + ωC·gC − ωR·gR (حوالی صفحات 11 تا 13)

        U = np.zeros(3, dtype=float)
        # یک آرایه طول 3 برای نگه داشتن U(P), U(S), U(R)

        for a in range(3):
            # روی هر اقدام a=0..2 حلقه می‌زنیم.

            U[a] = (self.omega_S @ self.gS(a)) + (self.omega_C @ self.gC(a)) - (self.omega_R @ self.gR(a))
            # محاسبه Utility طبق فرم کتابچه:
            # - ωS·gS: سود/زیان راهبردی
            # - ωC·gC: سود/زیان عملیاتی
            # - ωR·gR: ریسک/هزینه فنی با علامت منفی
            # @ در numpy یعنی ضرب داخلی (dot product)

        return U
        # خروجی: بردار مطلوبیت 3تایی برای استفاده در قانون انتخاب (لاجیت).

    # ---------- choice rule ----------
    def choice_probs(self) -> np.ndarray:
        # این تابع احتمال انتخاب هر اقدام را طبق MNL/Logit می‌دهد.
        # کتابچه: قانون لاجیت/رفتار عقلانیت محدود (حوالی صفحه 14)

        U = self.utilities()
        # ابتدا Utility های سه اقدام را حساب می‌کنیم.

        logits = (self.beta_c * U) + self.omega_a
        # ساخت logit ها:
        # β_c * U: حساسیت به سود
        # + ω_a: ترجیح تاکتیکی/عادت
        # کتابچه: βU + ω_{a,c}

        logits = logits - np.max(logits)
        # ترفند پایداری عددی:
        # اگر logits بزرگ باشند exp(logits) overflow می‌دهد.
        # کم کردن ماکزیمم باعث می‌شود همه logits <= 0 شوند ولی نسبت‌ها حفظ شود.

        ex = np.exp(logits)
        # تبدیل logits به exp(logits) برای softmax.

        return ex / (ex.sum() + 1e-12)
        # softmax:
        # احتمال هر اقدام = exp(logit) / sum(exp(logit))
        # +1e-12 برای جلوگیری از تقسیم بر صفر در شرایط خیلی خاص.

    def choose_action(self):
        # این تابع یک اقدام واقعی را با توجه به توزیع احتمال انتخاب می‌کند.

        probs = self.choice_probs()
        # گرفتن احتمال انتخاب هر اقدام.

        a = int(np.random.choice(3, p=probs))
        # انتخاب تصادفی وزن‌دار از بین 0..2
        # این همان «bounded rationality» است: همیشه بهترین اقدام را قطعی انتخاب نمی‌کند.

        return a, probs
        # خروجی:
        # - a اندیس اقدام
        # - probs توزیع احتمال (برای محاسبات مورد انتظار و تحلیل)

    # ---------- escalation ψ_c ----------
    def psi_c(self, action_idx: int, esc: EscalationCoeffs) -> float:
        # این تابع ψ_c(t) را می‌سازد:
        # کتابچه: ψ_c(t)=σ(αS^T S + αO^T O + αT^T T + δ^T Z) (صفحه 17)

        S = self.gS(action_idx)
        # محاسبه بردار راهبردی S برای اقدام انتخاب‌شده.

        O = self.gC(action_idx)
        # محاسبه بردار عملیاتی O برای اقدام انتخاب‌شده.

        T = self.gR(action_idx)
        # محاسبه بردار فنی/ریسکی T برای اقدام انتخاب‌شده.

        resource_norm = self.resource / (self.resource + 1000.0)
        # نرمال‌سازی منابع به بازه (0,1):
        # اگر resource خیلی بزرگ شود، اثرش اشباع می‌شود و scale کنترل می‌شود.

        Z = np.array([self.v_c, resource_norm], dtype=float)
        # ساخت بردار زمینه‌ای Z = [v_c, resource_norm]
        # که در δ^T Z وارد می‌شود.

        lin = (esc.alpha_S @ S) + (esc.alpha_O @ O) + (esc.alpha_T @ T) + (esc.delta @ Z)
        # ترکیب خطی فرمول کتابچه:
        # αS·S + αO·O + αT·T + δ·Z

        # calibration to avoid saturation
        lin = esc.psi_scale * (lin - esc.psi_bias)
        # این قسمت «اصلاح مهندسی» است (در کتابچه نیست):
        # - lin - bias باعث می‌شود ورودی سیگموید حول محدوده میانی باشد
        # - scale دامنه را کنترل می‌کند
        # نتیجه: ψ_c کمتر اشباع می‌شود ⇒ تنوع رفتاری بیشتر.

        return float(sigmoid(lin))
        # تبدیل ترکیب خطی به مقدار بین 0 و 1 با سیگموید و بازگرداندن.

    # ---------- learning ----------
    def update_beliefs(self, chosen_a: int, success: bool, escalated_any: bool):
        # این تابع یادگیری/آپدیت باورها را انجام می‌دهد.
        # کتابچه: به‌روزرسانی‌های تاکتیکی (عادت)، و فنی (Beta updates) در پیوست/بخش یادگیری.

        # tactical preference update (smooth)
        lr = 0.05
        # نرخ یادگیری برای آپدیت ترجیح تاکتیکی (عدد کوچک برای تغییر ملایم).

        target = np.zeros(3, dtype=float)
        # بردار هدف که نشان می‌دهد این بار کدام اقدام انجام شده.

        target[chosen_a] = 1.0
        # اقدام انتخاب‌شده را 1 می‌کنیم (one-hot).

        self.omega_a = (1 - lr) * self.omega_a + lr * target
        # آپدیت نرم:
        # ω_a جدید = ترکیب ω_a قبلی و one-hot اقدام انتخاب‌شده.
        # نتیجه: اگر یک اقدام زیاد تکرار شود، ترجیح تاکتیکی به سمت آن می‌رود.

        self.omega_a = self.omega_a / (self.omega_a.sum() + 1e-12)
        # نرمال‌سازی تا دوباره شبیه یک توزیع شود.

        # update p (success rate)
        if success:
            # اگر اقدام موفق بوده:
            self.p_ab[0] += 1.0
            # α (موفقیت‌ها) +1  (Beta-Bernoulli update)
        else:
            # اگر شکست خورده:
            self.p_ab[1] += 1.0
            # β (شکست‌ها) +1

        # update r (reliability under escalation)
        if escalated_any and (not success):
            # اگر در شرایط تشدید بوده و شکست هم خورده:
            self.r_ab[1] += 1.0
            # β در r زیاد می‌شود ⇒ میانگین r کاهش می‌یابد ⇒ اعتماد کمتر می‌شود.
        else:
            # در غیر این صورت:
            self.r_ab[0] += 0.3
            # کمی α اضافه می‌کنیم ⇒ r آرام آرام بالا می‌رود (یادگیری مثبت ملایم).

    # ---------- doctrine update (every N occurrences of an action) ----------
    def record_action_and_maybe_update_doctrine(self, action_idx: int, doctrine_every: int):
        """
        doctrine_every:
          - اگر 0 یا None باشد: دکترین ثابت می‌ماند.
          - اگر N باشد: هر بار تعداد کل اجرای یک اقدام (مثلاً R) به مضرب N برسد، دکترین کمی تغییر می‌کند.
        """
        # این تابع جزء کتابچه نیست؛ پاسخ به درخواست توست:
        # «بعد از N بار تکرار یک اقدام (پشت سر هم مهم نیست) دکترین تغییر کند».

        self.action_counts[action_idx] += 1
        # شمارنده اقدام مربوطه را 1 واحد زیاد می‌کنیم.

        if doctrine_every is None or int(doctrine_every) <= 0:
            # اگر کاربر N=0 یا None داده، یعنی دکترین ثابت بماند.
            return
            # خروج بدون تغییر.

        N = int(doctrine_every)
        # تبدیل ورودی به عدد صحیح قابل استفاده.

        if self.action_counts[action_idx] % N != 0:
            # اگر تعداد انجام این اقدام هنوز به مضرب N نرسیده، کاری نمی‌کنیم.
            return

        # منطق تغییر (ملایم و قابل فهم):
        # این قسمت «طراحی رفتاری» است، نه متن مستقیم کتابچه.
        # ولی با روح مدل سازگار است: تکرار رفتار => تغییر دکترین/نگرش.

        if action_idx == 2:  # R
            # اگر اقدام R زیاد انجام شود:
            # برداشت: کشور «تهاجمی‌تر» می‌شود، ریسک‌پذیرتر می‌شود، و منابع بیشتری برای اعمال قدرت خرج می‌کند.

            self.rho_c = float(np.clip(self.rho_c + 0.03, 0.0, 1.0))
            # rho ↑ : ریسک‌پذیری کمی بیشتر

            self.f_c = float(np.clip(self.f_c - 0.02, 0.0, 1.0))
            # f ↓ : آستانه استفاده از زور کمی کمتر (راحت‌تر وارد زور می‌شود)

            self.d_c = float(np.clip(self.d_c - 0.01, 0.0, 1.0))
            # d ↓ : تمرکز روی نفوذ/بازدارندگی نرم کمی کم می‌شود (چون به زور تکیه کرده)

            self.chi_c = float(np.clip(self.chi_c + 0.03, 0.4, 3.0))
            # chi ↑ : هزینه‌کرد عملیاتی/مصرف منابع بیشتر (رفتار تهاجمی معمولاً پرهزینه‌تر است)

        elif action_idx == 1:  # S
            # اگر اقدام S زیاد انجام شود:
            # برداشت: کشور بیشتر دنبال نفوذ/بازدارندگی نرم است.

            self.d_c = float(np.clip(self.d_c + 0.03, 0.0, 1.0))
            # d ↑ : تمرکز روی نفوذ بیشتر

            self.f_c = float(np.clip(self.f_c + 0.01, 0.0, 1.0))
            # f ↑ : کمی محتاط‌تر نسبت به زور می‌شود

            self.rho_c = float(np.clip(self.rho_c - 0.01, 0.0, 1.0))
            # rho ↓ : کمی ریسک‌گریزتر می‌شود (چون مسیرش نرم‌تر/دیپلماتیک‌تر است)

        else:  # P
            # اگر اقدام P زیاد انجام شود:
            # برداشت: کشور محافظه‌کارتر می‌شود و به ثبات/حضور مستمر متکی است.

            self.f_c = float(np.clip(self.f_c + 0.02, 0.0, 1.0))
            # f ↑ : آستانه زور بالاتر ⇒ کمتر زور می‌زند

            self.rho_c = float(np.clip(self.rho_c - 0.02, 0.0, 1.0))
            # rho ↓ : ریسک‌پذیری کمتر

            self.chi_c = float(np.clip(self.chi_c + 0.01, 0.4, 3.0))
            # chi کمی ↑ : چون حتی حضور مستمر هم هزینه دارد، ولی افزایش کوچک است.

    # ---------- state dynamics ----------
    def update_state(self, E_U: float, psi: float, chosen_action: int):
        """
        تنش:
          tension_{t+1}=σ(α0+αv v + αψ ψ + αa E[U] - αr resource_norm)

        منابع:
          resource_{t+1} = resource_t + income_c - spend(action)
        spend(action) = chi_c * cost(action) * scale(resource)
        """
        # این تابع حالت کشور را بعد از هر گام آپدیت می‌کند:
        # - تنش (کتابچه: دینامیک تنش/State update، حدود صفحات 15 تا 21)
        # - منابع (این بخش را مهندسی کرده‌ایم تا منطقی و پایدار باشد)

        resource_norm = self.resource / (self.resource + 1000.0)
        # منابع را نرمال می‌کنیم تا در فرمول تنش مقیاس درست شود.

        t_next = sigmoid(
            # شروع ساختن ورودی سیگموید برای tension_{t+1} طبق فرمول کتابچه.

            self.dyn.alpha0
            # α0 بایاس پایه

            + self.dyn.alpha_v * self.v_c
            # + αv * v_c اثر آسیب‌پذیری مرزی

            + self.dyn.alpha_psi * psi
            # + αψ * ψ اثر مستعد بودن برای تشدید

            + self.dyn.alpha_a * E_U
            # + αa * E[U] اثر مطلوبیت مورد انتظار تصمیم‌ها

            - self.dyn.alpha_r * resource_norm
            # - αr * resource_norm اثر کاهش‌دهنده منابع (منابع بیشتر ⇒ فشار/تنش کمتر)
        )

        base_cost = float(self.action_bases.cost[chosen_action])
        # هزینه پایه اقدام انتخاب‌شده را می‌گیریم.
        # نکته مهم: اینجا «هزینه واقعاً اقدام انتخاب‌شده» کم می‌شود
        # (نه میانگین مورد انتظار روی همه اقدامات) تا رفتار ملموس‌تر باشد.

        scale = (self.resource / 1000.0)
        # یک مقیاس هزینه: اگر کشور منابع بیشتری دارد، عملیات‌ها در مقیاس بزرگتری انجام می‌دهد.
        # این باعث می‌شود خرج کردن برای کشور بزرگ‌تر طبیعی‌تر باشد.

        spend = self.chi_c * base_cost * (50.0 * scale)
        # خرج کردن:
        # - chi_c شدت هزینه‌کرد
        # - base_cost هزینه اقدام
        # - 50.0 یک ضریب کالیبراسیون است (مهندسی) تا اعداد در بازه مناسب باشند.
        # دلیل وجود این ضریب: اگر فقط base_cost باشد، خرج‌ها خیلی کوچک می‌شود و منابع دائماً رشد می‌کند.

        r_next = self.resource + self.income_c - spend
        # آپدیت منابع:
        # منابع جدید = منابع قبلی + درآمد - خرج
        # این تفکیک درآمد و خرج مشکل «همیشه افزایش منابع» را حل می‌کند.

        self.tension = float(np.clip(t_next, 0.0, 1.0))
        # تنش را در [0,1] نگه می‌داریم (چون معنای احتمال/شدت دارد).

        self.resource = float(max(0.0, r_next))
        # منابع منفی معنا ندارد ⇒ حداقل 0 در نظر می‌گیریم.


# ==========================================================
# 4) World with directed targeting (solves "who acts against whom")
# ==========================================================
# این کلاس «جهان» است: یعنی محیطی که کشورها در آن با هم تعامل می‌کنند.
# بخش کلیدی که مشکل تو را حل می‌کند: هر کشور علاوه بر Action، یک Target هم دارد.
# بنابراین تعامل «علیه چه کسی» مشخص می‌شود (کتابچه: تعاملات بین کشورها / ψ_ij / صفحه 17).

class MultiAgentWorld:
    """
    - هر کشور در هر گام: (action, target) انتخاب می‌کند.
    - تعاملات روی یال‌های جهت‌دار (i -> j) اتفاق می‌افتند.
    - وزن تعامل (W_ij) مشخص می‌کند احتمال هدف‌گیری j توسط i چقدر است.
    """

    # این docstring توضیح می‌دهد چرا این نسخه با نسخه ساده فرق دارد:
    # نسخه ساده: فقط تعداد R ها را می‌شمردیم ⇒ تعامل واقعی i و j نداشت.
    # این نسخه: هدف‌گیری جهت‌دار + ψ_ij فقط روی همان یال‌های انتخاب‌شده اعمال می‌شود.

    def __init__(self, agents, interaction_W=None, esc_coeffs=None, doctrine_update_every: int = 0,
                 bayes_update_every: int = 10, bayes_window: int = 2000, bayes_min_samples: int = 200):
        # سازنده جهان:
        # - agents: لیست کشورها
        # - interaction_W: ماتریس وزن تعامل W_ij
        # - esc_coeffs: ضرایب فرمول‌های ψ
        # - doctrine_update_every: هر N بار تکرار اقدام، دکترین تغییر کند

        self.agents = list(agents)
        # لیست عامل‌ها را ذخیره می‌کنیم.

        self.esc = esc_coeffs if esc_coeffs is not None else EscalationCoeffs()
        # اگر ضرایب داده شد از آن استفاده می‌کنیم، وگرنه پیش‌فرض EscalationCoeffs می‌سازیم.

        self.history = []
        # ---------------------------
        # Buffers for booklet-style Bayesian/MAP updating of escalation coefficients (α, η)
        # ---------------------------
        self.bayes_update_every = int(bayes_update_every) if bayes_update_every is not None else 0
        self.bayes_window = int(bayes_window) if bayes_window is not None else 2000
        self.bayes_min_samples = int(bayes_min_samples) if bayes_min_samples is not None else 200

        # country-level dataset: X_c = [S(3), O(3), T(3), Z(2)]  -> y_c = {0,1} "escalated_any"
        self._country_X = []
        self._country_y = []

        # edge-level dataset: X_ij = [psi_i, psi_j, psi_i*psi_j, (W_ij-0.5), 1] -> y_ij
        self._edge_X = []
        self._edge_y = []

        # تاریخچه هر گام زمانی را در این لیست ذخیره می‌کنیم تا بعداً DataFrame بسازیم.

        self.doctrine_update_every = int(doctrine_update_every) if doctrine_update_every is not None else 0
        # مقدار N برای آپدیت دکترین را ذخیره می‌کنیم.

        n = len(self.agents)
        # تعداد کشورها.

        if interaction_W is None:
            # اگر ماتریس تعامل داده نشد:

            W = np.zeros((n, n), dtype=float)
            # پیش‌فرض: رابطه خنثی (۰) بین همه.

            np.fill_diagonal(W, 0.0)
            # قطر اصلی صفر می‌شود چون کشور نباید خودش را هدف بگیرد.

            self.W = W
            # ذخیره ماتریس.

        else:
            # اگر W داده شد:

            W = np.array(interaction_W, dtype=float)
            # تبدیل به آرایه numpy برای محاسبات و انتخاب هدف.

            W = np.clip(W, -1.0, 1.0)
            # از این نسخه به بعد W بازه [-1,+1] دارد:
            # -1 = بیشترین تقابل (هدف‌گیری بیشتر)
            # +1 = بیشترین همسویی (هدف‌گیری کمتر)

            np.fill_diagonal(W, 0.0)
            # تضمین اینکه قطر اصلی صفر باشد (حتی اگر کاربر اشتباه داده باشد).

            self.W = W
            # ذخیره ماتریس.

    @staticmethod
    def _w_signed_to_weight01(w_signed: float) -> float:
        """Convert signed W in [-1,+1] to a nonnegative weight in [0,1].

        -1 (تقابل شدید)  -> 1.0  (هدف‌گیری زیاد)
         0 (خنثی)        -> 0.5
        +1 (همسویی)      -> 0.0  (هدف‌گیری کم)
        """
        w = float(w_signed)
        w01 = (1.0 - w) / 2.0
        return float(np.clip(w01, 0.0, 1.0))

    def _pick_target(self, i: int) -> int:
        # این تابع برای کشور i یک هدف j انتخاب می‌کند.
        # W امضادار است ([-1,+1]) و برای انتخاب هدف به وزن [0,1] تبدیل می‌شود.

        w_signed = self.W[i].copy()
        # یک کپی از رابطه‌های i با همه j ها

        w = (1.0 - w_signed) / 2.0
        # نگاشت: -1→1 ، +1→0

        w[i] = 0.0
        w = np.clip(w, 0.0, 1.0)

        if w.sum() <= 1e-12:
            # اگر کل وزن‌ها تقریباً صفر بود (یعنی کاربر همه صفر داده):

            choices = [j for j in range(len(self.agents)) if j != i]
            # تمام کشورهای غیر از خود i را لیست می‌کنیم.

            return int(np.random.choice(choices))
            # انتخاب یکنواخت تصادفی از بین آن‌ها (fallback منطقی).

        probs = w / w.sum()
        # نرمال‌سازی وزن‌ها به احتمال (جمع=1).

        return int(np.random.choice(len(self.agents), p=probs))
        # انتخاب تصادفی وزن‌دار از بین همه کشورها بر اساس probs.

    def _maybe_update_escalation_coeffs(self, t: int):
        """Booklet-style updating of escalation coefficients.

        To keep the engine stable and fast, we implement MAP (not full MCMC):
        - α = argmax p(α | y_c, X_c)  with Bernoulli-Logit likelihood + Normal prior
        - η = argmax p(η | y_ij, X_ij) with Bernoulli-Logit likelihood + Normal prior

        This is the minimal correction requested when someone says:
        'the booklet's posterior update for α/η is not implemented in the code'.
        """
        if self.bayes_update_every <= 0:
            return
        if t <= 0:
            return
        if (t % self.bayes_update_every) != 0:
            return

        # need enough samples to avoid noisy updates
        if (len(self._edge_y) < self.bayes_min_samples) or (len(self._country_y) < self.bayes_min_samples):
            return

        # ---- update α (11-dim) ----
        Xc = np.vstack(self._country_X)
        yc = np.asarray(self._country_y, dtype=float)

        w0_alpha = np.concatenate([self.esc.alpha_S, self.esc.alpha_O, self.esc.alpha_T, self.esc.delta])
        w_alpha = fit_logistic_map(Xc, yc, w0=w0_alpha, l2=0.8, lr=0.25, iters=160)

        self.esc.alpha_S = w_alpha[0:3].astype(float)
        self.esc.alpha_O = w_alpha[3:6].astype(float)
        self.esc.alpha_T = w_alpha[6:9].astype(float)
        self.esc.delta = w_alpha[9:11].astype(float)

        # ---- update η (5-dim) ----
        Xe = np.vstack(self._edge_X)
        ye = np.asarray(self._edge_y, dtype=float)

        w0_eta = np.array([self.esc.eta1, self.esc.eta2, self.esc.eta3, self.esc.eta_W, self.esc.eta_bias], dtype=float)
        w_eta = fit_logistic_map(Xe, ye, w0=w0_eta, l2=0.8, lr=0.25, iters=160)

        self.esc.eta1 = float(w_eta[0])
        self.esc.eta2 = float(w_eta[1])
        self.esc.eta3 = float(w_eta[2])
        self.esc.eta_W = float(w_eta[3])
        # note: last weight multiplies constant 1.0 feature, so it's a bias term
        self.esc.eta_bias = float(w_eta[4])

    def _dyad_tension(self, psi_i: float, psi_j: float, w_ij: float) -> float:
        """Pairwise (directed) tension proxy in [0,1].

        Notes
        -----
        - This is *not* W (structural targeting preference).
        - It is a dynamic, time-indexed dyadic signal derived from the same
          Bernoulli-Logit form used for ψ_ij in the model.
        """
        w_ij = self._w_signed_to_weight01(w_ij)
        base = (
                (self.esc.eta1 * float(psi_i))
                + (self.esc.eta2 * float(psi_j))
                + (self.esc.eta3 * float(psi_i) * float(psi_j))
                + float(self.esc.eta_bias)
                + (float(self.esc.eta_W) * (w_ij - 0.5))
        )
        return float(sigmoid(base))

    def step(self, t: int):
        # اجرای یک گام زمانی t:
        # اینجا سه فاز داریم:
        # 1) انتخاب action و target و محاسبه ψ_c
        # 2) محاسبه ψ_ij و نمونه‌گیری Y_ij برای یال‌های انتخاب‌شده
        # 3) یادگیری + آپدیت حالت (تنش/منابع)

        step_data = {"Time": t}

        # --- initialize crisis attribution matrix for this timestep ---
        # Crisis_{src}_{dst} in {0,1}  and CrisisProb_{src}_{dst} in [0,1]
        _names = [ag.name for ag in self.agents]
        for _src in _names:
            for _dst in _names:
                step_data[f"Crisis_{_src}_{_dst}"] = 0
                step_data[f"CrisisProb_{_src}_{_dst}"] = 0.0
        # یک دیکشنری می‌سازیم که همه نتایج این گام داخلش ذخیره شود.
        # بعداً هر step_data یک ردیف در DataFrame خواهد شد.

        n = len(self.agents)
        # تعداد عامل‌ها برای حلقه‌ها.

        actions = [None] * n
        # لیست اقدامات انتخابی هر کشور در این گام.

        targets = [None] * n
        # لیست هدف انتخابی هر کشور در این گام (جهت‌دار بودن را ایجاد می‌کند).

        probs_list = [None] * n
        # ذخیره توزیع احتمال انتخاب اقدام برای هر کشور (برای E[U]).

        utilities_list = [None] * n
        # ذخیره بردار U(P,S,R) برای هر کشور.

        psi_list = [None] * n
        # ذخیره ψ_c هر کشور.

        # Phase 1: each agent chooses action + target, compute ψ_c
        for i, ag in enumerate(self.agents):
            # روی هر کشور ag با اندیس i حلقه می‌زنیم.

            a, probs = ag.choose_action()
            # کشور اقدام را طبق لاجیت انتخاب می‌کند (کتابچه: صفحه 14).
            # a اندیس اقدام، probs توزیع احتمال.

            U = ag.utilities()
            # محاسبه Utility های سه اقدام (کتابچه: U=... صفحات 11-13).

            psi = ag.psi_c(a, self.esc)
            # محاسبه ψ_c برای همین اقدام انتخاب‌شده (کتابچه: ψ_c صفحه 17).

            j = self._pick_target(i)
            # انتخاب هدف j برای کشور i بر اساس ماتریس تعامل W.
            # این بخش در کتابچه «جهت‌دار» به این شکل صریح نیست،
            # اما همان مفهوم تعامل i و j را عملیاتی می‌کند.

            # record action for doctrine update
            ag.record_action_and_maybe_update_doctrine(a, self.doctrine_update_every)
            # شمارش اقدام و اگر به N رسید ⇒ تغییر دکترین (این بخش افزوده‌ی مهندسی طبق درخواست توست).

            actions[i] = a
            # ذخیره اقدام انتخابی کشور i.

            targets[i] = j
            # ذخیره هدف انتخابی کشور i.

            probs_list[i] = probs
            # ذخیره توزیع احتمال انتخاب اقدام (برای محاسبه E[U] در update_state).

            utilities_list[i] = U
            # ذخیره Utility ها برای کشور i.

            psi_list[i] = psi
            # ذخیره ψ_c کشور i.

            step_data[f"Action_{ag.name}"] = ACT_MAP[a]
            # ثبت اقدام به شکل P/S/R در تاریخچه (برای نمودارها).

            step_data[f"Target_{ag.name}"] = self.agents[j].name
            # ثبت نام هدف انتخاب‌شده توسط کشور (برای گراف تعاملات).

            step_data[f"Tension_{ag.name}"] = ag.tension
            # ثبت تنش فعلی کشور در این گام (قبل از آپدیت).

            step_data[f"Resource_{ag.name}"] = ag.resource
            # ثبت منابع فعلی کشور در این گام (قبل از آپدیت).

            step_data[f"Psi_{ag.name}"] = psi
            # ثبت ψ_c کشور.

        # --- NEW OUTPUT: directed dyadic tension matrix (all pairs) ---
        # DyadTension_{src}_{dst} in [0,1]
        for i in range(n):
            src = self.agents[i].name
            for j in range(n):
                if j == i:
                    continue
                dst = self.agents[j].name
                step_data[f"DyadTension_{src}_{dst}"] = self._dyad_tension(
                    psi_list[i],
                    psi_list[j],
                    self.W[i, j],
                )

        # Phase 2: directed dyadic escalation ψ_ij + Y_ij (only for chosen targets)
        escalated_any_for_agent = [False] * n
        # یک لیست پرچم برای هر کشور:
        # اگر در این گام درگیر تشدید شد True می‌شود (برای یادگیری/کاهش موفقیت).

        global_escalation = 0
        # یک پرچم کلی: اگر حداقل یک یال تشدید شد، این 1 می‌شود (برای نمودار global).

        for i in range(n):
            # روی هر کشور i:

            j = targets[i]
            # هدف انتخاب‌شده توسط i.

            psi_i = psi_list[i]
            # ψ کشور i.

            psi_j = psi_list[j]
            # ψ کشور j (هدف).

            w_ij_signed = float(self.W[i, j])
            # رابطه i با j از ماتریس W (امضادار [-1,+1]).

            w_ij = self._w_signed_to_weight01(w_ij_signed)
            # تبدیل برای استفاده در ψ_ij (بازه 0..1)

            base = (self.esc.eta1 * psi_i) + (self.esc.eta2 * psi_j) + (
                        self.esc.eta3 * psi_i * psi_j) + self.esc.eta_bias
            # ساختن ورودی پایه ψ_ij طبق کتابچه:
            # ψ_ij = σ(η1 ψ_i + η2 ψ_j + η3 ψ_i ψ_j)  (کتابچه: صفحه 17)
            # هنوز سیگموید را اعمال نکردیم؛ فعلاً base = داخل σ.

            base = base + (self.esc.eta_W * (w_ij - 0.5))
            # این جمله در کتابچه نیست (اصلاح مهندسی):
            # اثر وزن تعامل W_ij را اضافه می‌کند.
            # - اگر w_ij=0.5 ⇒ اثر صفر (خنثی)
            # - اگر w_ij>0.5 ⇒ احتمال تشدید کمی بیشتر
            # - اگر w_ij<0.5 ⇒ احتمال کمتر
            # ضریب 0.6 برای اینکه اثر زیاد انفجاری نشود.

            psi_ij = float(sigmoid(base))
            # تبدیل base به احتمال بین 0 و 1 با سیگموید.

            y_ij = 1 if (np.random.random() < psi_ij) else 0
            # نمونه‌گیری برنولی:
            # با احتمال ψ_ij تشدید رخ می‌دهد (Y=1)، وگرنه رخ نمی‌دهد (Y=0).
            # این منطق همان Bernoulli observation در کتابچه است (تعامل مشاهده‌ای y_ij).

            step_data[f"PsiEdge_{self.agents[i].name}_{self.agents[j].name}"] = psi_ij
            # ذخیره احتمال یال برای استفاده در UI/تحلیل (اختیاری ولی مفید).

            step_data[f"Y_{self.agents[i].name}_{self.agents[j].name}"] = int(y_ij)

            # --- New: directed crisis attribution (matrix-friendly) ---
            # برای اینکه UI بتواند یک ماتریس N×N از «عامل بحران» بسازد،
            # همین رخداد یال i→j را داخل ستون‌های Crisis_* ذخیره می‌کنیم.
            step_data[f"CrisisProb_{self.agents[i].name}_{self.agents[j].name}"] = float(psi_ij)
            step_data[f"Crisis_{self.agents[i].name}_{self.agents[j].name}"] = int(y_ij)

            # --- booklet-style data for η (edge-level Bernoulli-Logit) ---
            if self.bayes_update_every > 0:
                Xij = np.array([psi_i, psi_j, psi_i * psi_j, (w_ij - 0.5), 1.0], dtype=float)
                self._edge_X.append(Xij)
                self._edge_y.append(float(y_ij))

                # keep buffers bounded
                if len(self._edge_y) > self.bayes_window:
                    excess = len(self._edge_y) - self.bayes_window
                    if excess > 0:
                        del self._edge_X[:excess]
                        del self._edge_y[:excess]
            # ذخیره رخداد واقعی تشدید روی یال i→j برای گراف:
            # Y=1 یعنی تشدید رخ داده، Y=0 یعنی رخ نداده.

            if y_ij == 1:
                # اگر تشدید رخ داده:

                global_escalation = 1
                # پرچم کلی تشدید روشن می‌شود.

                escalated_any_for_agent[i] = True
                # کشور i درگیر تشدید شده.

                escalated_any_for_agent[j] = True
                # کشور j هم به عنوان طرف مقابل درگیر تشدید شده.

        step_data["Global_Escalation"] = int(global_escalation)

        # --- booklet-style data for α (country-level Bernoulli-Logit) ---
        # y_c,t : whether the country was involved in any escalation this step (as initiator or target)
        if self.bayes_update_every > 0:
            for i, ag in enumerate(self.agents):
                a = actions[i]
                S = ag.gS(a)
                O = ag.gC(a)
                T = ag.gR(a)
                resource_norm = ag.resource / (ag.resource + 1000.0)
                Z = np.array([ag.v_c, resource_norm], dtype=float)
                Xc = np.concatenate([S, O, T, Z])  # 11-dim
                yc = 1.0 if escalated_any_for_agent[i] else 0.0
                self._country_X.append(Xc)
                self._country_y.append(yc)

            # keep buffers bounded
            if len(self._country_y) > self.bayes_window:
                excess = len(self._country_y) - self.bayes_window
                if excess > 0:
                    del self._country_X[:excess]
                    del self._country_y[:excess]
        # ثبت وضعیت کلی تشدید برای نمودار global escalation.

        # Phase 3: feedback + learning + state update
        # booklet-style MAP update for escalation coefficients (α, η)
        self._maybe_update_escalation_coeffs(t)

        for i, ag in enumerate(self.agents):
            # روی هر کشور برای یادگیری و آپدیت حالت:

            probs = probs_list[i]
            # توزیع احتمال انتخاب اقدام.

            U = utilities_list[i]
            # بردار Utility.

            a = actions[i]
            # اقدام انتخابی واقعی در این گام.

            base_success = 0.82 if a != 2 else 0.60
            # احتمال موفقیت پایه برای اقدام:
            # - P/S موفق‌تر و آسان‌تر (0.82)
            # - R سخت‌تر و پرریسک‌تر (0.60)
            # این یک تقریب مهندسی است چون محیط واقعی نداریم؛ ولی منطقی است.

            if escalated_any_for_agent[i] and a == 2:
                # اگر کشور در شرایط تشدید بوده و اقدامش هم R بوده:
                base_success -= 0.08
                # موفقیت کمتر می‌شود چون درگیری واقعی سخت‌تر است.

            base_success = max(0.05, min(0.95, base_success))
            # محدود کردن احتمال موفقیت به بازه امن (نه 0، نه 1) برای جلوگیری از یکنواختی.

            success = (np.random.random() < base_success)
            # نمونه‌گیری موفقیت/شکست (Bernoulli).
            # نتیجه وارد update_beliefs می‌شود تا p و r آپدیت شود.

            ag.update_beliefs(a, success, escalated_any_for_agent[i])
            # یادگیری:
            # - آپدیت ω_a (تاکتیک)
            # - آپدیت p و r (تکنیک) با Beta updates
            # - escalated_any کمک می‌کند r در بحران حساس شود.

            E_U = float(np.sum(probs * U))
            # محاسبه مطلوبیت مورد انتظار E[U] با استفاده از توزیع انتخاب.
            # این مطابق ایده کتابچه است که E[U] در دینامیک حالت استفاده می‌شود.

            ag.update_state(E_U=E_U, psi=psi_list[i], chosen_action=a)
            # آپدیت حالت:
            # - تنش با فرمول سیگموید (کتابچه)
            # - منابع با درآمد - خرج (اصلاح مهندسی برای واقعی‌تر شدن)

        self.history.append(step_data)
        # ثبت همه اطلاعات این گام در history تا بعداً DataFrame ساخته شود.
