import datetime as dt

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import norm

st.set_page_config(page_title="Starrs Options Lab", layout="wide", initial_sidebar_state="collapsed")

# ------------------------------------------------------------------ terminal palette
BG, PANEL = "#000000", "#0A0A0A"
ORANGE, YELLOW, WHITE = "#FB8B1E", "#FFD400", "#E6E6E6"
GREEN, RED, CYAN = "#3CE35A", "#FF433D", "#4FC3F7"
REDBAR, BLUEBAR, GRAYTAB = "#A3141C", "#1530A8", "#3A3A3A"
GRID = "#2E2E2E"
MONO = "'Lucida Console', Menlo, Consolas, 'Courier New', monospace"

st.markdown(f"""
<style>
header[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"], #MainMenu, footer {{display:none !important;}}
.stApp {{background:{BG};}}
.block-container {{padding:0.35rem 0.6rem 0.5rem 0.6rem !important; max-width:100% !important;}}
html, body, .stApp, p, div, span, label, input, button, td, th {{font-family:{MONO} !important;}}
[data-testid="stVerticalBlock"] {{gap:2px !important;}}
[data-testid="stHorizontalBlock"] {{gap:6px !important;}}
[data-testid="stMarkdownContainer"] p {{margin:0;}}
[data-testid="stNumberInputStepUp"], [data-testid="stNumberInputStepDown"] {{display:none !important;}}
[data-testid="stMarkdownContainer"] {{margin-bottom:0 !important;}}
[data-testid="stNumberInputContainer"], [data-testid="stTextInputRootElement"] {{background:#F39F1E !important; border:0 !important; border-radius:0 !important; height:19px !important; min-height:19px !important; box-shadow:none !important;}}
[data-testid="stNumberInputField"], [data-testid="stTextInputField"] {{color:#000 !important; background:#F39F1E !important; font-size:12px !important; padding:0 5px !important; height:19px !important; min-height:19px !important; text-align:right; font-weight:700; border:0 !important;}}
[data-testid="stTextInputField"] {{text-align:left;}}
[data-testid="stSelectbox"] div[data-baseweb="select"] > div {{background:#F39F1E !important; border:0 !important; border-radius:0 !important; min-height:19px !important; height:19px !important;}}
[data-testid="stSelectbox"] div[data-baseweb="select"] * {{color:#000 !important; fill:#000 !important; font-size:12px !important; padding-top:0 !important; padding-bottom:0 !important;}}
[data-testid="stRadio"] label p {{font-size:11px !important; color:{WHITE} !important;}}
[data-testid="stRadio"] > div {{gap:8px !important;}}
[data-testid="stRadio"] [role="radiogroup"] {{gap:10px !important;}}
[data-testid="stWidgetLabel"] {{min-height:0 !important; margin:0 !important;}}
[role="tablist"] {{gap:2px !important; background:{BG}; border-bottom:1px solid #555; min-height:0 !important; height:21px !important;}}
[role="tab"] {{background:{GRAYTAB} !important; height:20px !important; padding:0 14px !important; border-radius:0 !important; margin:0 !important;}}
[role="tab"] p {{color:{WHITE} !important; font-size:12px !important;}}
[role="tab"][aria-selected="true"] {{background:#BDBDBD !important;}}
[role="tab"][aria-selected="true"] p {{color:#000 !important; font-weight:700;}}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {{display:none !important;}}
.stTabs [data-baseweb="tab-panel"] {{padding-top:4px !important;}}

.menu {{display:flex; gap:4px; align-items:center; background:#1B1B1B; color:#BDBDBD; font-size:11px; padding:1px 4px;}}
.menu .dd {{background:#2B3A8C; color:#fff; padding:0 6px;}}
.menu .rt {{margin-left:auto; color:#BDBDBD;}}
.sec1 {{display:flex; gap:22px; align-items:baseline; padding:2px 4px; font-size:13px; color:{WHITE};}}
.sec1 .tk {{color:{WHITE}; font-size:17px; font-weight:700;}}
.sec1 .px {{color:{GREEN}; font-size:17px; font-weight:700;}}
.sec1 .l {{color:{ORANGE};}}
.redbar {{display:flex; gap:28px; background:{REDBAR}; color:#fff; font-size:12px; padding:1px 6px;}}
.redbar .ttl {{margin-left:auto; font-weight:700;}}
.ph {{background:#262626; color:#fff; font-size:11px; padding:1px 5px; margin-top:4px; display:flex; justify-content:space-between;}}
.ph b {{color:{YELLOW}; font-weight:400;}}
.fl {{color:{ORANGE}; font-size:11.5px; line-height:19px; white-space:nowrap; overflow:hidden;}}
.fl span {{color:#7A7A7A;}}
table.bb {{border-collapse:collapse; width:100%; font-size:12px; line-height:15px;}}
.bb th {{color:{WHITE}; font-weight:400; text-align:right; padding:1px 6px; border-bottom:1px solid #4A4A4A; white-space:nowrap;}}
.bb td {{color:{WHITE}; text-align:right; padding:0 6px; white-space:nowrap;}}
.bb .l {{text-align:left;}}
.bb tr:nth-child(even) td {{background:#0B0B0B;}}
.bb tr.sec td {{background:#13207A !important; color:#fff; text-align:left;}}
.bb td.o {{color:{ORANGE};}} .bb td.g {{color:#8C8C8C;}} .bb td.y {{color:{YELLOW}; font-weight:700;}}
.bb td.p {{color:{GREEN};}} .bb td.n {{color:{RED};}}
.bb td.k {{background:#1D1D1D !important; color:{YELLOW}; text-align:center; font-weight:700;}}
.bb td.itm {{background:#14240F !important;}}
.bb tr.atm td {{border-top:1px solid {ORANGE}; border-bottom:1px solid {ORANGE};}}
.bluebar {{display:flex; justify-content:space-between; background:{BLUEBAR}; color:#fff; font-size:12px; padding:2px 8px; margin-top:6px;}}
.status {{color:#9A9A9A; font-size:11px; padding:1px 4px;}}
.status b {{color:{YELLOW}; font-weight:400;}}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ math
def black_scholes(S, K, T, r, q, sigma, kind):
    d1 = (np.log(S / K) + (r - q + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    dq, dr = np.exp(-q * T), np.exp(-r * T)
    if kind == "Call":
        price = S * dq * norm.cdf(d1) - K * dr * norm.cdf(d2)
        delta = dq * norm.cdf(d1)
        theta = (-S * dq * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                 - r * K * dr * norm.cdf(d2) + q * S * dq * norm.cdf(d1)) / 365
    else:
        price = K * dr * norm.cdf(-d2) - S * dq * norm.cdf(-d1)
        delta = dq * (norm.cdf(d1) - 1)
        theta = (-S * dq * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                 + r * K * dr * norm.cdf(-d2) - q * S * dq * norm.cdf(-d1)) / 365
    gamma = dq * norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * dq * norm.pdf(d1) * np.sqrt(T) / 100
    return price, delta, gamma, theta, vega


def ivol(m, T, atm, skew, smile, tslope):
    """Parametric smile: m = K/S. skew & smile in vol pts at 10% moneyness, term slope in vol pts per sqrt-year."""
    x = 1 - m
    v = atm + (skew / 100) * (x / 0.10) + (smile / 100) * (x / 0.10) ** 2 + (tslope / 100) * (np.sqrt(T) - np.sqrt(0.25))
    return np.maximum(v, 0.01)


# ------------------------------------------------------------------ formatting
def f_money(x, dec=0):
    s = f"{abs(x):,.{dec}f}"
    return f"({s})" if x < 0 else s


def cls(x):
    return "p" if x > 0 else ("n" if x < 0 else "")


def table(header, rows, cols_left=(0,)):
    h = "".join(f'<th class="{"l" if i in cols_left else ""}">{c}</th>' for i, c in enumerate(header))
    body = ""
    for row in rows:
        if isinstance(row, str):
            body += f'<tr class="sec"><td colspan="{len(header)}">{row}</td></tr>'
            continue
        tds = ""
        for i, cell in enumerate(row):
            text, c = cell if isinstance(cell, tuple) else (cell, "")
            c = (c + " l") if i in cols_left else c
            tds += f'<td class="{c}">{text}</td>'
        body += f"<tr>{tds}</tr>"
    st.markdown(f'<table class="bb"><thead><tr>{h}</tr></thead><tbody>{body}</tbody></table>', unsafe_allow_html=True)


def ph(title, right=""):
    st.markdown(f'<div class="ph"><span>{title}</span><b>{right}</b></div>', unsafe_allow_html=True)


def field(label, sym, **kw):
    a, b = st.columns([1.35, 1])
    a.markdown(f'<div class="fl">{label} <span>{sym}</span></div>', unsafe_allow_html=True)
    with b:
        return st.number_input(label, label_visibility="collapsed", **kw)


def bb_fig(fig, h, xt="", yt=""):
    ax = dict(gridcolor=GRID, griddash="dot", zeroline=False, linecolor="#777", tickfont=dict(size=10),
              title_font=dict(size=10, color=ORANGE))
    fig.update_layout(paper_bgcolor=BG, plot_bgcolor=BG, height=h, font=dict(family=MONO, color=WHITE, size=10),
                      margin=dict(l=46, r=10, t=8, b=34), showlegend=True,
                      legend=dict(orientation="h", y=1.02, x=0, yanchor="bottom", font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
                      xaxis=dict(title=xt, **ax), yaxis=dict(title=yt, **ax))
    return fig


CFG = {"displayModeBar": False}

# ------------------------------------------------------------------ header
st.session_state.setdefault("tkr", "SPX Index")
now = dt.datetime.now()
hdr = st.empty()

left, main = st.columns([1, 3.7])

# ------------------------------------------------------------------ input panel (label-left, amber fields)
with left:
    ph("SECURITY / CONTRACT", "OVME")
    a, b = st.columns([1.35, 1])
    a.markdown('<div class="fl">Underlying</div>', unsafe_allow_html=True)
    with b:
        st.text_input("Underlying", key="tkr", label_visibility="collapsed")
    kind = st.radio("Type", ["Call", "Put"], horizontal=True, label_visibility="collapsed", key="kind")
    S = field("Spot", "St", value=5000.0, step=10.0, format="%.2f", key="S")
    K = field("Strike", "K", value=5000.0, step=10.0, format="%.2f", key="K")
    days = field("Days to expiry", "T", value=90, min_value=1, step=1, key="days")
    Qo = field("Multiplier", "Qo", value=100.0, step=10.0, format="%.0f", key="Qo")

    ph("PORTFOLIO", "BETA")
    Vs = field("Portfolio value", "Vs", value=60_000_000.0, step=1e6, format="%.0f", key="Vs")
    b_s = field("Current beta", "βs", value=0.95, step=0.01, format="%.4f", key="bs")
    b_t = field("Target beta", "β*", value=1.30, step=0.01, format="%.4f", key="bt")
    xi = field("Leverage factor", "ξ", value=1.00, step=0.05, format="%.2f", key="xi")
    bI = field("Index beta", "βI", value=1.00, step=0.01, format="%.2f", key="bI")

    ph("PRICING / VOL MODEL", "BSM")
    atm = field("ATM implied vol %", "σ", value=16.0, step=0.5, format="%.2f", key="iv") / 100
    r = field("Risk-free rate %", "r", value=4.25, step=0.05, format="%.2f", key="r") / 100
    q = field("Dividend yield %", "q", value=1.30, step=0.05, format="%.2f", key="q") / 100
    skew = field("Skew, pts @10%", "", value=2.0, step=0.5, format="%.2f", key="skew")
    smile = field("Smile, pts @10%", "", value=0.5, step=0.25, format="%.2f", key="smile")
    tslope = field("Term slope, pts/√y", "", value=0.0, step=0.5, format="%.2f", key="tslope")
    src = st.radio("Greeks", ["Model", "Manual"], horizontal=True, label_visibility="collapsed", key="src")
    if src == "Manual":
        mD = field("Delta", "Δo", value=0.50 if kind == "Call" else -0.40, step=0.01, format="%.4f", key="mD")
        mC = field("Premium (pts)", "C", value=180.0, step=1.0, format="%.2f", key="mC")
        mG = field("Gamma", "Γ", value=0.0008, step=0.0001, format="%.5f", key="mG")
        mT = field("Theta (pts/day)", "Θ", value=-1.2, step=0.1, format="%.3f", key="mT")

    ph("SCENARIO / FUTURES", "HEDGE")
    S1 = field("Spot after move", "St+1", value=5200.0, step=10.0, format="%.2f", key="S1")
    F = field("Futures price", "Ft", value=5000.0, step=10.0, format="%.2f", key="F")
    Qf = field("Futures multiplier", "Qf", value=250.0, step=50.0, format="%.0f", key="Qf")
    bF = field("Futures beta", "βf", value=1.00, step=0.01, format="%.2f", key="bF")
    dpw = field("Trading days / week", "", value=5, min_value=1, step=1, key="dpw")
    kstep = field("Chain strike step", "", value=50.0, step=5.0, format="%.0f", key="kstep")

# ------------------------------------------------------------------ core calculations
T = days / 365
sig = float(ivol(K / S, T, atm, skew, smile, tslope))
bsP, bsD, bsG, bsT, bsV = black_scholes(S, K, T, r, q, sig, kind)
if src == "Manual":
    D, C, G, Th, Vg = mD, mC, mG, mT, bsV
else:
    D, C, G, Th, Vg = bsD, bsP, bsG, bsT, bsV

gap = (b_t * xi - b_s) * Vs
per = D * bI * Qo * S
N_raw = gap / per if per else 0.0
N = int(round(N_raw))
wrong = N_raw < 0
added = N * per / Vs
beta_real = b_s + added
prem = N * C * Qo
expo = N * per
lev = abs(expo) / prem if prem else 0.0
omega = D * S / C if C else 0.0

dS = S1 - S
D_new = float(np.clip(D + G * dS, 0, 1) if kind == "Call" else np.clip(D + G * dS, -1, 0))
per_new = D_new * bI * Qo * S1
beta_drift = b_s + N * per_new / Vs
N_new = int(round(gap / per_new)) if per_new else 0
rebal = N_new - N
action = f"BUY {rebal}" if rebal > 0 else (f"SELL {-rebal}" if rebal < 0 else "HOLD")
D_exact = float(black_scholes(S1, K, T, r, q, float(ivol(K / S1, T, atm, skew, smile, tslope)), kind)[1])

th_day = N * Th * Qo
th_week = th_day * dpw
th_pct = abs(th_week) / prem if prem else 0.0
pos_vega = N * Vg * Qo

f_per = bF * F * Qf
Nf_raw = gap / f_per if f_per else 0.0
Nf = int(round(Nf_raw))
beta_fut = b_s + Nf * f_per / Vs
on_tgt = abs(beta_real - b_t * xi) <= 0.005
trade = "CHECK TYPE" if wrong else f"BUY {N:,} {kind.upper()}S"

# ------------------------------------------------------------------ header (filled after inputs)
tkr = st.session_state.tkr
hdr.markdown(f"""
<div class="menu"><span>◀ ▶</span><span class="dd">{tkr} ▾</span><span class="dd">OVME ▾</span><span class="dd">Related Functions Menu ▾</span>
<span class="rt">Message ★ ⚙ ?</span></div>
<div class="sec1"><span class="tk">{tkr.split()[0].upper()}</span><span class="px">{S:,.2f}</span>
<span><span class="l">ATM Vol</span> {atm:.2%}</span><span><span class="l">Rf</span> {r:.2%}</span><span><span class="l">Div</span> {q:.2%}</span>
<span><span class="l">Exp</span> {days}d</span><span><span class="l">As of</span> {now:%d-%b-%Y %H:%M}</span></div>
<div class="redbar"><span>90) Actions ▾</span><span>91) Settings ▾</span><span>92) Export ▾</span><span class="ttl">Starrs Options Lab · Option Overlay</span></div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------ main area
with main:
    t1, t2, t3, t4 = st.tabs(["1) Beta Overlay", "2) OMON Chain", "3) Payoff", "4) Vol Surface"])

    # ============================================================== 1) OVERLAY
    with t1:
        c1, c2 = st.columns([1, 2.15])
        with c1:
            ph("TRADE TICKET", "LIVE CALC")
            table(["Field", "Value"], [
                ("Order", (trade, "n" if wrong else "y")),
                ("Contracts N", (f"{N:,}", "y")),
                ("Strike / Expiry", f"{K:,.0f} / {days}d"),
                ("Exposure gap $", f_money(gap)),
                ("Premium outlay $", f_money(prem)),
                ("Effective exposure $", f_money(expo)),
                ("Realized beta", (f"{beta_real:.4f}", "p" if on_tgt else "n")),
                ("Target beta β*·ξ", f"{b_t * xi:.4f}"),
                ("Leverage", f"{lev:.2f}x"),
                ("After move to " + f"{S1:,.0f}", (action, cls(rebal))),
            ])
            ph("GREEKS", "PER OPT · POSITION")
            table(["Greek", "Per opt", "Position"], [
                ("Price (pts)", f"{C:,.2f}", f_money(prem)),
                ("Delta Δ", f"{D:.4f}", f"{D * N:,.2f}"),
                ("Gamma Γ", f"{G:.6f}", f"{G * N:.4f}"),
                ("Theta /day", (f"{Th:.4f}", "n"), (f_money(th_day), "n")),
                ("Vega /vol pt", f"{Vg:.4f}", f_money(pos_vega)),
                ("Impl vol σ", f"{sig:.2%}", ""),
                ("Elasticity Ω", f"{omega:.2f}", ""),
            ])
            ph("OPTIONS vs FUTURES")
            table(["", "Options", "Futures"], [
                ("Contracts", f"{N:,}", f"{Nf:,}"),
                ("Realized β", f"{beta_real:.4f}", f"{beta_fut:.4f}"),
                ("Cash cost $", f_money(prem), "0"),
                ("Daily carry $", (f_money(th_day), "n"), "0"),
                ("Max loss", f_money(prem), "Unlimited"),
                ("β drifts?", "Yes (Γ)", "No"),
            ])
        with c2:
            ph("CALCULATION  ·  N = (β*ξ − βs)·Vs / (Δo·βI·Qo·St)", "QEPM")
            rows = [
                "LAYER 1  CONTRACTS NEEDED",
                ("1.1", "Exposure gap", ("(β*·ξ − βs) × Vs", "g"), f"({b_t:.4f}×{xi:.2f} − {b_s:.4f}) × {Vs:,.0f}", (f_money(gap), "y")),
                ("1.2", "Exposure / contract", ("Δo × βI × Qo × St", "g"), f"{D:.4f} × {bI:.2f} × {Qo:,.0f} × {S:,.2f}", f_money(per)),
                ("1.3", "Contracts (raw)", ("gap ÷ exp/contract", "g"), f"{gap:,.0f} ÷ {per:,.0f}", f"{N_raw:,.2f}"),
                ("1.4", "Contracts N", ("round(raw)", "g"), "nearest whole contract", (f"{N:,}", "y")),
                "LAYER 2  REALIZED BETA",
                ("2.1", "Beta added", ("N·Δo·βI·Qo·St / Vs", "g"), f"{N:,} × {per:,.0f} ÷ {Vs:,.0f}", f"{added:.4f}"),
                ("2.2", "Realized beta", ("βs + beta added", "g"), f"{b_s:.4f} + {added:.4f}", (f"{beta_real:.4f}", "y")),
                ("2.3", "Rounding error", ("realized − β*·ξ", "g"), f"{beta_real:.4f} − {b_t * xi:.4f}", (f"{beta_real - b_t * xi:+.4f}", cls(beta_real - b_t * xi))),
                "LAYER 3  COST & LEVERAGE",
                ("3.1", "Premium outlay", ("N × C × Qo", "g"), f"{N:,} × {C:,.2f} × {Qo:,.0f}", f_money(prem)),
                ("3.2", "Effective exposure", ("N × exp/contract", "g"), f"{N:,} × {per:,.0f}", f_money(expo)),
                ("3.3", "Leverage", ("|exposure| ÷ premium", "g"), f"{abs(expo):,.0f} ÷ {prem:,.0f}", (f"{lev:.2f}x", "y")),
                ("3.4", "Premium % of Vs", ("premium ÷ Vs", "g"), f"{prem:,.0f} ÷ {Vs:,.0f}", f"{prem / Vs:.2%}"),
                "LAYER 4  ELASTICITY",
                ("4.1", "Elasticity Ω", ("Δo × St ÷ C", "g"), f"{D:.4f} × {S:,.2f} ÷ {C:,.2f}", (f"{omega:.2f}", "y")),
                ("4.2", "Check |Ω| = leverage", ("N, Qo cancel", "g"), f"{abs(omega):.2f} vs {lev:.2f}",
                 ("MATCH", "p") if abs(abs(omega) - lev) < 0.01 else ("CHECK", "n")),
                f"LAYER 5  DELTA DRIFT & REBALANCE   {S:,.0f} → {S1:,.0f}",
                ("5.1", "Index move ΔS", ("St+1 − St", "g"), f"{S1:,.2f} − {S:,.2f}", (f"{dS:+,.2f}", cls(dS))),
                ("5.2", "New delta", ("Δo + Γ × ΔS", "g"), f"{D:.4f} + {G:.6f} × {dS:,.2f}", f"{D_new:.4f}"),
                ("5.3", "New exp/contract", ("Δnew·βI·Qo·St+1", "g"), f"{D_new:.4f} × {bI:.2f} × {Qo:,.0f} × {S1:,.2f}", f_money(per_new)),
                ("5.4", "Drifted beta", ("βs + N·exp_new / Vs", "g"), f"{b_s:.4f} + {N:,} × {per_new:,.0f} ÷ {Vs:,.0f}",
                 (f"{beta_drift:.4f}", cls(beta_drift - b_t * xi))),
                ("5.5", "Contracts needed", ("gap ÷ exp_new", "g"), f"{gap:,.0f} ÷ {per_new:,.0f}", f"{N_new:,}"),
                ("5.6", "Rebalance", ("N_new − N", "g"), f"{N_new:,} − {N:,}", (action, cls(rebal))),
                ("5.7", "Check: exact Δ", ("BSM at St+1", "g"), "full reprice vs Γ approx", f"{D_exact:.4f}"),
                "LAYER 6  THETA & VEGA",
                ("6.1", "Daily theta", ("N × Theta × Qo", "g"), f"{N:,} × {Th:.4f} × {Qo:,.0f}", (f_money(th_day), "n")),
                ("6.2", "Weekly theta", ("daily × days/week", "g"), f"{th_day:,.0f} × {dpw}", (f_money(th_week), "n")),
                ("6.3", "Weekly bleed", ("|weekly| ÷ premium", "g"), f"{abs(th_week):,.0f} ÷ {prem:,.0f}", (f"{th_pct:.2%}", "y")),
                ("6.4", "Position vega", ("N × vega × Qo", "g"), f"{N:,} × {Vg:.4f} × {Qo:,.0f}", f_money(pos_vega)),
                ("6.5", "P&L if vol −5 pts", ("pos vega × −5", "g"), f"{pos_vega:,.0f} × −5", (f_money(-5 * pos_vega), "n")),
                "LAYER 7  FUTURES ALTERNATIVE",
                ("7.1", "Exp / future", ("βf × Ft × Qf", "g"), f"{bF:.2f} × {F:,.2f} × {Qf:,.0f}", f_money(f_per)),
                ("7.2", "Futures Nf", ("gap ÷ exp/future", "g"), f"{gap:,.0f} ÷ {f_per:,.0f} = {Nf_raw:,.2f}", (f"{Nf:,}", "y")),
                ("7.3", "Futures beta", ("βs + Nf·βf·Ft·Qf / Vs", "g"), f"{b_s:.4f} + {Nf:,} × {f_per:,.0f} ÷ {Vs:,.0f}", f"{beta_fut:.4f}"),
            ]
            table(["#", "Description", "Formula", "Calculation", "Result"], rows, cols_left=(0, 1, 2, 3))

            ph("PORTFOLIO BETA vs INDEX LEVEL", "NO REBALANCE")
            lv = np.linspace(S * 0.9, S * 1.1, 61)
            dl = np.clip(D + G * (lv - S), 0, 1) if kind == "Call" else np.clip(D + G * (lv - S), -1, 0)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=lv, y=b_s + N * dl * bI * Qo * lv / Vs, name="Options", mode="lines+markers",
                                     line=dict(color=RED, width=1.5), marker=dict(size=4)))
            fig.add_trace(go.Scatter(x=lv, y=b_s + Nf * bF * F * (lv / S) * Qf / Vs, name="Futures",
                                     line=dict(color=CYAN, width=1.5)))
            fig.add_hline(y=b_t * xi, line_dash="dash", line_color=YELLOW, line_width=1)
            fig.add_vline(x=S, line_dash="dot", line_color="#777", line_width=1)
            st.plotly_chart(bb_fig(fig, 230, "Index level", "Beta"), width="stretch", config=CFG)

    # ============================================================== 2) OMON
    with t2:
        center = round(S / kstep) * kstep
        strikes = center + kstep * np.arange(-10, 11)
        sig_k = ivol(strikes / S, T, atm, skew, smile, tslope)
        cP, cD, cG, cT, cV = black_scholes(S, strikes, T, r, q, sig_k, "Call")
        pP, pD, pG, pT, pV = black_scholes(S, strikes, T, r, q, sig_k, "Put")

        def ba(p):
            hs = np.maximum(0.05, 0.004 * p)
            return np.maximum(p - hs, 0), p + hs

        cB, cA = ba(cP)
        pB, pA = ba(pP)
        ph(f"{tkr.upper()}  OPTION MONITOR  ·  Exp {(now + dt.timedelta(days=int(days))):%d-%b-%y} ({days}d)",
           f"Center {center:,.0f} · Theo BSM")
        hd = ["Bid", "Ask", "Theo", "IVM", "Delta", "Gamma", "Theta", "Vega", "Strike",
              "Bid", "Ask", "Theo", "IVM", "Delta", "Gamma", "Theta", "Vega"]
        html = '<table class="bb"><thead><tr><th colspan="8" style="text-align:center;color:#FB8B1E">CALLS</th><th></th>' \
               '<th colspan="8" style="text-align:center;color:#FB8B1E">PUTS</th></tr><tr>'
        html += "".join(f'<th style="text-align:{"center" if h == "Strike" else "right"}">{h}</th>' for h in hd) + "</tr></thead><tbody>"
        for i, k in enumerate(strikes):
            ci = "itm" if k < S else ""
            pi = "itm" if k > S else ""
            sel = ' style="outline:1px solid #FFD400"' if abs(k - K) < 1e-9 else ""
            trc = ' class="atm"' if k == center else ""
            c_cells = [f"{cB[i]:,.2f}", f"{cA[i]:,.2f}", f"{cP[i]:,.2f}", f"{sig_k[i]:.2%}", f"{cD[i]:.3f}",
                       f"{cG[i]:.5f}", f"{cT[i]:.2f}", f"{cV[i]:.2f}"]
            p_cells = [f"{pB[i]:,.2f}", f"{pA[i]:,.2f}", f"{pP[i]:,.2f}", f"{sig_k[i]:.2%}", f"{pD[i]:.3f}",
                       f"{pG[i]:.5f}", f"{pT[i]:.2f}", f"{pV[i]:.2f}"]
            html += f"<tr{trc}>"
            html += "".join(f'<td class="{ci} {"n" if j == 6 else ("o" if j == 3 else "")}">{v}</td>' for j, v in enumerate(c_cells))
            html += f'<td class="k"{sel}>{k:,.0f}</td>'
            html += "".join(f'<td class="{pi} {"n" if j in (4, 6) else ("o" if j == 3 else "")}">{v}</td>' for j, v in enumerate(p_cells))
            html += "</tr>"
        st.markdown(html + "</tbody></table>", unsafe_allow_html=True)
        st.markdown('<div class="status">Green shading = in the money · orange rule = at the money · yellow box = strike used in the overlay. '
                    'Bid/ask = theoretical value ± 0.4% (min 0.05). <b>Model prices, not live quotes.</b></div>', unsafe_allow_html=True)

    # ============================================================== 3) PAYOFF
    with t3:
        n = int(max(abs(N), 1))
        px = np.linspace(S * 0.7, S * 1.3, 241)
        if kind == "Call":
            exp_pl = (np.maximum(px - K, 0) - C) * Qo * n
            be, mp = K + C, "Unlimited"
        else:
            exp_pl = (np.maximum(K - px, 0) - C) * Qo * n
            be, mp = K - C, f_money((K - C) * Qo * n)
        now_pl = (black_scholes(px, K, T, r, q, ivol(K / px, T, atm, skew, smile, tslope), kind)[0] - C) * Qo * n
        c1, c2 = st.columns([1, 3])
        with c1:
            ph("POSITION", f"{n:,} × {kind.upper()}")
            table(["Field", "Value"], [
                ("Strike", f"{K:,.2f}"), ("Premium (pts)", f"{C:,.2f}"), ("Model theo", f"{bsP:,.2f}"),
                ("Breakeven", (f"{be:,.2f}", "y")), ("Move to BE", f"{be / S - 1:+.2%}"),
                ("Max profit", (mp, "p")), ("Max loss", (f_money(-C * Qo * n), "n")),
                ("Pos delta", f"{D * n:,.2f}"), ("Pos theta /d", (f_money(Th * Qo * n), "n")),
                ("Pos vega", f_money(Vg * Qo * n)),
            ])
            ph("P&L GRID", "AT EXPIRY")
            grid = []
            for mv in (-0.10, -0.05, -0.02, 0.0, 0.02, 0.05, 0.10):
                lvl = S * (1 + mv)
                pl = ((max(lvl - K, 0) if kind == "Call" else max(K - lvl, 0)) - C) * Qo * n
                grid.append((f"{mv:+.0%}", f"{lvl:,.0f}", (f_money(pl), cls(pl))))
            table(["Move", "Index", "P&L $"], grid)
        with c2:
            ph("PAYOFF DIAGRAM", f"BE {be:,.2f}")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=px, y=exp_pl, name="At expiry", line=dict(color=ORANGE, width=2)))
            fig.add_trace(go.Scatter(x=px, y=now_pl, name="Today (BSM)", line=dict(color=CYAN, width=1.5, dash="dot")))
            fig.add_hline(y=0, line_color="#888", line_width=1)
            fig.add_vline(x=S, line_dash="dot", line_color="#888", line_width=1)
            fig.add_vline(x=be, line_dash="dash", line_color=YELLOW, line_width=1)
            st.plotly_chart(bb_fig(fig, 470, "Index level at expiry", "P&L $"), width="stretch", config=CFG)

    # ============================================================== 4) SURFACE
    with t4:
        o1, o2, o3 = st.columns([1.2, 1, 1])
        with o1:
            metric = st.radio("Surface", ["Impl Vol", "Premium", "Delta", "Gamma", "Vega"], horizontal=True,
                              label_visibility="collapsed", key="metric")
        with o2:
            m_pick = field("Term slice: moneyness %", "", value=100.0, step=5.0, format="%.1f", key="mpick")
        with o3:
            t_pick = st.radio("Term", ["1M", "3M", "6M", "1Y", "2Y"], index=1, horizontal=True, label_visibility="collapsed", key="tpick")

        mg = np.linspace(0.70, 1.30, 31)
        tdays = np.array([7, 14, 30, 60, 90, 120, 180, 270, 365, 540, 730])
        tlbl = ["1W", "2W", "1M", "2M", "3M", "4M", "6M", "9M", "1Y", "18M", "2Y"]
        tmap = {"1M": 30, "3M": 90, "6M": 180, "1Y": 365, "2Y": 730}

        def surf_val(m, Ty):
            v = ivol(m, Ty, atm, skew, smile, tslope)
            if metric == "Impl Vol":
                return v * 100
            idx = {"Premium": 0, "Delta": 1, "Gamma": 2, "Vega": 4}[metric]
            return black_scholes(S, m * S, Ty, r, q, v, kind)[idx]

        MM, TT = np.meshgrid(mg, tdays / 365)
        Z = surf_val(MM, TT)
        s1, s2 = st.columns([1.6, 1])
        with s1:
            ph(f"3D SURFACE · {metric.upper()}", "Moneyness × Term")
            fig = go.Figure(go.Surface(x=mg * 100, y=np.arange(len(tdays)), z=Z, showscale=False,
                                       colorscale=[[0, "#1FA34A"], [0.45, "#8FD14F"], [0.7, "#F2C94C"], [1, "#E8452C"]],
                                       contours=dict(x=dict(show=True, color="#1b1b1b"), y=dict(show=True, color="#1b1b1b"))))
            ax = dict(backgroundcolor=BG, gridcolor="#3a3a3a", color="#BBB", tickfont=dict(size=9))
            fig.update_layout(paper_bgcolor=BG, height=470, margin=dict(l=0, r=0, t=0, b=0),
                              font=dict(family=MONO, color=WHITE, size=10),
                              scene=dict(xaxis=dict(title="Moneyness %", nticks=7, **ax),
                                         yaxis=dict(title="Term", tickvals=list(range(len(tdays))), ticktext=tlbl, **ax),
                                         zaxis=dict(title=metric, **ax), aspectmode="manual",
                                         aspectratio=dict(x=1.5, y=1.5, z=0.75),
                                         camera=dict(eye=dict(x=-1.45, y=-1.65, z=0.75))))
            st.plotly_chart(fig, width="stretch", config=CFG)
        with s2:
            ph(f"{metric} vs. Term", f"Moneyness {m_pick:.1f}%")
            ty = surf_val(m_pick / 100, tdays / 365)
            fig = go.Figure(go.Scatter(x=tlbl, y=ty, mode="lines+markers", line=dict(color=RED, width=1.5),
                                       marker=dict(size=6, color=RED), showlegend=False))
            st.plotly_chart(bb_fig(fig, 220), width="stretch", config=CFG)
            ph(f"{metric} vs. Moneyness", f"Term {t_pick}")
            my = surf_val(mg, tmap[t_pick] / 365)
            fig = go.Figure(go.Scatter(x=mg * 100, y=my, mode="lines+markers", line=dict(color=RED, width=1.5),
                                       marker=dict(size=5, color=RED), showlegend=False))
            st.plotly_chart(bb_fig(fig, 220), width="stretch", config=CFG)
        st.markdown('<div class="status">Vol surface is a parametric model (ATM vol + skew + smile + term slope) set in the left panel, '
                    '<b>not market-implied quotes</b>. Set skew and smile to 0 for flat Black-Scholes vol.</div>', unsafe_allow_html=True)

# ------------------------------------------------------------------ footer bars
st.markdown(f"""
<div class="bluebar"><span>300) Edit Panel &nbsp;&nbsp; 301) Expand Panel &nbsp;&nbsp; 302) Export to Excel</span>
<span>STARRS OPTIONS LAB · OVERLAY {trade} · β {beta_real:.4f} → TGT {b_t * xi:.4f}</span></div>
<div class="status">Theoretical pricing · Black-Scholes-Merton with continuous dividend yield · QEPM option-overlay framework · not investment advice</div>
""", unsafe_allow_html=True)
