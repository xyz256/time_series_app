import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Hamilton Ch.2 — Lag Operators",
    page_icon="📉",
    layout="wide",
)

st.title("Time Series Analysis — Chapter 2")
st.markdown("### Lag Operators")
st.markdown(
    "> *Based on Hamilton, J.D. (1994). **Time Series Analysis**. "
    "Princeton University Press.*"
)
st.divider()

tabs = st.tabs([
    "📖 Introduction",
    "§2.1 The Lag Operator",
    "§2.2 First-Order Equations",
    "§2.3 Second-Order Equations",
    "§2.4 pth-Order Equations",
    "🧠 Quiz & Practice",
])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 0 — INTRODUCTION
# ─────────────────────────────────────────────────────────────────────────────
with tabs[0]:
    st.header("Why Lag Operators?")

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown(
            """
Chapter 1 solved difference equations by recursive substitution — writing out
every term explicitly. Chapter 2 introduces a far more compact and powerful
notation: the **lag operator** $L$.

With $L$, the $p$th-order difference equation

$$y_t = \\phi_1 y_{t-1} + \\phi_2 y_{t-2} + \\cdots + \\phi_p y_{t-p} + w_t$$

collapses to a single line of algebra:

$$\\phi(L)\\,y_t = w_t$$

This is not merely notational convenience. Lag-operator algebra allows us to:
- **Invert** difference equations to express $y_t$ as a function of current and
  past $w$'s — the *moving-average representation*
- **Factor** polynomials to decompose complex systems into simple parts
- **Derive** impulse-response functions systematically
- **State** invertibility and stability conditions elegantly
"""
        )

        st.subheader("Connection to Chapter 1")
        st.markdown(
            "Every result from Chapter 1 reappears here — but the lag-operator "
            "approach makes derivations shorter and reveals structure that is "
            "hidden in recursive substitution."
        )

        st.info(
            "**Key insight (Hamilton, p. 25):** Treating $L$ like an ordinary "
            "algebraic symbol and applying familiar rules (power series, factoring, "
            "partial fractions) gives correct and rigorous results."
        )

    with col2:
        st.subheader("The big picture")

        summary = pd.DataFrame({
            "Section": ["§2.1", "§2.2", "§2.3", "§2.4"],
            "Topic": [
                "Definition and algebra of $L$",
                "Inverting a first-order lag polynomial",
                "Factoring and partial fractions — second-order",
                "General $p$th-order; impulse responses",
            ],
            "Core equation": [
                "$L\\,y_t = y_{t-1}$",
                "$(1-\\phi L)^{-1} = \\sum_{j=0}^\\infty \\phi^j L^j$",
                "$(1-m_1 L)(1-m_2 L)\\,y_t = w_t$",
                "$\\phi(L)\\,y_t = w_t$",
            ],
        })
        st.dataframe(summary, hide_index=True, use_container_width=True)

        st.markdown("---")
        st.markdown("**A first glimpse — same model, two notations:**")
        st.latex(r"\text{Chapter 1: } y_t = \phi\,y_{t-1} + w_t")
        st.latex(r"\text{Chapter 2: } (1 - \phi L)\,y_t = w_t")
        st.latex(
            r"\Rightarrow\quad y_t = (1-\phi L)^{-1}w_t"
            r"= \sum_{j=0}^{\infty}\phi^j\,w_{t-j}"
        )
        st.success(
            "The lag-operator inversion recovers exactly the recursive-substitution "
            "solution from Chapter 1 — in two lines instead of a full derivation."
        )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — §2.1 THE LAG OPERATOR
# ─────────────────────────────────────────────────────────────────────────────
with tabs[1]:
    st.header("§2.1  The Lag Operator")

    st.markdown(
        "Hamilton defines the lag operator $L$ as the operator that shifts a "
        "time-series observation back by one period."
    )

    # ── Definition ─────────────────────────────────────────────────────────
    st.subheader("Definition")
    st.latex(r"L\,y_t \;=\; y_{t-1}")
    st.markdown("Applying $L$ repeatedly:")
    st.latex(r"L^k\,y_t \;=\; y_{t-k} \quad (k = 0, 1, 2, \ldots)")
    st.markdown("By convention $L^0 = 1$ (identity), and the **lead operator** $L^{-1}$:")
    st.latex(r"L^{-1}\,y_t \;=\; y_{t+1}")

    # ── Arithmetic ─────────────────────────────────────────────────────────
    st.subheader("Algebra with Lag Operators")

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown("**Linearity**")
        st.latex(r"(aL^i + bL^j)\,y_t = a\,y_{t-i} + b\,y_{t-j}")
        st.markdown("**Multiplication**")
        st.latex(r"L^i \cdot L^j = L^{i+j}")
    with col2:
        st.markdown("**Lag polynomial**")
        st.latex(
            r"\phi(L) = \phi_0 + \phi_1 L + \phi_2 L^2 + \cdots + \phi_p L^p"
        )
        st.markdown("**AR($p$) in compact form**")
        st.latex(
            r"(1 - \phi_1 L - \phi_2 L^2 - \cdots - \phi_p L^p)\,y_t = w_t"
        )
        st.latex(r"\phi(L)\,y_t = w_t")

    # ── Differencing operator ───────────────────────────────────────────────
    st.subheader("The Differencing Operator $\\Delta$")
    st.markdown(
        "One of the most important lag polynomials is the **first-difference operator**:"
    )
    st.latex(r"\Delta \;=\; 1 - L \quad \Rightarrow \quad \Delta\,y_t = y_t - y_{t-1}")
    st.markdown(
        "Higher-order differences are powers of $\\Delta$:"
    )
    st.latex(r"\Delta^2\,y_t = (1-L)^2\,y_t = y_t - 2y_{t-1} + y_{t-2}")
    st.latex(r"\Delta^d\,y_t = (1-L)^d\,y_t")

    st.info(
        "**Why differencing matters:** If $y_t$ has a unit root ($\\phi = 1$), "
        "then $\\Delta y_t = (1-L)y_t$ is stationary. This is the foundation "
        "of the $I(d)$ classification in Chapters 15–18."
    )

    # ── Worked example: GDP levels vs growth ───────────────────────────────
    st.subheader("Worked Example — GDP Levels vs. Growth Rates")
    st.markdown(
        "US GDP levels are non-stationary (unit root). Applying the difference "
        "operator transforms them into stationary growth rates:"
    )

    col_ex1, col_ex2 = st.columns(2, gap="large")

    with col_ex1:
        st.latex(r"\text{Level: } Y_t \quad \xrightarrow{\;\Delta = 1-L\;} \quad \Delta Y_t = Y_t - Y_{t-1}")
        st.markdown(
            "In log terms: $\\Delta \\ln Y_t \\approx$ quarterly growth rate.\n\n"
            "This is why **applied economists almost always work with growth rates** "
            "rather than raw levels — the difference operator removes the unit root."
        )
        st.latex(
            r"\underbrace{Y_t}_{\text{non-stationary}} \xrightarrow{(1-L)}"
            r"\underbrace{\Delta Y_t}_{\text{stationary}}"
        )

    with col_ex2:
        np.random.seed(5)
        T_gdp = 80
        growth = np.random.normal(0.6, 0.8, T_gdp)           # quarterly % growth
        log_gdp = np.cumsum(growth)                           # simulated log-GDP (random walk + drift)

        fig_gdp = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Log GDP level  Y_t  (non-stationary)", "Growth rate  ΔY_t  (stationary)"),
            vertical_spacing=0.18,
        )
        fig_gdp.add_trace(
            go.Scatter(x=list(range(T_gdp)), y=log_gdp.tolist(),
                       mode="lines", line=dict(color="steelblue", width=2)),
            row=1, col=1,
        )
        delta_gdp = np.diff(log_gdp)
        fig_gdp.add_trace(
            go.Scatter(x=list(range(1, T_gdp)), y=delta_gdp.tolist(),
                       mode="lines", line=dict(color="darkorange", width=2)),
            row=2, col=1,
        )
        fig_gdp.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
        fig_gdp.update_layout(
            height=420, showlegend=False,
            margin=dict(l=10, r=10, t=45, b=10),
        )
        fig_gdp.update_yaxes(title_text="log Y_t", row=1, col=1)
        fig_gdp.update_yaxes(title_text="ΔY_t (%)", row=2, col=1)
        st.plotly_chart(fig_gdp, use_container_width=True)

    # ── Interactive: apply lag polynomial ──────────────────────────────────
    st.divider()
    st.subheader("Interactive — Apply a Lag Polynomial to a Series")

    col_i1, col_i2 = st.columns([1, 2], gap="large")
    with col_i1:
        st.markdown("Choose coefficients $a_0 + a_1 L + a_2 L^2$:")
        a0 = st.slider("a₀", -2.0, 2.0, 1.0, 0.1, key="a0_21")
        a1 = st.slider("a₁", -2.0, 2.0, -1.0, 0.1, key="a1_21")
        a2 = st.slider("a₂", -2.0, 2.0, 0.0, 0.1, key="a2_21")
        series_type = st.radio(
            "Input series $y_t$",
            ["Trend (y_t = t)", "Sine wave", "Random walk", "AR(1) φ=0.8"],
            key="ser_21",
        )
        st.latex(
            rf"({a0:.1f} + {a1:.1f}L + {a2:.1f}L^2)\,y_t"
            rf"= {a0:.1f}\,y_t + {a1:.1f}\,y_{{t-1}} + {a2:.1f}\,y_{{t-2}}"
        )

    with col_i2:
        T_la = 60
        t_arr = np.arange(T_la)
        if series_type == "Trend (y_t = t)":
            y_la = t_arr.astype(float)
        elif series_type == "Sine wave":
            y_la = np.sin(2 * np.pi * t_arr / 12)
        elif series_type == "Random walk":
            np.random.seed(1)
            y_la = np.cumsum(np.random.normal(0, 1, T_la))
        else:
            np.random.seed(1)
            y_tmp = np.zeros(T_la)
            eps = np.random.normal(0, 1, T_la)
            for t in range(1, T_la):
                y_tmp[t] = 0.8 * y_tmp[t - 1] + eps[t]
            y_la = y_tmp

        result = np.zeros(T_la)
        for t in range(T_la):
            r = a0 * y_la[t]
            if t >= 1:
                r += a1 * y_la[t - 1]
            if t >= 2:
                r += a2 * y_la[t - 2]
            result[t] = r

        fig_la = go.Figure()
        fig_la.add_trace(go.Scatter(
            x=t_arr.tolist(), y=y_la.tolist(),
            mode="lines", line=dict(color="steelblue", width=2, dash="dot"),
            name="y_t (input)",
        ))
        fig_la.add_trace(go.Scatter(
            x=t_arr.tolist(), y=result.tolist(),
            mode="lines", line=dict(color="crimson", width=2),
            name="φ(L)·y_t (output)",
        ))
        fig_la.add_hline(y=0, line_color="gray", line_dash="dash")
        fig_la.update_layout(
            title="Effect of the lag polynomial on the series",
            xaxis_title="t", yaxis_title="value",
            height=360, margin=dict(l=10, r=10, t=45, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_la, use_container_width=True)
        st.caption(
            "Try a₀=1, a₁=−1, a₂=0 on the Random walk — this is the difference operator Δ, "
            "which removes the trend."
        )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — §2.2 FIRST-ORDER EQUATIONS
# ─────────────────────────────────────────────────────────────────────────────
with tabs[2]:
    st.header("§2.2  First-Order Difference Equations")

    st.markdown(
        "Hamilton revisits the first-order equation from Chapter 1, but now "
        "through the lens of lag-operator inversion."
    )

    # ── Writing in lag-operator form ────────────────────────────────────────
    st.subheader("Writing the Equation with $L$")
    st.latex(r"y_t = \phi\,y_{t-1} + w_t")
    st.markdown("Subtract $\\phi L y_t$ from both sides:")
    st.latex(r"(1 - \phi L)\,y_t = w_t")

    # ── Inverting ──────────────────────────────────────────────────────────
    st.subheader("Inverting the Lag Polynomial")
    st.markdown(
        "We want $y_t = (1-\\phi L)^{-1}\\,w_t$. "
        "Hamilton argues (p. 27) by analogy with the geometric series: "
        "for a scalar $|x| < 1$,"
    )
    st.latex(r"\frac{1}{1-x} = 1 + x + x^2 + \cdots = \sum_{j=0}^{\infty} x^j")
    st.markdown("Replace $x$ with $\\phi L$:")
    st.latex(
        r"(1-\phi L)^{-1} = \sum_{j=0}^{\infty} (\phi L)^j"
        r"= \sum_{j=0}^{\infty} \phi^j L^j"
    )
    st.markdown("Apply to $w_t$:")
    st.latex(
        r"y_t = (1-\phi L)^{-1}\,w_t"
        r"= \sum_{j=0}^{\infty} \phi^j L^j\,w_t"
        r"= \sum_{j=0}^{\infty} \phi^j\,w_{t-j}"
    )

    st.success(
        "This is exactly the Chapter 1 solution $y_t = \\phi^t y_0 + \\sum_{j=0}^{t-1}\\phi^j w_{t-j}$ "
        "(as $t \\to \\infty$ with $|\\phi|<1$ so initial conditions vanish). "
        "Lag-operator inversion delivers it in two lines."
    )

    # ── MA(∞) representation ───────────────────────────────────────────────
    st.subheader("The MA($\\infty$) Representation")
    st.markdown(
        "The inverted form expresses $y_t$ as a weighted sum of current and past "
        "shocks — the **moving-average representation**:"
    )
    st.latex(
        r"y_t = \psi(L)\,w_t, \quad \psi(L) = \sum_{j=0}^{\infty}\psi_j L^j, \quad \psi_j = \phi^j"
    )
    st.markdown(
        "The coefficients $\\psi_j$ are the **impulse-response weights**: "
        "$\\psi_j = \\partial y_t / \\partial w_{t-j}$, "
        "identical to the dynamic multipliers of Chapter 1."
    )

    st.subheader("Convergence Condition")
    st.latex(r"\sum_{j=0}^{\infty}|\psi_j| < \infty \iff |\phi| < 1")
    st.markdown(
        "The inversion is valid — the MA($\\infty$) representation exists — "
        "**if and only if** $|\\phi| < 1$. This is precisely the stability condition "
        "from Proposition 1.1, now phrased as an invertibility condition on $\\phi(L)$."
    )

    # ── Worked example: monetary policy ────────────────────────────────────
    st.subheader("Worked Example — Monetary Policy Transmission")
    st.markdown(
        "Suppose the output gap $\\tilde{y}_t$ responds to a central-bank rate "
        "shock $\\varepsilon_t$ via:"
    )
    st.latex(r"(1 - 0.80L)\,\tilde{y}_t = \varepsilon_t")
    st.markdown("Inverting:")
    st.latex(
        r"\tilde{y}_t = (1-0.80L)^{-1}\varepsilon_t"
        r"= \varepsilon_t + 0.80\,\varepsilon_{t-1} + 0.64\,\varepsilon_{t-2} + \cdots"
    )

    col_mp1, col_mp2 = st.columns(2, gap="large")
    with col_mp1:
        phi_mp = 0.80
        n_mp = 12
        psi_mp = [round(phi_mp**j, 4) for j in range(n_mp)]
        df_mp = pd.DataFrame({
            "Lag  j": list(range(n_mp)),
            "Weight  ψ_j = 0.8^j": psi_mp,
            "% of shock remaining": [f"{100*v:.1f}%" for v in psi_mp],
        })
        st.dataframe(df_mp, hide_index=True, use_container_width=True)
        st.markdown(
            f"- Each dollar of rate shock still has **{100*phi_mp**4:.0f}¢** effect "
            f"after 4 quarters\n"
            f"- Half-life: $-\\ln 2/\\ln 0.8 \\approx {-np.log(2)/np.log(phi_mp):.1f}$ quarters"
        )

    with col_mp2:
        fig_mp = go.Figure()
        fig_mp.add_trace(go.Bar(
            x=list(range(n_mp)), y=psi_mp,
            marker_color="steelblue",
            text=[f"{v:.3f}" for v in psi_mp],
            textposition="outside",
        ))
        fig_mp.update_layout(
            title="MA(∞) weights  ψ_j = 0.8^j — impulse response",
            xaxis_title="Lag  j (quarters)",
            yaxis_title="ψ_j",
            height=340,
            showlegend=False,
            margin=dict(l=10, r=10, t=45, b=20),
        )
        st.plotly_chart(fig_mp, use_container_width=True)

    # ── Interactive ─────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Interactive — Invert $(1 - \\phi L)$")

    col_i1, col_i2 = st.columns([1, 2], gap="large")
    with col_i1:
        phi_22 = st.slider("φ", -0.99, 0.99, 0.75, 0.01, key="phi_22")
        J_22 = st.slider("Show first J weights", 5, 40, 20, key="J_22")
        shock_22 = st.radio(
            "Apply inverse to $w_t$:",
            ["Single impulse", "White noise", "Seasonal sine"],
            key="shock_22",
        )
        T_22 = 80
        if shock_22 == "Single impulse":
            w_22 = np.zeros(T_22); w_22[10] = 1.0
        elif shock_22 == "White noise":
            np.random.seed(3); w_22 = np.random.normal(0, 1, T_22)
        else:
            w_22 = np.sin(2 * np.pi * np.arange(T_22) / 12)

        psi_22 = [phi_22**j for j in range(J_22)]
        lrm = 1 / (1 - phi_22)
        st.latex(rf"(1-{phi_22:.2f}L)^{{-1}} = \sum_{{j=0}}^{{\infty}} {phi_22:.2f}^j L^j")
        st.markdown(f"Long-run multiplier: $1/(1-\\phi) = {lrm:.3f}$")

    with col_i2:
        y_22 = np.zeros(T_22)
        for t in range(1, T_22):
            y_22[t] = phi_22 * y_22[t - 1] + w_22[t]

        fig_22 = make_subplots(
            rows=1, cols=2,
            subplot_titles=(f"Recovered y_t = (1−{phi_22:.2f}L)⁻¹ w_t", "MA weights ψ_j"),
        )
        fig_22.add_trace(
            go.Scatter(x=list(range(T_22)), y=w_22.tolist(),
                       mode="lines", line=dict(color="gray", width=1, dash="dot"), name="w_t"),
            row=1, col=1,
        )
        fig_22.add_trace(
            go.Scatter(x=list(range(T_22)), y=y_22.tolist(),
                       mode="lines", line=dict(color="royalblue", width=2), name="y_t"),
            row=1, col=1,
        )
        fig_22.add_trace(
            go.Bar(x=list(range(J_22)), y=psi_22,
                   marker_color=["steelblue" if v >= 0 else "crimson" for v in psi_22]),
            row=1, col=2,
        )
        fig_22.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)
        fig_22.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=2)
        fig_22.update_layout(
            height=380, showlegend=False,
            margin=dict(l=10, r=10, t=45, b=20),
        )
        fig_22.update_xaxes(title_text="t", row=1, col=1)
        fig_22.update_xaxes(title_text="j", row=1, col=2)
        st.plotly_chart(fig_22, use_container_width=True)
        st.caption("Dotted gray = input w_t. Blue = output y_t after applying inverse filter.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — §2.3 SECOND-ORDER EQUATIONS
# ─────────────────────────────────────────────────────────────────────────────
with tabs[3]:
    st.header("§2.3  Second-Order Difference Equations")

    st.markdown(
        "For second-order equations, Hamilton introduces **factoring** and "
        "**partial fractions** — the same techniques used for rational functions "
        "in calculus, applied to lag polynomials."
    )

    # ── Setup ──────────────────────────────────────────────────────────────
    st.subheader("Setup")
    st.latex(r"(1 - \phi_1 L - \phi_2 L^2)\,y_t = w_t")

    # ── Factoring ──────────────────────────────────────────────────────────
    st.subheader("Factoring the Lag Polynomial")
    st.markdown(
        "The quadratic $1 - \\phi_1 z - \\phi_2 z^2$ has roots $z_1, z_2$ satisfying:"
    )
    st.latex(r"1 - \phi_1 z - \phi_2 z^2 = -\phi_2(z - z_1)(z - z_2)")
    st.markdown(
        "In terms of $m_i = 1/z_i$ (the **reciprocal roots**, i.e. eigenvalues of $\\mathbf{F}$):"
    )
    st.latex(r"1 - \phi_1 L - \phi_2 L^2 = (1 - m_1 L)(1 - m_2 L)")
    st.markdown("where $m_1$ and $m_2$ satisfy:")
    st.latex(r"m_1 + m_2 = \phi_1, \qquad m_1 m_2 = -\phi_2")

    # ── Partial fractions ──────────────────────────────────────────────────
    st.subheader("Partial Fraction Decomposition")
    st.markdown("When $m_1 \\neq m_2$, the inverse factors via partial fractions:")
    st.latex(
        r"\frac{1}{(1-m_1 L)(1-m_2 L)} = \frac{A}{1-m_1 L} + \frac{B}{1-m_2 L}"
    )
    st.markdown("where:")
    st.latex(
        r"A = \frac{m_1}{m_1 - m_2}, \qquad B = \frac{-m_2}{m_1 - m_2}"
    )
    st.markdown("Applying each factor's inverse:")
    st.latex(
        r"y_t = A\sum_{j=0}^{\infty}m_1^j\,w_{t-j} \;+\; B\sum_{j=0}^{\infty}m_2^j\,w_{t-j}"
    )
    st.latex(
        r"\psi_j = \frac{m_1^{j+1} - m_2^{j+1}}{m_1 - m_2} \quad (j=0,1,2,\ldots)"
    )

    # ── Real vs complex roots ──────────────────────────────────────────────
    st.subheader("Real vs. Complex Roots — Two Types of Dynamics")

    col_a, col_b = st.columns(2, gap="large")
    with col_a:
        st.markdown("#### Real distinct roots ($m_1, m_2 \\in \\mathbb{R}$)")
        st.markdown(
            "- Impulse response is a **sum of two decaying exponentials**\n"
            "- Monotone convergence or convergent overshooting depending on signs\n"
            "- Stable iff $|m_1| < 1$ and $|m_2| < 1$"
        )
        st.latex(r"\psi_j = A\,m_1^j + B\,m_2^j")

    with col_b:
        st.markdown("#### Complex conjugate roots ($m_{1,2} = r e^{\\pm i\\omega}$)")
        st.markdown(
            "- Arise when discriminant $\\phi_1^2 + 4\\phi_2 < 0$\n"
            "- Impulse response **oscillates with frequency** $\\omega$ "
            "and decays at rate $r$\n"
            "- Stable iff $|m| = r < 1$"
        )
        st.latex(r"\psi_j = r^j \cdot \frac{\sin[(j+1)\omega]}{\sin\,\omega}")

    # ── Worked example: business cycles ────────────────────────────────────
    st.subheader("Worked Example — Business Cycles (Yule, 1927)")
    st.markdown(
        "G. Udny Yule showed that an AR(2) with complex roots produces the "
        "regular oscillations typical of business cycles:"
    )
    st.latex(r"(1 - 1.5L + 0.9L^2)\,y_t = w_t \quad \Rightarrow \quad \phi_1 = 1.5,\;\phi_2 = -0.9")

    col_bc1, col_bc2 = st.columns(2, gap="large")
    with col_bc1:
        phi1_bc = 1.5
        phi2_bc = -0.9
        disc = phi1_bc**2 + 4 * phi2_bc
        r_bc = np.sqrt(-phi2_bc)
        omega_bc = np.arccos(phi1_bc / (2 * r_bc))
        period_bc = 2 * np.pi / omega_bc

        st.markdown("**Characteristic analysis:**")
        st.latex(rf"\phi_1^2 + 4\phi_2 = {disc:.2f} < 0 \quad \Rightarrow \text{{complex roots}}")
        st.latex(rf"r = \sqrt{{-\phi_2}} = \sqrt{{0.9}} \approx {r_bc:.4f}")
        st.latex(rf"\omega = \arccos\!\left(\frac{{\phi_1}}{{2r}}\right) \approx {omega_bc:.4f} \text{{ rad/period}}")
        st.latex(rf"\text{{Cycle length}} = \frac{{2\pi}}{{\omega}} \approx {period_bc:.1f} \text{{ periods}}")
        st.success(
            f"r = {r_bc:.4f} < 1 → **stable damped oscillation**. "
            f"Peaks repeat every ≈{period_bc:.0f} periods."
        )

        n_irf = 30
        psi_bc = np.zeros(n_irf)
        for j in range(n_irf):
            psi_bc[j] = r_bc**j * np.sin((j + 1) * omega_bc) / np.sin(omega_bc)

        df_bc = pd.DataFrame({
            "j": list(range(8)),
            "ψ_j": [round(psi_bc[j], 4) for j in range(8)],
        })
        st.dataframe(df_bc, hide_index=True, use_container_width=True)

    with col_bc2:
        np.random.seed(10)
        T_bc = 100
        w_bc = np.random.normal(0, 1, T_bc)
        y_bc = np.zeros(T_bc)
        for t in range(2, T_bc):
            y_bc[t] = phi1_bc * y_bc[t-1] + phi2_bc * y_bc[t-2] + w_bc[t]

        fig_bc = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Sample path — damped business cycles", "Impulse response ψ_j"),
            vertical_spacing=0.18,
        )
        fig_bc.add_trace(
            go.Scatter(x=list(range(T_bc)), y=y_bc.tolist(),
                       mode="lines", line=dict(color="royalblue", width=2)),
            row=1, col=1,
        )
        fig_bc.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)
        colors_irf = ["steelblue" if v >= 0 else "crimson" for v in psi_bc]
        fig_bc.add_trace(
            go.Bar(x=list(range(n_irf)), y=psi_bc.tolist(), marker_color=colors_irf),
            row=2, col=1,
        )
        fig_bc.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
        fig_bc.update_layout(
            height=500, showlegend=False,
            margin=dict(l=10, r=10, t=45, b=10),
        )
        st.plotly_chart(fig_bc, use_container_width=True)

    # ── Interactive ─────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Interactive — Second-Order Factoring Explorer")

    col_i1, col_i2 = st.columns([1, 2], gap="large")
    with col_i1:
        phi1_23 = st.slider("φ₁", -2.0, 2.0, 1.2, 0.05, key="phi1_23")
        phi2_23 = st.slider("φ₂", -1.5, 0.0, -0.5, 0.05, key="phi2_23")
        T_23 = st.slider("Periods T", 20, 120, 80, key="T_23")

        F_23 = np.array([[phi1_23, phi2_23], [1.0, 0.0]])
        eigs_23 = np.linalg.eigvals(F_23)
        stable_23 = all(abs(e) < 1 for e in eigs_23)
        disc_23 = phi1_23**2 + 4 * phi2_23
        root_type = "Complex conjugate" if disc_23 < 0 else "Real distinct" if disc_23 > 0 else "Repeated"

        st.markdown(f"**Root type:** {root_type}")
        for i, e in enumerate(eigs_23):
            sign = "+" if e.imag >= 0 else "−"
            st.latex(rf"m_{{{i+1}}} = {e.real:.3f} {sign} {abs(e.imag):.3f}i,\; |m| = {abs(e):.3f}")

        if disc_23 < 0:
            r_23 = np.sqrt(-phi2_23)
            omega_23 = np.arccos(np.clip(phi1_23 / (2 * r_23), -1, 1))
            period_23 = 2 * np.pi / omega_23
            st.markdown(f"Cycle length ≈ **{period_23:.1f}** periods, amplitude decay r = {r_23:.3f}")

        if stable_23:
            st.success("Stable")
        else:
            st.error("Unstable")

    with col_i2:
        np.random.seed(42)
        w_23 = np.zeros(T_23); w_23[5] = 1.0   # impulse
        y_23 = np.zeros(T_23)
        for t in range(2, T_23):
            y_23[t] = phi1_23 * y_23[t-1] + phi2_23 * y_23[t-2] + w_23[t]

        theta_uc = np.linspace(0, 2*np.pi, 300)

        fig_23 = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Impulse response (unit shock at t=5)", "Roots on Unit Circle"),
        )
        fig_23.add_trace(
            go.Scatter(x=list(range(T_23)), y=y_23.tolist(),
                       mode="lines", line=dict(color="royalblue", width=2)),
            row=1, col=1,
        )
        fig_23.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)
        fig_23.add_trace(
            go.Scatter(x=np.cos(theta_uc).tolist(), y=np.sin(theta_uc).tolist(),
                       mode="lines", line=dict(color="gray", dash="dot")),
            row=1, col=2,
        )
        fig_23.add_trace(
            go.Scatter(
                x=[e.real for e in eigs_23], y=[e.imag for e in eigs_23],
                mode="markers",
                marker=dict(
                    color=["green" if abs(e) < 1 else "red" for e in eigs_23],
                    size=14, symbol="x", line=dict(width=2),
                ),
            ),
            row=1, col=2,
        )
        fig_23.add_hline(y=0, line_color="lightgray", row=1, col=2)
        fig_23.add_vline(x=0, line_color="lightgray", row=1, col=2)
        fig_23.update_layout(
            height=400, showlegend=False,
            margin=dict(l=10, r=10, t=45, b=20),
        )
        fig_23.update_xaxes(title_text="t", row=1, col=1)
        fig_23.update_xaxes(title_text="Real", range=[-1.8, 1.8], row=1, col=2)
        fig_23.update_yaxes(title_text="y_t", row=1, col=1)
        fig_23.update_yaxes(
            title_text="Imaginary", range=[-1.8, 1.8],
            scaleanchor="x2", scaleratio=1, row=1, col=2,
        )
        st.plotly_chart(fig_23, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — §2.4 pTH-ORDER EQUATIONS
# ─────────────────────────────────────────────────────────────────────────────
with tabs[4]:
    st.header("§2.4  $p$th-Order Difference Equations")

    st.markdown(
        "Hamilton now consolidates everything into the general $p$th-order case, "
        "introducing the **impulse-response function** and the general "
        "invertibility condition."
    )

    # ── General form ───────────────────────────────────────────────────────
    st.subheader("General Lag-Polynomial Form")
    st.latex(r"\phi(L)\,y_t = w_t")
    st.latex(
        r"\phi(L) = 1 - \phi_1 L - \phi_2 L^2 - \cdots - \phi_p L^p"
    )
    st.markdown("Factoring over the $p$ reciprocal roots $m_1, \ldots, m_p$:")
    st.latex(
        r"\phi(L) = (1 - m_1 L)(1 - m_2 L)\cdots(1 - m_p L)"
    )

    # ── Invertibility ──────────────────────────────────────────────────────
    st.subheader("Invertibility Condition")
    st.success(
        "**Hamilton Proposition 2.1** — The lag polynomial $\\phi(L)$ is **invertible** "
        "(the MA($\\infty$) representation exists) if and only if all roots $m_i$ of "
        "$\\phi(L)$ satisfy $|m_i| < 1$, equivalently all roots $z_i$ of "
        "$\\phi(z) = 0$ satisfy $|z_i| > 1$."
    )
    st.markdown("When invertible:")
    st.latex(
        r"y_t = [\phi(L)]^{-1}\,w_t = \psi(L)\,w_t = \sum_{j=0}^{\infty}\psi_j\,w_{t-j}"
    )

    # ── Impulse response function ───────────────────────────────────────────
    st.subheader("Impulse-Response Function (IRF)")
    st.markdown(
        "The **impulse-response weights** $\\psi_j$ measure the dynamic effect "
        "of a unit shock in $w_t$ on $y_{t+j}$:"
    )
    st.latex(r"\psi_j = \frac{\partial\,y_{t+j}}{\partial\,w_t}")
    st.markdown(
        "They satisfy the same difference equation as $y_t$ itself, with initial condition $\\psi_0 = 1$:"
    )
    st.latex(
        r"\psi_j = \phi_1\psi_{j-1} + \phi_2\psi_{j-2} + \cdots + \phi_p\psi_{j-p}, \quad j \geq 1"
    )

    # ── ARMA extension ─────────────────────────────────────────────────────
    st.subheader("Extension: ARMA($p$, $q$) Models")
    st.markdown(
        "Chapter 2 also lays the groundwork for ARMA models, where shocks "
        "enter through a lag polynomial on the right-hand side too:"
    )
    st.latex(r"\phi(L)\,y_t = \theta(L)\,w_t")
    st.latex(
        r"\phi(L) = 1 - \phi_1 L - \cdots - \phi_p L^p, \quad"
        r"\theta(L) = 1 + \theta_1 L + \cdots + \theta_q L^q"
    )
    st.markdown("The MA($\\infty$) representation is:")
    st.latex(r"y_t = \frac{\theta(L)}{\phi(L)}\,w_t = \psi(L)\,w_t")
    st.info(
        "Invertibility requires $|m_i| < 1$ for all roots of $\\phi(L)$. "
        "Identifiability requires $|n_i| < 1$ for all roots of $\\theta(L)$ — "
        "the **invertibility of the MA part**, discussed further in Chapter 3."
    )

    # ── Worked example: ARMA(1,1) inflation ────────────────────────────────
    st.subheader("Worked Example — ARMA(1,1) Inflation Dynamics")
    st.markdown(
        "Inflation $\\pi_t$ is often modelled as an ARMA(1,1) — persistence "
        "from the AR part, short-lived cost-push shocks from the MA part:"
    )
    st.latex(r"(1 - 0.7L)\,\pi_t = (1 + 0.4L)\,\varepsilon_t")
    st.markdown("MA($\\infty$) weights come from expanding $\\theta(L)/\\phi(L)$:")
    st.latex(
        r"\psi(L) = \frac{1 + 0.4L}{1 - 0.7L}"
        r"= (1 + 0.4L)\sum_{j=0}^{\infty}0.7^j L^j"
    )
    st.latex(
        r"\psi_0 = 1, \quad \psi_j = 0.7^j + 0.4\cdot 0.7^{j-1} = 0.7^{j-1}(0.7 + 0.4)"
        r"= 1.1\cdot 0.7^{j-1} \quad (j \geq 1)"
    )

    col_ar1, col_ar2 = st.columns(2, gap="large")
    with col_ar1:
        phi_arma = 0.7
        theta_arma = 0.4
        n_arma = 12
        psi_arma = [1.0] + [1.1 * phi_arma**(j - 1) for j in range(1, n_arma)]
        df_arma = pd.DataFrame({
            "j": list(range(n_arma)),
            "ψ_j (ARMA 1,1)": [round(v, 4) for v in psi_arma],
            "ψ_j (AR(1) only)": [round(phi_arma**j, 4) for j in range(n_arma)],
        })
        st.dataframe(df_arma, hide_index=True, use_container_width=True)
        st.markdown(
            "The MA term boosts the initial response ($\\psi_1 > \\phi$) "
            "then both decay at the same AR(1) rate of 0.7."
        )

    with col_ar2:
        psi_ar_only = [phi_arma**j for j in range(n_arma)]
        fig_arma = go.Figure()
        fig_arma.add_trace(go.Bar(
            x=list(range(n_arma)), y=psi_arma,
            name="ARMA(1,1)", marker_color="steelblue",
            offsetgroup=0,
        ))
        fig_arma.add_trace(go.Bar(
            x=list(range(n_arma)), y=psi_ar_only,
            name="AR(1) only", marker_color="lightcoral",
            offsetgroup=1,
        ))
        fig_arma.update_layout(
            title="Impulse response: ARMA(1,1) vs. pure AR(1)",
            xaxis_title="Lag  j",
            yaxis_title="ψ_j",
            barmode="group",
            height=340,
            margin=dict(l=10, r=10, t=45, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_arma, use_container_width=True)

    # ── Interactive IRF ─────────────────────────────────────────────────────
    st.divider()
    st.subheader("Interactive — General $p$th-Order IRF")

    col_g1, col_g2 = st.columns([1, 2], gap="large")
    with col_g1:
        p_24 = st.slider("Order p", 1, 4, 2, key="p_24")
        phi_24 = []
        for i in range(p_24):
            default = [0.6, 0.2, 0.1, 0.05][i]
            v = st.slider(f"φ_{i+1}", -2.0, 2.0, default, 0.05, key=f"phi24_{i}")
            phi_24.append(v)
        J_24 = st.slider("IRF horizon (periods)", 10, 60, 30, key="J_24")

        F_24 = np.zeros((p_24, p_24))
        F_24[0, :] = phi_24
        for i in range(1, p_24):
            F_24[i, i - 1] = 1.0
        eigs_24 = np.linalg.eigvals(F_24)
        stable_24 = all(abs(e) < 1 for e in eigs_24)
        if stable_24:
            st.success("Invertible — MA(∞) representation exists")
        else:
            st.error("Not invertible — process is explosive")

    with col_g2:
        psi_24 = np.zeros(J_24)
        psi_24[0] = 1.0
        for j in range(1, J_24):
            for k in range(min(j, p_24)):
                psi_24[j] += phi_24[k] * psi_24[j - k - 1]

        cum_irf = np.cumsum(psi_24)

        fig_24 = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Impulse Response  ψ_j", "Cumulative IRF  Σψ_j"),
        )
        bar_colors = ["steelblue" if v >= 0 else "crimson" for v in psi_24]
        fig_24.add_trace(
            go.Bar(x=list(range(J_24)), y=psi_24.tolist(), marker_color=bar_colors),
            row=1, col=1,
        )
        fig_24.add_trace(
            go.Scatter(x=list(range(J_24)), y=cum_irf.tolist(),
                       mode="lines+markers", line=dict(color="darkorange", width=2)),
            row=1, col=2,
        )
        if stable_24:
            lrm_24 = 1 / (1 - sum(phi_24))
            fig_24.add_hline(
                y=lrm_24, line_dash="dash", line_color="green",
                annotation_text=f"LRM = {lrm_24:.2f}",
                annotation_position="right",
                row=1, col=2,
            )
        fig_24.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)
        fig_24.update_layout(
            height=380, showlegend=False,
            margin=dict(l=10, r=10, t=45, b=20),
        )
        fig_24.update_xaxes(title_text="j", row=1, col=1)
        fig_24.update_xaxes(title_text="j", row=1, col=2)
        fig_24.update_yaxes(title_text="ψ_j", row=1, col=1)
        fig_24.update_yaxes(title_text="Σψ_j", row=1, col=2)
        st.plotly_chart(fig_24, use_container_width=True)
        if stable_24:
            st.markdown(
                f"Long-run multiplier $= 1/(1-\\sum\\phi_i) = {lrm_24:.3f}$ — "
                "the cumulative effect of a permanent shock."
            )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — QUIZ & PRACTICE
# ─────────────────────────────────────────────────────────────────────────────
with tabs[5]:
    st.header("Quiz & Practice")
    st.markdown("Test your understanding of Chapter 2 — Lag Operators.")

    # ── Q1 ─────────────────────────────────────────────────────────────────
    st.subheader("Question 1 — Lag Operator Basics")
    st.markdown("If $L^3 y_t = y_{t-3}$, what does $(1 - L)^2 y_t$ equal?")
    q1 = st.radio(
        "Choose the correct expression:",
        [
            "$y_t - y_{t-2}$",
            "$y_t - 2y_{t-1} + y_{t-2}$",
            "$y_t + 2y_{t-1} + y_{t-2}$",
            "$y_{t-2}$",
        ],
        index=None, key="q2_1",
    )
    if q1 is not None:
        if q1 == "$y_t - 2y_{t-1} + y_{t-2}$":
            st.success(
                r"✅ Correct! $(1-L)^2 = 1 - 2L + L^2$, so "
                r"$(1-L)^2 y_t = y_t - 2y_{t-1} + y_{t-2} = \Delta^2 y_t$."
            )
        else:
            st.error(
                r"❌ Expand $(1-L)^2 = 1 - 2L + L^2$. Applying to $y_t$: "
                r"$y_t - 2y_{t-1} + y_{t-2}$."
            )

    st.divider()

    # ── Q2 ─────────────────────────────────────────────────────────────────
    st.subheader("Question 2 — Inverting a Lag Polynomial")
    st.markdown("What are the MA($\\infty$) weights $\\psi_j$ for $(1 - 0.5L)^{-1}$?")
    q2 = st.radio(
        "Select:",
        [
            "$\\psi_j = j \\times 0.5$",
            "$\\psi_j = 0.5^j$",
            "$\\psi_j = (-0.5)^j$",
            "$\\psi_j = 1/(1 - 0.5j)$",
        ],
        index=None, key="q2_2",
    )
    if q2 is not None:
        if q2 == "$\\psi_j = 0.5^j$":
            st.success(
                r"✅ Correct! $(1-0.5L)^{-1} = \sum_{j=0}^\infty 0.5^j L^j$, so $\psi_j = 0.5^j$."
            )
        else:
            st.error(
                r"❌ The geometric series gives $(1-\phi L)^{-1} = \sum_j \phi^j L^j$, "
                r"so $\psi_j = 0.5^j$."
            )

    st.divider()

    # ── Q3 ─────────────────────────────────────────────────────────────────
    st.subheader("Question 3 — Factoring and Roots")
    st.markdown(
        "The lag polynomial $(1 - 1.2L + 0.4L^2)$ has roots $m_1, m_2$. "
        "Which condition guarantees invertibility?"
    )
    q3 = st.radio(
        "Invertibility requires:",
        [
            "$m_1 + m_2 > 1$",
            "$|m_1| < 1$ and $|m_2| < 1$",
            "$m_1 \\times m_2 < 0$",
            "$|m_1| > 1$ and $|m_2| > 1$",
        ],
        index=None, key="q2_3",
    )
    if q3 is not None:
        if q3 == "$|m_1| < 1$ and $|m_2| < 1$":
            F_q3 = np.array([[1.2, -0.4], [1.0, 0.0]])
            eigs_q3 = np.linalg.eigvals(F_q3)
            st.success(
                f"✅ Correct! Invertibility requires all reciprocal roots (eigenvalues) "
                f"inside the unit circle. Here: |m₁| = {abs(eigs_q3[0]):.3f}, "
                f"|m₂| = {abs(eigs_q3[1]):.3f} → "
                f"{'invertible' if all(abs(e)<1 for e in eigs_q3) else 'not invertible'}."
            )
        else:
            st.error(
                "❌ Invertibility of $\\phi(L)$ requires all reciprocal roots $m_i$ to "
                "satisfy $|m_i| < 1$ — same as the stability condition on $\\mathbf{F}$."
            )

    st.divider()

    # ── Q4 ─────────────────────────────────────────────────────────────────
    st.subheader("Question 4 — Impulse Response")
    st.markdown(
        "For $\\phi(L)\\,y_t = w_t$ with $\\phi_1 = 0.6,\\,\\phi_2 = 0.3$, "
        "the IRF satisfies $\\psi_j = \\phi_1\\psi_{j-1} + \\phi_2\\psi_{j-2}$. "
        "Given $\\psi_0 = 1$, $\\psi_1 = 0.6$, what is $\\psi_2$?"
    )
    q4 = st.radio(
        "$\\psi_2 =$",
        ["0.36", "0.66", "0.9", "0.18"],
        index=None, key="q2_4",
    )
    if q4 is not None:
        psi2_correct = 0.6 * 0.6 + 0.3 * 1.0
        if q4 == "0.66":
            st.success(
                rf"✅ Correct! $\psi_2 = 0.6\times\psi_1 + 0.3\times\psi_0 "
                rf"= 0.6\times 0.6 + 0.3\times 1 = {psi2_correct:.2f}$."
            )
        else:
            st.error(
                rf"❌ $\psi_2 = \phi_1\psi_1 + \phi_2\psi_0 "
                rf"= 0.6\times 0.6 + 0.3\times 1.0 = {psi2_correct:.2f}$."
            )

    st.divider()

    # ── Summary ─────────────────────────────────────────────────────────────
    st.subheader("Chapter 2 — Key Results at a Glance")
    st.markdown("""
| Concept | Formula / Condition |
|---|---|
| Lag operator | $L\\,y_t = y_{t-1}$, $L^k y_t = y_{t-k}$ |
| Difference operator | $\\Delta = 1-L$, $\\Delta^d = (1-L)^d$ |
| AR($p$) compact form | $\\phi(L)\\,y_t = w_t$ |
| Inversion (1st order) | $(1-\\phi L)^{-1} = \\sum_{j=0}^\\infty \\phi^j L^j$ (need $|\\phi|<1$) |
| Factoring (2nd order) | $(1-\\phi_1 L - \\phi_2 L^2) = (1-m_1 L)(1-m_2 L)$ |
| Root relations | $m_1+m_2 = \\phi_1$, $m_1 m_2 = -\\phi_2$ |
| Complex roots | $m = r\\,e^{\\pm i\\omega}$ → damped oscillation at frequency $\\omega$ |
| Invertibility | All reciprocal roots $|m_i| < 1$ |
| Impulse response | $\\psi_j = \\phi_1\\psi_{j-1}+\\cdots+\\phi_p\\psi_{j-p}$, $\\psi_0=1$ |
| Long-run multiplier | $\\sum_{j=0}^\\infty\\psi_j = [\\phi(1)]^{-1} = 1/(1-\\sum\\phi_i)$ |
| ARMA($p$,$q$) | $\\phi(L)\\,y_t = \\theta(L)\\,w_t$, IRF $= \\theta(L)/\\phi(L)$ |
""")
    st.markdown(
        "*Continue to Chapter 3 — Stationary ARMA Processes — to see how "
        "lag-operator algebra underpins autocorrelation and spectral analysis.*"
    )
