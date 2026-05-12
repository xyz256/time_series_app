import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Hamilton Ch.1 — Difference Equations",
    page_icon="📈",
    layout="wide",
)

st.title("Time Series Analysis — Chapter 1")
st.markdown("### Difference Equations")
st.markdown(
    "> *Based on Hamilton, J.D. (1994). **Time Series Analysis**. "
    "Princeton University Press.*"
)
st.divider()

tabs = st.tabs(
    [
        "📖 Introduction",
        "§1.1 First-Order Equations",
        "§1.2 pth-Order Equations",
        "§1.3 Initial Conditions & Stability",
        "🧠 Quiz & Practice",
    ]
)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 0 — INTRODUCTION
# ─────────────────────────────────────────────────────────────────────────────
with tabs[0]:
    st.header("What Is a Difference Equation?")

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown(
            """
A **difference equation** (or *recurrence relation*) expresses the current value of a
variable as a function of its own past values and an external forcing term.

They are the fundamental building block of time series econometrics. Virtually every
model encountered in macroeconomics and finance — AR, VAR, ARMA, unit-root processes —
can be understood through the lens of Chapter 1.
"""
        )

        st.subheader("General $p$th-order form")
        st.latex(
            r"y_t = \phi_1 y_{t-1} + \phi_2 y_{t-2} + \cdots + \phi_p y_{t-p} + w_t"
        )

        st.markdown(
            """
**Notation used throughout (Hamilton, Ch. 1):**

| Symbol | Meaning |
|--------|---------|
| $y_t$ | variable of interest at time $t$ |
| $\\phi_1, \\ldots, \\phi_p$ | autoregressive coefficients |
| $w_t$ | exogenous input / innovation |
| $p$ | order of the equation |
"""
        )

        st.subheader("What you will learn")
        st.markdown(
            """
- How to **solve** a difference equation by recursive substitution
- The concept of **dynamic multipliers** — how a shock today propagates forward
- The **stability condition** — when does the process remain bounded?
- The **companion matrix** representation for higher-order systems
- The role of **initial conditions**
"""
        )

    with col2:
        st.subheader("A quick example")
        st.markdown("Suppose quarterly output $y_t$ follows:")
        st.latex(r"y_t = 0.8\, y_{t-1} + w_t")

        np.random.seed(42)
        T = 60
        w = np.random.normal(0, 1, T)
        y = np.zeros(T)
        for t in range(1, T):
            y[t] = 0.8 * y[t - 1] + w[t]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=list(range(T)),
                y=y,
                mode="lines",
                line=dict(color="royalblue", width=2),
                name="y_t",
            )
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.update_layout(
            title="Sample path: φ = 0.8, white-noise w_t",
            xaxis_title="t",
            yaxis_title="y_t",
            height=320,
            margin=dict(l=10, r=10, t=40, b=30),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.info(
            "Because |φ| = 0.8 < 1 the series is **stable**: shocks eventually die out "
            "and the series does not explode. This is the central stability result of Ch. 1."
        )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — §1.1 FIRST-ORDER DIFFERENCE EQUATIONS
# ─────────────────────────────────────────────────────────────────────────────
with tabs[1]:
    st.header("§1.1 First-Order Difference Equations")

    st.markdown(
        "Hamilton opens Chapter 1 with the simplest case — a single lag — "
        "because every concept generalises directly to higher orders."
    )

    # ── Model ──────────────────────────────────────────────────────────────
    st.subheader("The Model")
    st.latex(r"y_t = \phi \cdot y_{t-1} + w_t, \quad t = 1, 2, 3, \ldots")
    st.markdown(
        "where $\\phi \\in \\mathbb{R}$ is the single autoregressive coefficient "
        "and $w_t$ is an exogenous forcing sequence."
    )

    # ── Solution ───────────────────────────────────────────────────────────
    st.subheader("Solution by Recursive Substitution")
    st.markdown(
        "Hamilton obtains the explicit solution by repeatedly back-substituting "
        "(p. 5 of the text):"
    )
    st.latex(
        r"y_t = \phi^t \cdot y_0 + \sum_{j=0}^{t-1} \phi^j \cdot w_{t-j}"
    )

    with st.expander("Show derivation step by step"):
        st.markdown("**Step 1** — write the equation at time $t$:")
        st.latex(r"y_t = \phi\, y_{t-1} + w_t")

        st.markdown("**Step 2** — substitute $y_{t-1} = \\phi\\, y_{t-2} + w_{t-1}$:")
        st.latex(
            r"y_t = \phi\bigl(\phi\, y_{t-2} + w_{t-1}\bigr) + w_t"
            r"     = \phi^2 y_{t-2} + \phi\, w_{t-1} + w_t"
        )

        st.markdown("**Step 3** — continue back to $y_0$:")
        st.latex(
            r"y_t = \phi^t y_0 + \phi^{t-1} w_1 + \phi^{t-2} w_2 + \cdots + \phi\, w_{t-1} + w_t"
        )
        st.latex(r"y_t = \phi^t y_0 + \sum_{j=0}^{t-1} \phi^j\, w_{t-j}")

    # ── Dynamic Multipliers ────────────────────────────────────────────────
    st.subheader("Dynamic Multipliers")
    st.markdown(
        "Hamilton defines the **dynamic multiplier** as the response of $y_{t+j}$ "
        "to a unit impulse in $w_t$, holding all other $w$'s fixed:"
    )
    st.latex(r"\frac{\partial\, y_{t+j}}{\partial\, w_t} = \phi^j")

    col_dm1, col_dm2, col_dm3 = st.columns(3)
    with col_dm1:
        st.info(r"$|\phi| < 1$: multiplier **decays** → shocks are transitory")
    with col_dm2:
        st.warning(r"$\phi = 1$: multiplier **persists** → permanent effect (unit root)")
    with col_dm3:
        st.error(r"$|\phi| > 1$: multiplier **explodes** → process is unstable")

    # ── Long-run multiplier ────────────────────────────────────────────────
    st.subheader("Long-Run Multiplier")
    st.markdown(
        "When $|\\phi| < 1$, the total long-run effect of a sustained unit change "
        "in $w_t$ is:"
    )
    st.latex(
        r"\sum_{j=0}^{\infty} \phi^j = \frac{1}{1-\phi}"
    )

    st.subheader("Stability Condition")
    st.success(
        "**Hamilton Proposition 1.1** — The sequence $\\{y_t\\}$ is bounded for "
        "every bounded $\\{w_t\\}$ if and only if $|\\phi| < 1$."
    )

    st.divider()

    # ── Interactive explorer ───────────────────────────────────────────────
    st.subheader("Interactive Explorer")

    c1, c2 = st.columns([1, 2], gap="large")

    with c1:
        phi_v = st.slider(
            "φ",
            min_value=-1.5,
            max_value=1.5,
            value=0.8,
            step=0.05,
            key="phi_11",
        )
        y0_v = st.number_input("Initial value y₀", value=1.0, step=0.5, key="y0_11")
        T_v = st.slider("Periods T", 10, 100, 50, key="T_11")
        shock_choice = st.radio(
            "Forcing term $w_t$",
            ["Single impulse at t = 1", "White noise N(0,1)", "Constant 1", "Zero"],
            key="shock_11",
        )

        if shock_choice == "Single impulse at t = 1":
            w_arr = np.zeros(T_v)
            w_arr[0] = 1.0
        elif shock_choice == "White noise N(0,1)":
            np.random.seed(7)
            w_arr = np.random.normal(0, 1, T_v)
        elif shock_choice == "Constant 1":
            w_arr = np.ones(T_v)
        else:
            w_arr = np.zeros(T_v)

        if abs(phi_v) < 1:
            st.success(f"**Stable** — |φ| = {abs(phi_v):.2f} < 1")
            st.latex(
                rf"\text{{Long-run multiplier}} = \frac{{1}}{{1-\phi}} = {1/(1-phi_v):.3f}"
            )
        elif abs(phi_v) == 1.0:
            st.warning("**Unit root** — φ = ±1")
        else:
            st.error(f"**Unstable** — |φ| = {abs(phi_v):.2f} > 1")

    with c2:
        y_arr = np.zeros(T_v)
        y_arr[0] = y0_v
        for t in range(1, T_v):
            y_arr[t] = phi_v * y_arr[t - 1] + w_arr[t]

        j_arr = np.arange(min(30, T_v))
        dm_arr = phi_v ** j_arr

        fig11 = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("Sequence  y_t", "Dynamic Multipliers  φ^j"),
        )
        fig11.add_trace(
            go.Scatter(
                x=list(range(T_v)),
                y=y_arr,
                mode="lines",
                line=dict(color="royalblue", width=2),
            ),
            row=1, col=1,
        )
        fig11.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)
        fig11.add_trace(
            go.Bar(x=list(j_arr), y=dm_arr, marker_color="coral"),
            row=1, col=2,
        )
        fig11.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=2)
        fig11.update_layout(
            height=400,
            showlegend=False,
            margin=dict(l=10, r=10, t=40, b=20),
        )
        fig11.update_xaxes(title_text="t", row=1, col=1)
        fig11.update_xaxes(title_text="j (periods ahead)", row=1, col=2)
        fig11.update_yaxes(title_text="y_t", row=1, col=1)
        fig11.update_yaxes(title_text="∂y_{t+j}/∂w_t", row=1, col=2)
        st.plotly_chart(fig11, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — §1.2 pTH-ORDER DIFFERENCE EQUATIONS
# ─────────────────────────────────────────────────────────────────────────────
with tabs[2]:
    st.header("§1.2  $p$th-Order Difference Equations")

    st.markdown(
        "Hamilton now generalises to $p$ lags.  The key insight is that any "
        "$p$th-order scalar equation can be rewritten as a *first-order vector* "
        "system via the **companion form**."
    )

    # ── General form ───────────────────────────────────────────────────────
    st.subheader("General Form")
    st.latex(
        r"y_t = \phi_1 y_{t-1} + \phi_2 y_{t-2} + \cdots + \phi_p y_{t-p} + w_t"
    )

    # ── Companion form ─────────────────────────────────────────────────────
    st.subheader("Companion (State-Space) Form")
    st.markdown("Define the $p \\times 1$ state vector:")
    st.latex(
        r"\boldsymbol{\xi}_t = \begin{pmatrix} y_t \\ y_{t-1} \\ \vdots \\ y_{t-p+1} \end{pmatrix}"
    )
    st.markdown("The $p$th-order equation becomes the first-order vector system:")
    st.latex(r"\boldsymbol{\xi}_t = \mathbf{F}\,\boldsymbol{\xi}_{t-1} + \mathbf{v}_t")
    st.markdown("where the **companion matrix** $\\mathbf{F}$ is:")
    st.latex(
        r"""
\mathbf{F} = \begin{pmatrix}
\phi_1  & \phi_2 & \phi_3 & \cdots & \phi_{p-1} & \phi_p \\
1       & 0      & 0      & \cdots & 0           & 0      \\
0       & 1      & 0      & \cdots & 0           & 0      \\
\vdots  &        & \ddots &        &             & \vdots \\
0       & 0      & 0      & \cdots & 1           & 0
\end{pmatrix}
"""
    )
    st.markdown(
        "and $\\mathbf{v}_t = (w_t, 0, \\ldots, 0)^\\top$. "
        "This reduces *all* stability questions to linear algebra on $\\mathbf{F}$."
    )

    # ── Characteristic roots ───────────────────────────────────────────────
    st.subheader("Characteristic Equation")
    st.markdown(
        "The **characteristic polynomial** of the difference equation is obtained "
        "by looking for solutions of the form $y_t = \\lambda^t$:"
    )
    st.latex(
        r"1 - \phi_1 z - \phi_2 z^2 - \cdots - \phi_p z^p = 0"
    )
    st.markdown(
        "The roots $z_1, \\ldots, z_p$ of this polynomial are the **reciprocals** "
        "of the eigenvalues of $\\mathbf{F}$."
    )

    # ── Stability theorem ──────────────────────────────────────────────────
    st.success(
        "**Hamilton Proposition 1.2 (Stability)** — The sequence is bounded for "
        "every bounded $\\{w_t\\}$ if and only if **all eigenvalues of "
        "$\\mathbf{F}$ lie strictly inside the unit circle**: $|\\lambda_i| < 1$ "
        "for $i = 1, \\ldots, p$."
    )
    st.markdown(
        "_Equivalently_: all roots $z_i$ of the characteristic polynomial satisfy "
        "$|z_i| > 1$ (roots outside the unit circle in the lag-operator convention "
        "used in Chapter 2)."
    )

    # ── Solution via eigendecomposition ───────────────────────────────────
    st.subheader("Solution via Eigendecomposition")
    st.markdown(
        "When $\\mathbf{F}$ has $p$ distinct eigenvalues $\\lambda_1, \\ldots, \\lambda_p$, "
        "the general solution is:"
    )
    st.latex(
        r"y_t = c_1 \lambda_1^t + c_2 \lambda_2^t + \cdots + c_p \lambda_p^t "
        r"+ \text{(particular solution)}"
    )
    st.markdown(
        "where the constants $c_i$ are determined by initial conditions. "
        "Complex conjugate eigenvalue pairs produce oscillatory (cyclical) behaviour."
    )

    st.divider()
    st.subheader("Interactive: $p$th-Order Explorer")

    c1, c2 = st.columns([1, 2], gap="large")

    with c1:
        p_ord = st.slider("Order p", 1, 4, 2, key="p_12")
        phi_c = []
        for i in range(p_ord):
            default = [0.5, 0.2, 0.1, 0.05][i]
            v = st.slider(
                f"φ_{i+1}",
                min_value=-2.0,
                max_value=2.0,
                value=default,
                step=0.05,
                key=f"phi12_{i}",
            )
            phi_c.append(v)

        T12 = st.slider("Periods T", 20, 150, 80, key="T_12")
        shock12 = st.radio(
            "$w_t$",
            ["Single impulse", "White noise", "Zero"],
            key="shock_12",
        )

        # Build companion matrix
        F_mat = np.zeros((p_ord, p_ord))
        F_mat[0, :] = phi_c
        for i in range(1, p_ord):
            F_mat[i, i - 1] = 1.0

        eigs = np.linalg.eigvals(F_mat)
        stable = all(abs(e) < 1 for e in eigs)

        st.markdown("**Eigenvalues of F:**")
        for i, e in enumerate(eigs):
            sign = "+" if e.imag >= 0 else "−"
            st.latex(
                rf"\lambda_{{{i+1}}} = {e.real:.3f} {sign} {abs(e.imag):.3f}i,"
                rf"\quad |\lambda_{{{i+1}}}| = {abs(e):.3f}"
            )

        if stable:
            st.success("**Stable** — all eigenvalues inside unit circle")
        else:
            st.error("**Unstable** — at least one eigenvalue outside unit circle")

    with c2:
        if shock12 == "Single impulse":
            w12 = np.zeros(T12)
            if p_ord < T12:
                w12[p_ord] = 1.0
        elif shock12 == "White noise":
            np.random.seed(42)
            w12 = np.random.normal(0, 1, T12)
        else:
            w12 = np.zeros(T12)

        y12 = np.zeros(T12)
        for t in range(p_ord, T12):
            y12[t] = sum(phi_c[j] * y12[t - j - 1] for j in range(p_ord)) + w12[t]

        theta = np.linspace(0, 2 * np.pi, 300)

        fig12 = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("Sequence  y_t", "Eigenvalues on Unit Circle"),
        )
        fig12.add_trace(
            go.Scatter(
                x=list(range(T12)),
                y=y12,
                mode="lines",
                line=dict(color="royalblue", width=2),
            ),
            row=1, col=1,
        )
        fig12.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)

        # Unit circle
        fig12.add_trace(
            go.Scatter(
                x=np.cos(theta).tolist(),
                y=np.sin(theta).tolist(),
                mode="lines",
                line=dict(color="gray", dash="dot", width=1),
                name="Unit circle",
            ),
            row=1, col=2,
        )
        fig12.add_trace(
            go.Scatter(
                x=[e.real for e in eigs],
                y=[e.imag for e in eigs],
                mode="markers",
                marker=dict(
                    color=["green" if abs(e) < 1 else "red" for e in eigs],
                    size=14,
                    symbol="x",
                    line=dict(width=2),
                ),
                name="Eigenvalues",
            ),
            row=1, col=2,
        )
        fig12.add_hline(y=0, line_dash="solid", line_color="lightgray", row=1, col=2)
        fig12.add_vline(x=0, line_dash="solid", line_color="lightgray", row=1, col=2)

        fig12.update_layout(
            height=420,
            showlegend=False,
            margin=dict(l=10, r=10, t=40, b=20),
        )
        fig12.update_xaxes(title_text="t", row=1, col=1)
        fig12.update_xaxes(title_text="Real part", range=[-2.2, 2.2], row=1, col=2)
        fig12.update_yaxes(title_text="y_t", row=1, col=1)
        fig12.update_yaxes(
            title_text="Imaginary part",
            range=[-2.2, 2.2],
            scaleanchor="x2",
            scaleratio=1,
            row=1,
            col=2,
        )
        st.plotly_chart(fig12, use_container_width=True)

        st.caption(
            "Green ✗ = eigenvalue inside unit circle (stable contribution). "
            "Red ✗ = outside (unstable contribution)."
        )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — §1.3 INITIAL CONDITIONS & STABILITY
# ─────────────────────────────────────────────────────────────────────────────
with tabs[3]:
    st.header("§1.3  Initial Conditions and Unbounded Sequences")

    st.markdown(
        "This section addresses a subtle but important point: every solution has "
        "two parts.  The **homogeneous** part encodes the effect of initial "
        "conditions; the **particular** part encodes the effect of the forcing "
        "sequence $\\{w_t\\}$."
    )

    # ── Decomposition ──────────────────────────────────────────────────────
    st.subheader("The Homogeneous + Particular Decomposition")
    st.latex(
        r"\underbrace{y_t}_{\text{general solution}}"
        r"= \underbrace{\phi^t c}_{\text{homogeneous (init. cond.)}}"
        r"+ \underbrace{\sum_{j=0}^{t-1}\phi^j w_{t-j}}_{\text{particular (forcing)}}"
    )
    st.markdown(
        "where $c = y_0$ for a first-order equation. "
        "For a $p$th-order system, the homogeneous solution is "
        "$\\sum_{i=1}^p c_i \\lambda_i^t$ where $\\lambda_i$ are eigenvalues."
    )

    # ── Three regimes ──────────────────────────────────────────────────────
    st.subheader("Three Regimes")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("#### $|\\phi| < 1$ — Stable")
        st.latex(r"\phi^t \xrightarrow{t\to\infty} 0")
        st.markdown(
            "Initial conditions **die out**. The long-run behaviour is "
            "governed entirely by $\\{w_t\\}$."
        )
        st.success("Stationary process")

    with col_b:
        st.markdown("#### $\\phi = 1$ — Unit Root")
        st.latex(r"y_t = y_0 + \sum_{j=1}^{t} w_j")
        st.markdown(
            "A random walk. Variance grows linearly: "
            "$\\operatorname{Var}(y_t) = t\\,\\sigma_w^2$. "
            "Initial conditions **never** vanish."
        )
        st.warning("Non-stationary (unit root)")

    with col_c:
        st.markdown("#### $|\\phi| > 1$ — Explosive")
        st.latex(r"\phi^t \xrightarrow{t\to\infty} \pm\infty")
        st.markdown(
            "Initial conditions are **amplified** each period. "
            "The sequence grows without bound."
        )
        st.error("Explosive / non-stationary")

    # ── Random walk detail ─────────────────────────────────────────────────
    st.subheader("The Random Walk: $\\phi = 1$")
    st.markdown(
        "Hamilton pays special attention to the unit-root case because it is "
        "empirically common in macroeconomics (GDP levels, prices, exchange rates)."
    )
    st.latex(
        r"\operatorname{Var}(y_t) = t \cdot \sigma_w^2 \quad \Rightarrow \quad"
        r"\text{variance } \to \infty"
    )
    st.markdown(
        "Unlike the stable case, no long-run mean exists: the process wanders "
        "without a fixed centre. This motivates the extensive treatment of "
        "**unit-root tests** in Chapters 15–18."
    )

    st.divider()
    st.subheader("Interactive: Effect of Initial Conditions")

    c1, c2 = st.columns([1, 2], gap="large")

    with c1:
        phi13 = st.slider(
            "φ", min_value=-1.5, max_value=1.5, value=0.9, step=0.05, key="phi_13"
        )
        y0_list = st.multiselect(
            "Initial conditions y₀ to compare",
            options=[-10, -5, -2, -1, 0, 1, 2, 5, 10],
            default=[-5, -1, 0, 2, 5],
        )
        T13 = st.slider("Periods T", 20, 120, 60, key="T_13")
        noise13 = st.checkbox("Add white noise ($\\sigma = 0.5$)", value=False)

    with c2:
        np.random.seed(99)
        w13 = np.random.normal(0, 0.5, T13) if noise13 else np.zeros(T13)
        colors = px.colors.qualitative.Safe

        fig13 = go.Figure()
        for idx, y0v in enumerate(y0_list if y0_list else [0]):
            yv = np.zeros(T13)
            yv[0] = y0v
            for t in range(1, T13):
                yv[t] = phi13 * yv[t - 1] + w13[t]
            fig13.add_trace(
                go.Scatter(
                    x=list(range(T13)),
                    y=yv,
                    mode="lines",
                    name=f"y₀ = {y0v}",
                    line=dict(color=colors[idx % len(colors)], width=2),
                )
            )

        fig13.add_hline(y=0, line_dash="dash", line_color="gray")
        fig13.update_layout(
            title=f"Trajectories for different initial conditions (φ = {phi13})",
            xaxis_title="t",
            yaxis_title="y_t",
            height=420,
            margin=dict(l=10, r=10, t=40, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig13, use_container_width=True)

        if abs(phi13) < 1:
            st.success(
                f"With |φ| = {abs(phi13):.2f} < 1 the trajectories converge "
                "regardless of starting point — initial conditions wash out."
            )
        elif abs(phi13) == 1.0:
            st.warning(
                "φ = ±1: unit root. Trajectories do not converge; they walk "
                "independently from their starting values."
            )
        else:
            st.error(
                f"|φ| = {abs(phi13):.2f} > 1: explosive regime. "
                "All trajectories diverge."
            )

    # ── Convergence speed ──────────────────────────────────────────────────
    st.divider()
    st.subheader("How Fast Do Initial Conditions Decay?")
    st.markdown(
        "After $T$ periods, the fraction of the initial condition that remains is $\\phi^T$:"
    )

    c1b, c2b = st.columns([1, 2], gap="large")
    with c1b:
        phi_hl = st.slider(
            "φ (stable only)", 0.01, 0.99, 0.85, step=0.01, key="phi_hl"
        )

    with c2b:
        hl = -np.log(2) / np.log(phi_hl)
        tt = np.arange(0, 60)
        decay = phi_hl ** tt

        fig_hl = go.Figure()
        fig_hl.add_trace(
            go.Scatter(
                x=tt.tolist(),
                y=decay.tolist(),
                mode="lines",
                line=dict(color="darkorange", width=2),
                name="φ^t",
            )
        )
        fig_hl.add_hline(y=0.5, line_dash="dash", line_color="blue",
                         annotation_text="50 % remaining", annotation_position="right")
        fig_hl.update_layout(
            title=f"Decay of initial condition: half-life ≈ {hl:.1f} periods",
            xaxis_title="t",
            yaxis_title="φ^t (fraction remaining)",
            height=300,
            margin=dict(l=10, r=10, t=40, b=20),
        )
        st.plotly_chart(fig_hl, use_container_width=True)
        st.markdown(
            rf"**Half-life:** $t_{{1/2}} = -\ln 2 / \ln \phi = {hl:.2f}$ periods. "
            "After this many periods, half the initial condition has dissipated."
        )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — QUIZ & PRACTICE
# ─────────────────────────────────────────────────────────────────────────────
with tabs[4]:
    st.header("Quiz & Practice")
    st.markdown("Test your understanding of Hamilton Chapter 1.")

    # ── Q1 ────────────────────────────────────────────────────────────────
    st.subheader("Question 1 — Dynamic Multiplier")
    st.markdown("Consider:")
    st.latex(r"y_t = 0.6\, y_{t-1} + w_t")
    q1 = st.radio(
        "What is the dynamic multiplier  ∂y_{t+3} / ∂w_t?",
        ["0.6", "1.8", "0.216", "3 × 0.6"],
        index=None,
        key="q1",
    )
    if q1 is not None:
        if q1 == "0.216":
            st.success(
                r"✅ Correct!  $\partial y_{t+3}/\partial w_t = \phi^3 = 0.6^3 = 0.216$"
            )
        else:
            st.error(
                r"❌ The dynamic multiplier is $\phi^j$, so for $j=3$: "
                r"$0.6^3 = 0.216$."
            )

    st.divider()

    # ── Q2 ────────────────────────────────────────────────────────────────
    st.subheader("Question 2 — Stability of a 2nd-Order System")
    st.markdown(
        "The companion matrix for $y_t = 1.2\\,y_{t-1} - 0.6\\,y_{t-2} + w_t$ is:"
    )
    st.latex(
        r"\mathbf{F} = \begin{pmatrix} 1.2 & -0.6 \\ 1 & 0 \end{pmatrix}"
    )
    q2 = st.radio(
        "Is this process stable?",
        [
            "Yes — all eigenvalues have modulus < 1",
            "No — at least one eigenvalue has modulus ≥ 1",
            "Cannot determine without knowing w_t",
        ],
        index=None,
        key="q2",
    )
    if q2 is not None:
        F_q2 = np.array([[1.2, -0.6], [1.0, 0.0]])
        eigs_q2 = np.linalg.eigvals(F_q2)
        stable_q2 = all(abs(e) < 1 for e in eigs_q2)
        correct_answer = (
            "Yes — all eigenvalues have modulus < 1"
            if stable_q2
            else "No — at least one eigenvalue has modulus ≥ 1"
        )
        if q2 == correct_answer:
            st.success(
                f"✅ Correct! Eigenvalues: "
                f"λ₁ ≈ {eigs_q2[0]:.3f}  (|λ₁| = {abs(eigs_q2[0]):.3f}), "
                f"λ₂ ≈ {eigs_q2[1]:.3f}  (|λ₂| = {abs(eigs_q2[1]):.3f}). "
                f"{'Both inside the unit circle → stable.' if stable_q2 else 'At least one outside → unstable.'}"
            )
        else:
            st.error(
                f"❌ Eigenvalues: {eigs_q2[0]:.3f}, {eigs_q2[1]:.3f}. "
                f"Moduli: {abs(eigs_q2[0]):.3f}, {abs(eigs_q2[1]):.3f}. "
                f"{'→ Stable.' if stable_q2 else '→ Unstable.'}"
            )

    st.divider()

    # ── Q3 ────────────────────────────────────────────────────────────────
    st.subheader("Question 3 — Role of Initial Conditions")
    q3 = st.radio(
        "When |φ| < 1, what happens to the effect of initial conditions as t → ∞?",
        [
            "It persists permanently",
            "It decays to zero exponentially",
            "It oscillates without damping",
            "It grows without bound",
        ],
        index=None,
        key="q3",
    )
    if q3 is not None:
        if q3 == "It decays to zero exponentially":
            st.success(
                r"✅ Correct! The homogeneous solution is $\phi^t c \to 0$ "
                r"when $|\phi| < 1$, so initial conditions vanish exponentially."
            )
        else:
            st.error(
                r"❌  When $|\phi| < 1$ we have $\phi^t \to 0$, so the "
                r"homogeneous part $\phi^t c$ decays to zero."
            )

    st.divider()

    # ── Q4 ────────────────────────────────────────────────────────────────
    st.subheader("Question 4 — Long-Run Multiplier")
    st.markdown("For $y_t = 0.75\\,y_{t-1} + w_t$ with $w_t = 1$ for all $t$, "
                "what is the long-run (steady-state) value of $y_t$?")
    q4 = st.radio(
        "Long-run value of y_t:",
        ["0.75", "1.0", "4.0", "∞"],
        index=None,
        key="q4",
    )
    if q4 is not None:
        if q4 == "4.0":
            st.success(
                r"✅ Correct! Long-run multiplier $= \frac{1}{1-\phi} = \frac{1}{1-0.75} = 4$."
            )
        else:
            st.error(
                r"❌ The steady state satisfies $y^* = \phi y^* + 1$, "
                r"giving $y^* = \frac{1}{1-\phi} = 4$."
            )

    st.divider()

    # ── Summary table ──────────────────────────────────────────────────────
    st.subheader("Chapter 1 — Key Results at a Glance")
    st.markdown(
        """
| Concept | Formula / Condition |
|---|---|
| First-order equation | $y_t = \\phi\\, y_{t-1} + w_t$ |
| Explicit solution | $y_t = \\phi^t y_0 + \\sum_{j=0}^{t-1} \\phi^j w_{t-j}$ |
| Dynamic multiplier | $\\partial y_{t+j}/\\partial w_t = \\phi^j$ |
| Long-run multiplier | $\\displaystyle\\sum_{j=0}^{\\infty}\\phi^j = \\frac{1}{1-\\phi}$ (requires $|\\phi|<1$) |
| Stability (1st order) | $|\\phi| < 1$ |
| Companion matrix | $p\\times p$ matrix $\\mathbf{F}$ with $\\phi_i$ in first row |
| Stability (pth order) | All eigenvalues of $\\mathbf{F}$ satisfy $|\\lambda_i| < 1$ |
| Unit root | $\\phi = 1$: variance grows as $t\\,\\sigma_w^2$ |
| Half-life of init. cond. | $t_{1/2} = -\\ln 2\\,/\\ln\\phi$ periods |
"""
    )

    st.markdown(
        "*Continue to Chapter 2 — Lag Operators — to see how the characteristic "
        "polynomial connects to the factorisation of lag-operator polynomials.*"
    )
