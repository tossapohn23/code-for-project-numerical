import tkinter as tk
import time
import os
from tkinter import messagebox, ttk

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
import sympy as sp

from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)


def setup_thai_matplotlib_font():
    preferred_font_files = [
        r"C:\Windows\Fonts\tahoma.ttf",
        r"C:\Windows\Fonts\tahomabd.ttf",
        r"C:\Windows\Fonts\LeelawUI.ttf",
        r"C:\Windows\Fonts\LeelUIsl.ttf",
        r"C:\Windows\Fonts\Nirmala.ttf",
    ]

    selected_font_name = None

    for font_path in preferred_font_files:
        if os.path.exists(font_path):
            try:
                font_manager.fontManager.addfont(font_path)

                font_prop = font_manager.FontProperties(
                    fname=font_path
                )

                selected_font_name = (
                    font_prop.get_name()
                )

                break

            except Exception:
                pass

    if selected_font_name is None:
        preferred_names = [
            "Tahoma",
            "Leelawadee UI",
            "Nirmala UI",
            "Arial",
        ]

        available_names = {
            item.name
            for item in font_manager.fontManager.ttflist
        }

        for name in preferred_names:
            if name in available_names:
                selected_font_name = name
                break

    # fallback สุดท้าย
    if selected_font_name is None:
        selected_font_name = "DejaVu Sans"

    plt.rcParams["font.family"] = selected_font_name
    plt.rcParams["font.sans-serif"] = [
        selected_font_name,
        "Tahoma",
        "Leelawadee UI",
        "Nirmala UI",
        "Arial",
        "DejaVu Sans",
    ]

    # ป้องกันเครื่องหมายลบถูกแทนด้วย glyph ที่บางฟอนต์ไม่มี
    plt.rcParams["axes.unicode_minus"] = False

    return selected_font_name


MATPLOTLIB_FONT_NAME = setup_thai_matplotlib_font()



# ============================================================
# 1. SMART PARSER
# ============================================================
def parse_flexible_expr(expr_str):
    """
    รองรับการพิมพ์ เช่น
        2x
        2*x
        x^2
        x**2
        e^x
        exp(x)
        sin(x)
        ln(x)
        sqrt(x)
    """
    expr_str = str(expr_str).strip()

    if not expr_str:
        return None

    transformations = standard_transformations + (
        convert_xor,
        implicit_multiplication_application,
    )

    local_dict = {
        "e": sp.E,
        "E": sp.E,
        "pi": sp.pi,
        "sin": sp.sin,
        "cos": sp.cos,
        "tan": sp.tan,
        "exp": sp.exp,
        "log": sp.log,
        "ln": sp.log,
        "sqrt": sp.sqrt,
    }

    return parse_expr(
        expr_str,
        local_dict=local_dict,
        transformations=transformations,
    )


# ============================================================
# 2. LINEAR / NONLINEAR CHECK
# ============================================================
def check_is_linear(f_str):
    """
    สำหรับ ODE:
        y' = f(x,y)

    ถ้า f เป็นเส้นตรงเมื่อเทียบกับ y เช่น
        a(x)y + b(x)
    จะถือเป็น Linear

    ถ้ามี y^2, sin(y), exp(y), y*y ฯลฯ
    จะถือเป็น Nonlinear
    """
    try:
        expr = parse_flexible_expr(f_str)

        if expr is None:
            return False

        y = sp.Symbol("y")

        second_derivative = sp.simplify(
            sp.diff(expr, y, 2)
        )

        return second_derivative == 0

    except Exception:
        return False


# ============================================================
# 3. EXACT SOLUTION
# ============================================================
def solve_exact_symbolic(f_str, x0, y0):
    """
    หา Exact Solution อัตโนมัติจาก
        dy/dx = f(x,y)
        y(x0) = y0

    ใช้ได้ทั้ง Linear และ Nonlinear เท่าที่ SymPy แก้ได้
    """
    try:
        x = sp.Symbol("x")
        y_func = sp.Function("y")(x)

        expr = parse_flexible_expr(f_str)

        if expr is None:
            return None

        expr_ode = expr.subs(
            sp.Symbol("y"),
            y_func,
        )

        ode = sp.Eq(
            y_func.diff(x),
            expr_ode,
        )

        ics = {
            y_func.subs(x, x0): y0
        }

        solution = sp.dsolve(
            ode,
            y_func,
            ics=ics,
        )

        return sp.simplify(
            solution.rhs
        )

    except Exception as e:
        print(f"SymPy Cannot solve exactly: {e}")
        return None


# ============================================================
# 4. CREATE NUMERICAL FUNCTION
# ============================================================
def create_evaluator(expr_str, is_f_xy=True):
    try:
        expr = parse_flexible_expr(expr_str)

        if expr is None:
            return None

        x = sp.Symbol("x")
        y = sp.Symbol("y")

        if is_f_xy:
            return sp.lambdify(
                (x, y),
                expr,
                "numpy",
            )

        return sp.lambdify(
            x,
            expr,
            "numpy",
        )

    except Exception as e:
        raise ValueError(
            f"เกิดข้อผิดพลาดในการอ่านสมการ: {e}"
        )


# ============================================================
# 5. DISPLAY EQUATION AS REAL MATH
# ============================================================
def equation_to_latex(expr):
    """
    แสดงผลสมการเป็นรูปคณิตศาสตร์ เช่น
        x**2     -> x^2
        exp(x)   -> e^x
        sqrt(x)  -> √x
        1/(x+1)  -> เศษส่วน
    """
    if expr is None:
        return r"\mathrm{Not\ Available}"

    try:
        if isinstance(expr, str):
            expr = parse_flexible_expr(expr)

        if expr is None:
            return r"\mathrm{Not\ Available}"

        expr = sp.simplify(expr)

        return sp.latex(
            expr,
            mul_symbol=" ",
        )

    except Exception:
        return r"\mathrm{Not\ Available}"


# ============================================================
# 6. NUMERICAL METHODS
# ============================================================
def run_calculations(
    f,
    exact_func,
    x0,
    y0,
    x_end,
    output_h,
    rkf_initial_h,
    tol,
):

    # --------------------------------------------------------
    # RK4
    # --------------------------------------------------------
    def rk4_step(x, y, h):
        k1 = f(x, y)

        k2 = f(
            x + 0.5 * h,
            y + 0.5 * h * k1,
        )

        k3 = f(
            x + 0.5 * h,
            y + 0.5 * h * k2,
        )

        k4 = f(
            x + h,
            y + h * k3,
        )

        y_next = y + (h / 6.0) * (
            k1
            + 2 * k2
            + 2 * k3
            + k4
        )

        return y_next, (k1, k2, k3, k4)

    # --------------------------------------------------------
    # CoM4
    # --------------------------------------------------------
    def com4_step(x, y, h):
        k1 = f(x, y)

        k2 = f(
            x + 0.5 * h,
            y + 0.5 * h * k1,
        )

        k3 = f(
            x + 0.5 * h,
            y
            + 0.125 * h * k1
            + 0.375 * h * k2,
        )

        k4 = f(
            x + h,
            y
            + 0.25 * h * k1
            - 0.75 * h * k2
            + 1.5 * h * k3,
        )

        denom1 = (
            k1 + k2
            if abs(k1 + k2) > 1e-15
            else 1e-15
        )

        denom2 = (
            k2 + k3
            if abs(k2 + k3) > 1e-15
            else 1e-15
        )

        denom3 = (
            k3 + k4
            if abs(k3 + k4) > 1e-15
            else 1e-15
        )

        y_next = y + (h / 3.0) * (
            ((k1**2 + k2**2) / denom1)
            + ((k2**2 + k3**2) / denom2)
            + ((k3**2 + k4**2) / denom3)
        )

        return y_next, (k1, k2, k3, k4)

    # --------------------------------------------------------
    # RKF45
    # --------------------------------------------------------
    def rkf45_pair(x, y, h):
        k1 = h * f(
            x,
            y,
        )

        k2 = h * f(
            x + (1 / 4) * h,
            y + (1 / 4) * k1,
        )

        k3 = h * f(
            x + (3 / 8) * h,
            y
            + (3 / 32) * k1
            + (9 / 32) * k2,
        )

        k4 = h * f(
            x + (12 / 13) * h,
            y
            + (1932 / 2197) * k1
            - (7200 / 2197) * k2
            + (7296 / 2197) * k3,
        )

        k5 = h * f(
            x + h,
            y
            + (439 / 216) * k1
            - 8 * k2
            + (3680 / 513) * k3
            - (845 / 4104) * k4,
        )

        k6 = h * f(
            x + (1 / 2) * h,
            y
            - (8 / 27) * k1
            + 2 * k2
            - (3544 / 2565) * k3
            + (1859 / 4104) * k4
            - (11 / 40) * k5,
        )

        # Order 4
        y4 = (
            y
            + (25 / 216) * k1
            + (1408 / 2565) * k3
            + (2197 / 4104) * k4
            - (1 / 5) * k5
        )

        # Order 5
        y5 = (
            y
            + (16 / 135) * k1
            + (6656 / 12825) * k3
            + (28561 / 56430) * k4
            - (9 / 50) * k5
            + (2 / 55) * k6
        )

        estimated_error = abs(
            y5 - y4
        )

        return (
            y4,
            y5,
            estimated_error,
            (k1, k2, k3, k4, k5, k6),
        )

    # --------------------------------------------------------
    # Adaptive RKF45
    #
    # ใช้ embedded error:
    #
    #       E = |y5 - y4|
    #
    # แล้วปรับ h หลังจาก "ทุก step" ด้วยสูตรที่กำหนด:
    #
    #       h_new = 0.84*h*((TOL*h)/E)^(1/4)
    # --------------------------------------------------------
    def rkf45_adaptive_full(
        x_start,
        y_start,
        x_target,
        h_initial,
        tolerance,
    ):
        if h_initial <= 0:
            raise ValueError(
                "Initial h ของ RKF4(5) ต้องมากกว่า 0"
            )

        if tolerance <= 0:
            raise ValueError(
                "Tolerance ของ RKF4(5) ต้องมากกว่า 0"
            )

        x = float(x_start)
        y = float(y_start)

        h_adaptive = min(
            abs(float(h_initial)),
            abs(float(x_target - x_start)),
        )

        history = [{
            "x": x,
            "y": y,
            "error_true": np.nan,
            "error_est": 0.0,
            "error_control": np.nan,
            "h_used": np.nan,
            "h_next": h_adaptive,
            "scale": np.nan,
            "k": (np.nan,) * 6,
        }]

        max_iterations = 100000
        iteration = 0

        while True:
            remaining = x_target - x

            # ใช้ machine precision เพื่อดูว่าถึงปลายช่วงแล้วหรือยัง
            x_eps = 8.0 * max(
                abs(np.spacing(float(x))),
                abs(np.spacing(float(x_target))),
                np.finfo(float).tiny,
            )

            if remaining <= x_eps:
                history[-1]["x"] = float(x_target)
                break

            iteration += 1

            if iteration > max_iterations:
                raise RuntimeError(
                    "RKF4(5) ใช้จำนวนรอบมากเกินไป"
                )

            # Step สุดท้ายต้องไม่เลย x_target
            h_used = min(
                h_adaptive,
                remaining,
            )

            if h_used <= 0.0 or x + h_used == x:
                raise RuntimeError(
                    "RKF4(5) step size เล็กเกินกว่าความละเอียดของ float64"
                )

            # ------------------------------------------------
            # คำนวณ k1-k6, Order 4, Order 5
            # ------------------------------------------------
            (
                y4,
                y5,
                estimated_error,
                k_values,
            ) = rkf45_pair(
                x,
                y,
                h_used,
            )

            values_to_check = [
                y4,
                y5,
                estimated_error,
                *k_values,
            ]

            if not all(
                np.isfinite(float(v))
                for v in values_to_check
            ):
                raise RuntimeError(
                    "RKF4(5) พบ NaN หรือ Infinity ระหว่างคำนวณ"
                )

            # ------------------------------------------------
            # E = |y5-y4|
            # ------------------------------------------------
            E = abs(
                float(estimated_error)
            )

            # ------------------------------------------------
            # ปรับ h ทุกครั้ง:
            #
            # h_new = 0.84*h*((TOL*h)/E)^(1/4)
            #
            # ------------------------------------------------
            E_safe = max(
                E,
                np.finfo(float).tiny,
            )

            # ------------------------------------------------
            # สูตรปรับ h ตามที่กำหนดโดยตรง:
            #
            # h_new = 0.84*h*((TOL*h)/E)^(1/4)
            # ------------------------------------------------
            h_new = (
                0.84
                * h_used
                * (
                    (
                        tolerance
                        * h_used
                    )
                    / E_safe
                ) ** (1.0 / 4.0)
            )

            if (
                not np.isfinite(h_new)
                or h_new <= 0.0
            ):
                raise RuntimeError(
                    "RKF4(5) ได้ค่า h_new ไม่ถูกต้อง"
                )

            # เก็บ scale ไว้สำหรับ history/debug เท่านั้น
            # โดย scale = h_new / h_used
            scale = (
                h_new / h_used
            )

            # ------------------------------------------------
            # ไม่มี Reject/Retry:
            # ทุก step ที่คำนวณแล้วจะเดินหน้า
            #
            # ใช้ Order 4 (y4) เป็น RKF4(5) approximation
            # ตามโครงสร้างเดิมของโปรแกรม
            # ------------------------------------------------
            x_new = x + h_used

            end_eps = 8.0 * max(
                abs(np.spacing(float(x_new))),
                abs(np.spacing(float(x_target))),
                np.finfo(float).tiny,
            )

            if abs(x_target - x_new) <= end_eps:
                x_new = float(x_target)

            x = float(x_new)
            y = float(y4)

            history.append({
                "x": x,
                "y": y,
                "error_true": np.nan,
                "error_est": E,
                "error_control": np.nan,
                "h_used": h_used,
                "h_next": h_new,
                "scale": scale,
                "k": k_values,
            })

            # ใช้ h_new ใน step ถัดไป
            h_adaptive = h_new

        return history

    # --------------------------------------------------------
    # Exact helper
    # --------------------------------------------------------
    def get_exact_value(x):
        if exact_func is None:
            return np.nan

        try:
            value = exact_func(x)
            value = np.real_if_close(value)
            return float(value)
        except Exception:
            return np.nan

    # --------------------------------------------------------
    # Output x-grid
    # --------------------------------------------------------
    x_values = [x0]
    current_x = x0

    while current_x < x_end - 1e-12:
        current_x = min(
            current_x + output_h,
            x_end,
        )

        x_values.append(current_x)

    # --------------------------------------------------------
    # RK4 / CoM4 ใช้ output grid คงที่
    # --------------------------------------------------------
    # วัดเฉพาะเวลาคำนวณของ numerical method
    # ไม่รวม Exact Solution, GUI และการวาดกราฟ
    rk4_runtime_ns = 0
    com4_runtime_ns = 0

    data = []
    curr_x = x0
    curr_y_rk4 = y0
    curr_y_com4 = y0
    exact_value = get_exact_value(curr_x)

    data.append({
        "x": curr_x,
        "Exact": exact_value,
        "RK4": curr_y_rk4,
        "RK4_Error": (abs(exact_value - curr_y_rk4) if pd.notna(exact_value) else np.nan),
        "RK4_k1": np.nan, "RK4_k2": np.nan, "RK4_k3": np.nan, "RK4_k4": np.nan,
        "CoM4": curr_y_com4,
        "CoM4_Error": (abs(exact_value - curr_y_com4) if pd.notna(exact_value) else np.nan),
        "CoM4_k1": np.nan, "CoM4_k2": np.nan, "CoM4_k3": np.nan, "CoM4_k4": np.nan,
    })

    for next_x in x_values[1:]:
        h_step = next_x - curr_x

        rk4_start_ns = time.perf_counter_ns()
        curr_y_rk4, rk4_k = rk4_step(
            curr_x,
            curr_y_rk4,
            h_step,
        )
        rk4_runtime_ns += time.perf_counter_ns() - rk4_start_ns

        com4_start_ns = time.perf_counter_ns()
        curr_y_com4, com4_k = com4_step(
            curr_x,
            curr_y_com4,
            h_step,
        )
        com4_runtime_ns += time.perf_counter_ns() - com4_start_ns

        exact_value = get_exact_value(next_x)

        data.append({
            "x": next_x,
            "Exact": exact_value,
            "RK4": curr_y_rk4,
            "RK4_Error": (abs(exact_value - curr_y_rk4) if pd.notna(exact_value) else np.nan),
            "RK4_k1": rk4_k[0], "RK4_k2": rk4_k[1], "RK4_k3": rk4_k[2], "RK4_k4": rk4_k[3],
            "CoM4": curr_y_com4,
            "CoM4_Error": (abs(exact_value - curr_y_com4) if pd.notna(exact_value) else np.nan),
            "CoM4_k1": com4_k[0], "CoM4_k2": com4_k[1], "CoM4_k3": com4_k[2], "CoM4_k4": com4_k[3],
        })
        curr_x = next_x

    df_fixed = pd.DataFrame(data)

    # RKF4(5) adaptive grid ของตัวเอง
    rkf_start_ns = time.perf_counter_ns()

    rkf_history = rkf45_adaptive_full(
        x0,
        y0,
        x_end,
        rkf_initial_h,
        tol,
    )

    rkf_runtime_ns = time.perf_counter_ns() - rkf_start_ns

    rkf_rows = []
    for item in rkf_history:
        exact_value = get_exact_value(item["x"])

        if pd.notna(exact_value):
            exact_for_table = float(f"{float(exact_value):.7f}")
            rkf_for_table = float(f"{float(item['y']):.7f}")
            rkf_true_error = float(
                f"{abs(exact_for_table - rkf_for_table):.7f}"
            )

            if abs(rkf_true_error) < 0.5e-7:
                rkf_true_error = 0.0
        else:
            rkf_true_error = np.nan

        rkf_rows.append({
            "x": item["x"],
            "Exact": exact_value,
            "RKF45": item["y"],
            "RKF45_Error": rkf_true_error,
            "RKF45_Estimated_Error": item["error_est"],
            "RKF45_Control_Error": item["error_control"],
            "RKF45_h": item["h_used"],
            "RKF45_h_next": item.get("h_next", np.nan),
            "RKF45_Scale": item.get("scale", np.nan),
            "RKF45_k1": item["k"][0],
            "RKF45_k2": item["k"][1],
            "RKF45_k3": item["k"][2],
            "RKF45_k4": item["k"][3],
            "RKF45_k5": item["k"][4],
            "RKF45_k6": item["k"][5],
        })

    df_rkf = pd.DataFrame(rkf_rows)

    df_fixed.attrs["RK4_Runtime_ms"] = rk4_runtime_ns / 1_000_000.0
    df_fixed.attrs["CoM4_Runtime_ms"] = com4_runtime_ns / 1_000_000.0
    df_rkf.attrs["RKF45_Runtime_ms"] = rkf_runtime_ns / 1_000_000.0

    return df_fixed, df_rkf


# ============================================================
# 7. NUMBER FORMAT
# ============================================================
def fmt7(value):
    if pd.isna(value):
        return "-"

    return f"{float(value):.7f}"


# ============================================================
# 8. PLOT RESULTS
# ============================================================
def plot_results(df_fixed, df_rkf, exact_expr, input_expr, is_linear):
    input_latex = equation_to_latex(input_expr)
    exact_latex = equation_to_latex(exact_expr)
    ode_type = "Linear ODE" if is_linear else "Nonlinear ODE"

    rk4_runtime_ms = df_fixed.attrs.get("RK4_Runtime_ms", np.nan)
    com4_runtime_ms = df_fixed.attrs.get("CoM4_Runtime_ms", np.nan)
    rkf_runtime_ms = df_rkf.attrs.get("RKF45_Runtime_ms", np.nan)

    def maximize_figure(fig):
        try:
            manager = fig.canvas.manager
            manager.window.state("zoomed")
        except Exception:
            try:
                manager.window.showMaximized()
            except Exception:
                pass

    def style_table(table, fontsize):
        table.auto_set_font_size(False)
        table.set_fontsize(fontsize)

        for (r, c), cell in table.get_celld().items():
            cell.set_edgecolor("#333333")
            cell.set_linewidth(0.55)

            if r == 0:
                cell.get_text().set_weight("bold")
                cell.set_facecolor("#F3F4F6")

    def add_summary_box(
        ax,
        final_value,
        iterations,
        runtime_ms,
    ):
        if pd.isna(final_value):
            final_text = "-"
        else:
            final_text = f"{float(final_value):.7f}"

        if pd.isna(runtime_ms):
            runtime_sec_text = "-"
        else:
            runtime_sec_text = f"{float(runtime_ms) / 1000.0:.6f}"

        summary_text = (
            f"คำตอบเชิงตัวเลข : {final_text}\n"
            f"จำนวนรอบ : {int(iterations)} รอบ\n"
            f"เวลา : {runtime_sec_text} วินาที"
        )

        ax.text(
            0.02,
            0.01,
            summary_text,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=9.0,
            fontweight="bold",
            linespacing=1.35,
            bbox=dict(
                boxstyle="round,pad=0.45",
                facecolor="#FFFFFF",
                edgecolor="#333333",
                linewidth=0.8,
            ),
        )

    # ========================================================
    # POPUP 1 : K VALUES
    # ========================================================
    fig_k = plt.figure(
        num="K Values - RK4, CoM4 and RKF4(5)",
        figsize=(20, 11),
    )

    grid_k = fig_k.add_gridspec(
        nrows=1,
        ncols=3,
        width_ratios=[1.0, 1.0, 1.45],
        wspace=0.08,
    )

    ax_rk4_k = fig_k.add_subplot(grid_k[0, 0])
    ax_com4_k = fig_k.add_subplot(grid_k[0, 1])
    ax_rkf_k = fig_k.add_subplot(grid_k[0, 2])

    fig_k.suptitle(
        "K Values for RK4, CoM4 and RKF4(5)",
        fontsize=15,
        fontweight="bold",
        y=0.975,
    )

    # RK4 K table
    ax_rk4_k.axis("off")
    ax_rk4_k.set_title(
        "RK4 Stage Values",
        fontsize=12,
        fontweight="bold",
        pad=8,
    )

    rk4_k_data = []
    for _, row in df_fixed.iterrows():
        rk4_k_data.append([
            f"{row['x']:.4f}",
            fmt7(row["RK4_k1"]),
            fmt7(row["RK4_k2"]),
            fmt7(row["RK4_k3"]),
            fmt7(row["RK4_k4"]),
        ])

    rk4_k_table = ax_rk4_k.table(
        cellText=rk4_k_data,
        colLabels=["x", "k1", "k2", "k3", "k4"],
        loc="center",
        cellLoc="center",
        bbox=[0.01, 0.02, 0.98, 0.92],
    )

    style_table(rk4_k_table, 7.0)

    # CoM4 K table
    ax_com4_k.axis("off")
    ax_com4_k.set_title(
        "CoM4 Stage Values",
        fontsize=12,
        fontweight="bold",
        pad=8,
    )

    com4_k_data = []
    for _, row in df_fixed.iterrows():
        com4_k_data.append([
            f"{row['x']:.4f}",
            fmt7(row["CoM4_k1"]),
            fmt7(row["CoM4_k2"]),
            fmt7(row["CoM4_k3"]),
            fmt7(row["CoM4_k4"]),
        ])

    com4_k_table = ax_com4_k.table(
        cellText=com4_k_data,
        colLabels=["x", "k1", "k2", "k3", "k4"],
        loc="center",
        cellLoc="center",
        bbox=[0.01, 0.02, 0.98, 0.92],
    )

    style_table(com4_k_table, 7.0)

    # RKF4(5) K table
    ax_rkf_k.axis("off")
    ax_rkf_k.set_title(
        "RKF4(5) Adaptive Step and Stage",
        fontsize=12,
        fontweight="bold",
        pad=8,
    )

    rkf_k_data = []
    for i, row in df_rkf.iterrows():
        rkf_k_data.append([
            str(i),
            f"{row['x']:.7f}",
            fmt7(row["RKF45_h"]),
            fmt7(row["RKF45_k1"]),
            fmt7(row["RKF45_k2"]),
            fmt7(row["RKF45_k3"]),
            fmt7(row["RKF45_k4"]),
            fmt7(row["RKF45_k5"]),
            fmt7(row["RKF45_k6"]),
        ])

    rkf_k_table = ax_rkf_k.table(
        cellText=rkf_k_data,
        colLabels=["k", "x", "h", "k1", "k2", "k3", "k4", "k5", "k6"],
        loc="center",
        cellLoc="center",
        bbox=[0.01, 0.02, 0.98, 0.92],
    )

    style_table(rkf_k_table, 6.2)

    fig_k.subplots_adjust(
        top=0.91,
        bottom=0.035,
        left=0.02,
        right=0.985,
    )

    maximize_figure(fig_k)

    # ========================================================
    # POPUP 2 : EXACT COMPARISON TABLES ONLY
    # ========================================================
    fig_result = plt.figure(
        num="Exact Comparison Tables - RK4, CoM4 and RKF4(5)",
        figsize=(22, 11),
    )

    grid_result = fig_result.add_gridspec(
        nrows=1,
        ncols=3,
        width_ratios=[1.0, 1.0, 1.22],
        wspace=0.07,
    )

    ax_rk4_result = fig_result.add_subplot(grid_result[0, 0])
    ax_com4_result = fig_result.add_subplot(grid_result[0, 1])
    ax_rkf_result = fig_result.add_subplot(grid_result[0, 2])

    title_text = (
        rf"$\frac{{dy}}{{dx}} = {input_latex}$"
        + "\n"
        + rf"$\mathrm{{{ode_type}}}$"
        + "    "
        + rf"$\mathrm{{Exact\ Solution:}}\quad y(x) = {exact_latex}$"
    )

    fig_result.suptitle(
        title_text,
        fontsize=14,
        fontweight="bold",
        y=0.965,
    )

    # RK4 vs Exact
    ax_rk4_result.axis("off")
    ax_rk4_result.set_title(
        "RK4 vs Exact",
        fontsize=13,
        fontweight="bold",
        pad=7,
    )

    rk4_result_data = []
    for _, row in df_fixed.iterrows():
        rk4_result_data.append([
            f"{row['x']:.4f}",
            fmt7(row["Exact"]),
            fmt7(row["RK4"]),
            fmt7(row["RK4_Error"]),
        ])

    rk4_result_table = ax_rk4_result.table(
        cellText=rk4_result_data,
        colLabels=["x", "Exact", "RK4", "Error"],
        loc="center",
        cellLoc="center",
        bbox=[0.0, 0.14, 1.0, 0.78],
    )

    style_table(rk4_result_table, 8.8)

    add_summary_box(
        ax_rk4_result,
        df_fixed.iloc[-1]["RK4"],
        max(len(df_fixed) - 1, 0),
        rk4_runtime_ms,
    )

    # CoM4 vs Exact
    ax_com4_result.axis("off")
    ax_com4_result.set_title(
        "CoM4 vs Exact",
        fontsize=13,
        fontweight="bold",
        pad=7,
    )

    com4_result_data = []
    for _, row in df_fixed.iterrows():
        com4_result_data.append([
            f"{row['x']:.4f}",
            fmt7(row["Exact"]),
            fmt7(row["CoM4"]),
            fmt7(row["CoM4_Error"]),
        ])

    com4_result_table = ax_com4_result.table(
        cellText=com4_result_data,
        colLabels=["x", "Exact", "CoM4", "Error"],
        loc="center",
        cellLoc="center",
        bbox=[0.0, 0.14, 1.0, 0.78],
    )

    style_table(com4_result_table, 8.8)

    add_summary_box(
        ax_com4_result,
        df_fixed.iloc[-1]["CoM4"],
        max(len(df_fixed) - 1, 0),
        com4_runtime_ms,
    )

    # RKF4(5) vs Exact
    ax_rkf_result.axis("off")
    ax_rkf_result.set_title(
        "RKF4(5) vs Exact",
        fontsize=13,
        fontweight="bold",
        pad=7,
    )

    rkf_result_data = []
    for i, row in df_rkf.iterrows():
        rkf_result_data.append([
            str(i),
            f"{row['x']:.7f}",
            fmt7(row["RKF45_h"]),
            fmt7(row["RKF45"]),
            fmt7(row["Exact"]),
            fmt7(row["RKF45_Error"]),
        ])

    rkf_result_table = ax_rkf_result.table(
        cellText=rkf_result_data,
        colLabels=["k", "x", "h", "RKF4(5)", "Exact", "Error"],
        loc="center",
        cellLoc="center",
        bbox=[0.0, 0.14, 1.0, 0.78],
    )

    style_table(rkf_result_table, 7.8)

    add_summary_box(
        ax_rkf_result,
        df_rkf.iloc[-1]["RKF45"],
        max(len(df_rkf) - 1, 0),
        rkf_runtime_ms,
    )

    fig_result.subplots_adjust(
        top=0.88,
        bottom=0.035,
        left=0.025,
        right=0.985,
    )

    maximize_figure(fig_result)

    # ========================================================
    # POPUP 3 : GRAPH ONLY
    # ========================================================
    fig_graph = plt.figure(
        num="Exact Solution Comparison Graph",
        figsize=(18, 8),
    )

    ax_graph = fig_graph.add_subplot(111)

    graph_title_text = (
        rf"$\frac{{dy}}{{dx}} = {input_latex}$"
        + "\n"
        + rf"$\mathrm{{{ode_type}}}$"
        + "    "
        + rf"$\mathrm{{Exact\ Solution:}}\quad y(x) = {exact_latex}$"
    )

    fig_graph.suptitle(
        graph_title_text,
        fontsize=14,
        fontweight="bold",
        y=0.965,
    )

    if df_fixed["Exact"].notna().any():
        ax_graph.plot(
            df_fixed["x"],
            df_fixed["Exact"],
            "r-d",
            label="Exact",
            linewidth=2.0,
            markersize=5,
            zorder=4,
        )

    ax_graph.plot(
        df_fixed["x"],
        df_fixed["RK4"],
        "b-*",
        label="RK-4",
        linewidth=1.2,
        markersize=5,
        zorder=3,
    )

    ax_graph.plot(
        df_fixed["x"],
        df_fixed["CoM4"],
        "g-s",
        label="CoM4",
        linewidth=1.2,
        markersize=4,
        zorder=2,
    )

    ax_graph.plot(
        df_rkf["x"],
        df_rkf["RKF45"],
        "m-o",
        label="RKF4(5)",
        linewidth=1.2,
        markersize=4,
        zorder=1,
    )

    ax_graph.set_title(
        "Exact Solution Comparison",
        fontsize=12,
        fontweight="bold",
        pad=7,
    )

    ax_graph.set_xlabel("x", fontsize=10)
    ax_graph.set_ylabel("y", fontsize=10)

    ax_graph.margins(
        x=0.04,
        y=0.08,
    )

    ax_graph.grid(False)

    ax_graph.legend(
        loc="upper left",
        frameon=True,
        edgecolor="#cccccc",
        fontsize=9,
    )

    for spine in ax_graph.spines.values():
        spine.set_color("#222222")
        spine.set_linewidth(0.8)

    fig_graph.subplots_adjust(
        top=0.82,
        bottom=0.11,
        left=0.07,
        right=0.98,
    )

    maximize_figure(fig_graph)

    plt.show(block=False)

    try:
        k_window = fig_k.canvas.manager.window

        try:
            k_window.lift()
            k_window.attributes("-topmost", True)
            k_window.after(
                200,
                lambda: k_window.attributes("-topmost", False),
            )
            k_window.focus_force()
        except Exception:
            pass

        try:
            k_window.raise_()
            k_window.activateWindow()
        except Exception:
            pass

    except Exception:
        pass

    plt.show(block=True)


# ============================================================
# 9. SUBMIT
# ============================================================
def on_submit():
    try:
        f_expr = (
            entry_f
            .get()
            .strip()
        )

        exact_input = (
            entry_exact
            .get()
            .strip()
        )

        x0 = float(
            entry_x0.get()
        )

        y0 = float(
            entry_y0.get()
        )

        x_end = float(
            entry_x_end.get()
        )

        h = float(
            entry_h.get()
        )

        rkf_h = float(
            entry_rkf_h.get()
        )

        tol = float(
            entry_tol.get()
        )

        if not f_expr:
            raise ValueError(
                "กรุณาระบุสมการ dy/dx = f(x,y)"
            )

        if h <= 0:
            raise ValueError(
                "ขนาดก้าว h ต้องมากกว่า 0"
            )

        if rkf_h <= 0:
            raise ValueError(
                "Initial h ของ RKF4(5) ต้องมากกว่า 0"
            )

        if tol <= 0:
            raise ValueError(
                "Tolerance ของ RKF4(5) ต้องมากกว่า 0"
            )

        if x_end <= x0:
            raise ValueError(
                "x_end ต้องมากกว่า x0"
            )

        # ----------------------------------------------------
        # ตรวจ Linear / Nonlinear
        # ----------------------------------------------------
        is_linear = check_is_linear(
            f_expr
        )

        if not is_linear:
            messagebox.showwarning(
                "Notification",
                "This is non-linear",
            )

        # ----------------------------------------------------
        # Exact Solution
        #
        # ถ้ากรอกเอง -> ใช้ค่าที่กรอก
        # ถ้าปล่อยว่าง -> ให้ SymPy หาอัตโนมัติ
        # ----------------------------------------------------
        exact_func = None

        if exact_input:
            exact_expr = parse_flexible_expr(
                exact_input
            )

            # Exact ที่ผู้ใช้กรอกเอง
            x_symbol = sp.Symbol("x")
            exact_func = sp.lambdify(
                x_symbol,
                exact_expr,
                "numpy",
            )

            # ทดสอบว่า Exact คำนวณได้
            exact_func(x0)

        else:
            exact_expr = solve_exact_symbolic(
                f_expr,
                x0,
                y0,
            )

            if exact_expr is not None:
                x_symbol = sp.Symbol("x")
                exact_func = sp.lambdify(
                    x_symbol,
                    exact_expr,
                    "numpy",
                )

        # ----------------------------------------------------
        # Numerical f(x,y)
        # ----------------------------------------------------
        f_func = create_evaluator(
            f_expr,
            is_f_xy=True,
        )

        if f_func is None:
            raise ValueError(
                "ไม่สามารถสร้างฟังก์ชัน f(x,y) ได้"
            )

        test_value = float(
            f_func(
                x0,
                y0,
            )
        )

        if not np.isfinite(
            test_value
        ):
            raise ValueError(
                "ค่า f(x0,y0) ไม่เป็นจำนวนจำกัด"
            )

        # ----------------------------------------------------
        # Calculate
        # ส่ง Exact เข้า run_calculations
        # เพื่อคำนวณ Error เทียบกับ Exact
        # ----------------------------------------------------
        df_fixed, df_rkf = run_calculations(
            f_func,
            exact_func,
            x0,
            y0,
            x_end,
            h,
            rkf_h,
            tol,
        )

        # ----------------------------------------------------
        # Show
        # ----------------------------------------------------
        root.withdraw()

        plot_results(
            df_fixed,
            df_rkf,
            exact_expr,
            f_expr,
            is_linear,
        )

        reset_form()

    except Exception as e:
        messagebox.showerror(
            "ข้อผิดพลาด",
            (
                "กรุณาตรวจสอบโครงสร้างสมการ"
                "หรือค่าพารามิเตอร์อีกครั้ง:\n\n"
                f"{e}"
            ),
        )


# ============================================================
# 10. RESET GUI
# ============================================================
def reset_form():
    root.deiconify()
    root.lift()
    root.focus_force()

    entry_f.focus_set()
    entry_f.selection_range(
        0,
        tk.END,
    )


# ============================================================
# 11. INPUT GUI
# ============================================================
root = tk.Tk()

root.title("Numerical ODE ")
root.geometry("680x600")
root.minsize(680, 600)
root.resizable(False, False)

root.update_idletasks()

window_width = 680
window_height = 600
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
position_x = int((screen_width - window_width) / 2)
position_y = int((screen_height - window_height) / 2)

root.geometry(
    f"{window_width}x{window_height}+{position_x}+{position_y}"
)

BG_COLOR = "#F4F7FB"
CARD_COLOR = "#FFFFFF"
PRIMARY_COLOR = "#2563EB"
PRIMARY_HOVER = "#1D4ED8"
TEXT_COLOR = "#1F2937"
SUBTEXT_COLOR = "#64748B"
BORDER_COLOR = "#D7E0EA"
ENTRY_BG = "#F8FAFC"
EXAMPLE_BG = "#EFF6FF"

root.configure(bg=BG_COLOR)

style = ttk.Style()
style.theme_use("clam")

style.configure("Page.TFrame", background=BG_COLOR)
style.configure("Card.TFrame", background=CARD_COLOR)

style.configure(
    "Title.TLabel",
    background=BG_COLOR,
    foreground=TEXT_COLOR,
    font=("Tahoma", 19, "bold"),
)

style.configure(
    "Subtitle.TLabel",
    background=BG_COLOR,
    foreground=SUBTEXT_COLOR,
    font=("Tahoma", 10),
)

style.configure(
    "Section.TLabel",
    background=CARD_COLOR,
    foreground=PRIMARY_COLOR,
    font=("Tahoma", 11, "bold"),
)

style.configure(
    "Field.TLabel",
    background=CARD_COLOR,
    foreground=TEXT_COLOR,
    font=("Tahoma", 10, "bold"),
)

style.configure(
    "Hint.TLabel",
    background=CARD_COLOR,
    foreground=SUBTEXT_COLOR,
    font=("Tahoma", 8),
)

style.configure(
    "Example.TLabel",
    background=EXAMPLE_BG,
    foreground="#1E40AF",
    font=("Tahoma", 9),
)

style.configure(
    "Input.TEntry",
    fieldbackground=ENTRY_BG,
    foreground=TEXT_COLOR,
    bordercolor=BORDER_COLOR,
    lightcolor=BORDER_COLOR,
    darkcolor=BORDER_COLOR,
    padding=(9, 6),
)

style.map(
    "Input.TEntry",
    bordercolor=[
        ("focus", PRIMARY_COLOR),
        ("!focus", BORDER_COLOR),
    ],
    lightcolor=[
        ("focus", PRIMARY_COLOR),
        ("!focus", BORDER_COLOR),
    ],
    darkcolor=[
        ("focus", PRIMARY_COLOR),
        ("!focus", BORDER_COLOR),
    ],
)

style.configure(
    "Primary.TButton",
    background=PRIMARY_COLOR,
    foreground="#FFFFFF",
    borderwidth=0,
    focusthickness=0,
    font=("Tahoma", 11, "bold"),
    padding=(18, 10),
)

style.map(
    "Primary.TButton",
    background=[
        ("active", PRIMARY_HOVER),
        ("pressed", "#1E3A8A"),
    ],
    foreground=[
        ("active", "#FFFFFF"),
        ("pressed", "#FFFFFF"),
    ],
)

style.configure(
    "Secondary.TButton",
    background="#E2E8F0",
    foreground=TEXT_COLOR,
    borderwidth=0,
    focusthickness=0,
    font=("Tahoma", 10),
    padding=(16, 9),
)

style.map(
    "Secondary.TButton",
    background=[
        ("active", "#CBD5E1"),
        ("pressed", "#B8C5D4"),
    ],
)


def clear_form():
    for entry in (
        entry_f,
        entry_exact,
        entry_x0,
        entry_y0,
        entry_x_end,
        entry_h,
        entry_rkf_h,
        entry_tol,
    ):
        entry.delete(0, tk.END)

    entry_f.focus_set()


EXAMPLE_PROBLEMS = {
    "Nonlinear 1: y' = y^2": {
        "equation": "y^2",
        "exact": "1/(1-x)",
        "x0": "0",
        "y0": "1",
        "x_end": "0.5",
        "h": "0.1",
        "tol": "0.000001",
        "description": "สมการ Nonlinear แบบแยกตัวแปรได้",
    },
    "Nonlinear 2: y' = x*y^2": {
        "equation": "x*y^2",
        "exact": "1/(1-x^2/2)",
        "x0": "0",
        "y0": "1",
        "x_end": "1",
        "h": "0.1",
        "tol": "0.000001",
        "description": "สมการ Nonlinear ที่มีทั้ง x และ y^2",
    },
    "Nonlinear 3: y' = y*(1-y)": {
        "equation": "y*(1-y)",
        "exact": "1/(1+9*e^(-x))",
        "x0": "0",
        "y0": "0.1",
        "x_end": "4",
        "h": "0.2",
        "tol": "0.000001",
        "description": "สมการ Logistic แบบ Nonlinear",
    },
    "Linear 1: y' = x + y": {
        "equation": "x + y",
        "exact": "2*e^x-x-1",
        "x0": "0",
        "y0": "1",
        "x_end": "1",
        "h": "0.1",
        "tol": "0.000001",
        "description": "สมการเชิงเส้นอันดับหนึ่ง",
    },
    "Linear 2: y' = e^x - y": {
        "equation": "e^x-y",
        "exact": "0.5*(e^x+e^(-x))",
        "x0": "0",
        "y0": "1",
        "x_end": "2",
        "h": "0.2",
        "tol": "0.000001",
        "description": "สมการเชิงเส้นที่มีฟังก์ชันเอกซ์โพเนนเชียล",
    },
    "Linear 3: y' = -2y": {
        "equation": "-2*y",
        "exact": "3*e^(-2*x)",
        "x0": "0",
        "y0": "3",
        "x_end": "2",
        "h": "0.2",
        "tol": "0.000001",
        "description": "สมการการลดแบบ Exponential decay",
    },
}


def fill_example(example_name):
    example = EXAMPLE_PROBLEMS[example_name]

    values = {
        entry_f: example["equation"],
        entry_exact: example["exact"],
        entry_x0: example["x0"],
        entry_y0: example["y0"],
        entry_x_end: example["x_end"],
        entry_h: example["h"],
        entry_rkf_h: example.get("rkf_h", example["h"]),
        entry_tol: example.get("tol", "0.000001"),
    }

    for entry, value in values.items():
        entry.delete(0, tk.END)
        entry.insert(0, value)

    entry_f.focus_set()


def load_example():
    example_window = tk.Toplevel(root)
    example_window.title("เลือกโจทย์ตัวอย่าง")
    example_window.configure(bg=BG_COLOR)
    example_window.resizable(False, False)
    example_window.transient(root)
    example_window.grab_set()

    width = 590
    height = 470

    example_window.update_idletasks()

    x_pos = root.winfo_x() + int((root.winfo_width() - width) / 2)
    y_pos = root.winfo_y() + int((root.winfo_height() - height) / 2)

    example_window.geometry(
        f"{width}x{height}+{x_pos}+{y_pos}"
    )

    # --------------------------------------------------------
    # Style สำหรับหน้าต่างเลือกตัวอย่าง
    # --------------------------------------------------------
    style.configure(
        "DialogTitle.TLabel",
        background=BG_COLOR,
        foreground=TEXT_COLOR,
        font=("Tahoma", 18, "bold"),
    )

    style.configure(
        "DialogSubtitle.TLabel",
        background=BG_COLOR,
        foreground=SUBTEXT_COLOR,
        font=("Tahoma", 9),
    )

    style.configure(
        "DialogCard.TFrame",
        background=CARD_COLOR,
    )

    style.configure(
        "DialogField.TLabel",
        background=CARD_COLOR,
        foreground=TEXT_COLOR,
        font=("Tahoma", 10, "bold"),
    )

    style.configure(
        "DialogHint.TLabel",
        background=CARD_COLOR,
        foreground=SUBTEXT_COLOR,
        font=("Tahoma", 9),
    )

    style.configure(
        "Example.TCombobox",
        fieldbackground=ENTRY_BG,
        background=ENTRY_BG,
        foreground=TEXT_COLOR,
        bordercolor=BORDER_COLOR,
        lightcolor=BORDER_COLOR,
        darkcolor=BORDER_COLOR,
        padding=(8, 7),
        arrowsize=16,
    )

    style.map(
        "Example.TCombobox",
        fieldbackground=[
            ("readonly", ENTRY_BG),
        ],
        selectbackground=[
            ("readonly", PRIMARY_COLOR),
        ],
        selectforeground=[
            ("readonly", "#FFFFFF"),
        ],
        bordercolor=[
            ("focus", PRIMARY_COLOR),
            ("!focus", BORDER_COLOR),
        ],
    )

    # --------------------------------------------------------
    # Main dialog page
    # --------------------------------------------------------
    dialog_page = ttk.Frame(
        example_window,
        style="Page.TFrame",
        padding=(30, 24, 30, 28),
    )

    dialog_page.pack(
        fill=tk.BOTH,
        expand=True,
    )

    ttk.Label(
        dialog_page,
        text="เลือกโจทย์ตัวอย่าง",
        style="DialogTitle.TLabel",
    ).pack(
        anchor="w",
    )

    ttk.Label(
        dialog_page,
        text="เลือกได้ทั้งสมการ Linear และ Nonlinear",
        style="DialogSubtitle.TLabel",
    ).pack(
        anchor="w",
        pady=(4, 16),
    )

    # --------------------------------------------------------
    # Card
    # --------------------------------------------------------
    dialog_border = tk.Frame(
        dialog_page,
        bg=BORDER_COLOR,
        highlightthickness=0,
    )

    dialog_border.pack(
        fill=tk.BOTH,
        expand=True,
    )

    dialog_card = ttk.Frame(
        dialog_border,
        style="DialogCard.TFrame",
        padding=(22, 20, 22, 20),
    )

    dialog_card.pack(
        fill=tk.BOTH,
        expand=True,
        padx=1,
        pady=1,
    )

    ttk.Label(
        dialog_card,
        text="โจทย์ที่ต้องการ",
        style="DialogField.TLabel",
    ).pack(
        anchor="w",
    )

    selected_example = tk.StringVar(
        value=list(EXAMPLE_PROBLEMS.keys())[0]
    )

    example_combo = ttk.Combobox(
        dialog_card,
        textvariable=selected_example,
        values=list(EXAMPLE_PROBLEMS.keys()),
        state="readonly",
        font=("Tahoma", 10),
        style="Example.TCombobox",
    )

    example_combo.pack(
        fill=tk.X,
        pady=(7, 15),
    )

    ttk.Label(
        dialog_card,
        text="รายละเอียดโจทย์",
        style="DialogField.TLabel",
    ).pack(
        anchor="w",
        pady=(0, 7),
    )

    # กล่องรายละเอียดสีฟ้าอ่อน
    detail_box = tk.Frame(
        dialog_card,
        bg=EXAMPLE_BG,
        highlightbackground="#BFDBFE",
        highlightthickness=1,
    )

    detail_box.pack(
        fill=tk.BOTH,
        expand=True,
        pady=(0, 5),
    )

    detail_text = tk.StringVar()

    detail_label = tk.Label(
        detail_box,
        textvariable=detail_text,
        justify="left",
        anchor="nw",
        bg=EXAMPLE_BG,
        fg=TEXT_COLOR,
        font=("Consolas", 10),
        padx=16,
        pady=14,
    )

    detail_label.pack(
        fill=tk.BOTH,
        expand=True,
    )

    def update_detail(event=None):
        item = EXAMPLE_PROBLEMS[
            selected_example.get()
        ]

        detail_text.set(
            f"{item['description']}\n\n"
            f"dy/dx   = {item['equation']}\n"
            f"Exact   = {item['exact']}\n"
            f"x0      = {item['x0']}\n"
            f"y(x0)   = {item['y0']}\n"
            f"x_end   = {item['x_end']}\n"
            f"h       = {item['h']}"
        )

    def confirm_example():
        fill_example(
            selected_example.get()
        )
        example_window.destroy()

    example_combo.bind(
        "<<ComboboxSelected>>",
        update_detail,
    )

    # --------------------------------------------------------
    # Buttons
    # --------------------------------------------------------
    button_area = ttk.Frame(
        dialog_card,
        style="DialogCard.TFrame",
    )

    button_area.pack(
        fill=tk.X,
    )

    button_area.columnconfigure(0, weight=1)
    button_area.columnconfigure(1, weight=1)

    cancel_button = ttk.Button(
        button_area,
        text="ยกเลิก",
        command=example_window.destroy,
        style="Secondary.TButton",
    )

    cancel_button.grid(
        row=0,
        column=0,
        sticky="ew",
        padx=(0, 7),
    )

    use_button = ttk.Button(
        button_area,
        text="ใช้ตัวอย่างนี้",
        command=confirm_example,
        style="Primary.TButton",
    )

    use_button.grid(
        row=0,
        column=1,
        sticky="ew",
        padx=(7, 0),
    )

    example_window.bind(
        "<Return>",
        lambda event: confirm_example(),
    )

    example_window.bind(
        "<Escape>",
        lambda event: example_window.destroy(),
    )

    example_window.protocol(
        "WM_DELETE_WINDOW",
        example_window.destroy,
    )

    update_detail()
    example_combo.focus_set()


def submit_with_enter(event=None):
    """
    กด Enter จากช่องกรอกใดก็ได้ เพื่อประมวลผล
    """
    on_submit()
    return "break"


page = ttk.Frame(
    root,
    style="Page.TFrame",
    padding=(28, 14, 28, 16),
)

page.pack(
    fill=tk.BOTH,
    expand=True,
)

header_frame = ttk.Frame(
    page,
    style="Page.TFrame",
)

header_frame.pack(
    fill=tk.X,
    pady=(0, 18),
)

ttk.Label(
    header_frame,
    text="Insert the equation",
    style="Title.TLabel",
).pack(
    anchor="center",
)

ttk.Label(
    header_frame,
    text="เปรียบเทียบ Exact Solution, RK-4, CoM4 และ RKF4(5)",
    style="Subtitle.TLabel",
).pack(
    anchor="center",
    pady=(5, 0),
)

card_border = tk.Frame(
    page,
    bg=BORDER_COLOR,
    highlightthickness=0,
)

card_border.pack(
    fill=tk.BOTH,
    expand=True,
)

card = ttk.Frame(
    card_border,
    style="Card.TFrame",
    padding=(20, 10, 20, 10),
)

card.pack(
    fill=tk.BOTH,
    expand=True,
    padx=1,
    pady=1,
)

ttk.Label(
    card,
    text="ข้อมูลสมการ",
    style="Section.TLabel",
).grid(
    row=0,
    column=0,
    columnspan=4,
    sticky="w",
    pady=(0, 9),
)

for column in range(4):
    card.columnconfigure(column, weight=1)

ttk.Label(
    card,
    text="สมการ dy/dx = f(x, y)",
    style="Field.TLabel",
).grid(
    row=1,
    column=0,
    columnspan=4,
    sticky="w",
)

entry_f = ttk.Entry(
    card,
    style="Input.TEntry",
    font=("Consolas", 11),
)

entry_f.grid(
    row=2,
    column=0,
    columnspan=4,
    sticky="ew",
    pady=(3, 2),
)

ttk.Label(
    card,
    text="ตัวอย่าง:  x + y^2,  x*y,  e^x - y,  sin(x) + y",
    style="Hint.TLabel",
).grid(
    row=3,
    column=0,
    columnspan=4,
    sticky="w",
    pady=(0, 7),
)

ttk.Label(
    card,
    text="Exact Solution",
    style="Field.TLabel",
).grid(
    row=4,
    column=0,
    columnspan=4,
    sticky="w",
)

entry_exact = ttk.Entry(
    card,
    style="Input.TEntry",
    font=("Consolas", 11),
)

entry_exact.grid(
    row=5,
    column=0,
    columnspan=4,
    sticky="ew",
    pady=(3, 2),
)

ttk.Label(
    card,
    text="กรอกเองหรือปล่อยว่างเพื่อให้โปรแกรมหาให้อัตโนมัติ",
    style="Hint.TLabel",
).grid(
    row=6,
    column=0,
    columnspan=4,
    sticky="w",
    pady=(0, 9),
)

ttk.Label(
    card,
    text="เงื่อนไขเริ่มต้นและช่วงคำนวณ",
    style="Section.TLabel",
).grid(
    row=7,
    column=0,
    columnspan=4,
    sticky="w",
    pady=(0, 5),
)

ttk.Label(
    card,
    text="ค่าเริ่มต้น x₀",
    style="Field.TLabel",
).grid(
    row=8,
    column=0,
    sticky="w",
)

entry_x0 = ttk.Entry(
    card,
    style="Input.TEntry",
    font=("Consolas", 11),
    width=14,
)

entry_x0.grid(
    row=9,
    column=0,
    columnspan=2,
    sticky="ew",
    padx=(0, 9),
    pady=(2, 3),
)

ttk.Label(
    card,
    text="ค่าเริ่มต้น y(x₀)",
    style="Field.TLabel",
).grid(
    row=8,
    column=2,
    sticky="w",
    padx=(9, 0),
)

entry_y0 = ttk.Entry(
    card,
    style="Input.TEntry",
    font=("Consolas", 11),
    width=14,
)

entry_y0.grid(
    row=9,
    column=2,
    columnspan=2,
    sticky="ew",
    padx=(9, 0),
    pady=(2, 3),
)

ttk.Label(
    card,
    text="ค่าปลายช่วง x",
    style="Field.TLabel",
).grid(
    row=10,
    column=0,
    sticky="w",
)

entry_x_end = ttk.Entry(
    card,
    style="Input.TEntry",
    font=("Consolas", 11),
    width=14,
)

entry_x_end.grid(
    row=11,
    column=0,
    columnspan=2,
    sticky="ew",
    padx=(0, 9),
    pady=(2, 3),
)

ttk.Label(
    card,
    text="h สำหรับ RK4 / CoM4",
    style="Field.TLabel",
).grid(
    row=10,
    column=2,
    sticky="w",
    padx=(9, 0),
)

entry_h = ttk.Entry(
    card,
    style="Input.TEntry",
    font=("Consolas", 11),
    width=14,
)

entry_h.grid(
    row=11,
    column=2,
    columnspan=2,
    sticky="ew",
    padx=(9, 0),
    pady=(2, 3),
)

# ------------------------------------------------------------
# RKF4(5): Initial h + Tolerance
# ------------------------------------------------------------
ttk.Label(
    card,
    text="h สำหรับ RKF4(5)",
    style="Field.TLabel",
).grid(
    row=12,
    column=0,
    columnspan=2,
    sticky="w",
    pady=(5, 0),
)

ttk.Label(
    card,
    text="Tolerance สำหรับ RKF4(5)",
    style="Field.TLabel",
).grid(
    row=12,
    column=2,
    columnspan=2,
    sticky="w",
    padx=(9, 0),
    pady=(5, 0),
)

entry_rkf_h = ttk.Entry(
    card,
    style="Input.TEntry",
    font=("Consolas", 11),
)

entry_rkf_h.grid(
    row=13,
    column=0,
    columnspan=2,
    sticky="ew",
    padx=(0, 9),
    pady=(3, 2),
)

entry_tol = ttk.Entry(
    card,
    style="Input.TEntry",
    font=("Consolas", 11),
)

entry_tol.grid(
    row=13,
    column=2,
    columnspan=2,
    sticky="ew",
    padx=(9, 0),
    pady=(3, 2),
)

ttk.Label(
    card,
    text="RKF4(5): h_new = 0.84·h·((TOL·h)/E)^(1/4),  E = |y5-y4|,  ปรับ h ทุก step",
    style="Hint.TLabel",
).grid(
    row=14,
    column=0,
    columnspan=4,
    sticky="w",
    pady=(0, 5),
)


button_frame = ttk.Frame(
    card,
    style="Card.TFrame",
)

button_frame.grid(
    row=15,
    column=0,
    columnspan=4,
    sticky="ew",
    pady=(7, 0),
)

button_frame.columnconfigure(0, weight=1)
button_frame.columnconfigure(1, weight=1)

button_clear = ttk.Button(
    button_frame,
    text="ล้างข้อมูล",
    command=clear_form,
    style="Secondary.TButton",
)

button_clear.grid(
    row=0,
    column=0,
    sticky="ew",
    padx=(0, 6),
)

button_submit = ttk.Button(
    button_frame,
    text="ประมวลผล",
    command=on_submit,
    style="Primary.TButton",
)

button_submit.grid(
    row=0,
    column=1,
    sticky="ew",
    padx=(6, 0),
)

# กด Enter จากช่องใดก็ได้ = ประมวลผล
for entry_widget in (
    entry_f,
    entry_exact,
    entry_x0,
    entry_y0,
    entry_x_end,
    entry_h,
    entry_rkf_h,
    entry_tol,
):
    entry_widget.bind("<Return>", submit_with_enter)

root.bind("<Return>", submit_with_enter)

# Esc = ล้างข้อมูล
root.bind("<Escape>", lambda event: clear_form())

entry_f.focus_set()


# ============================================================
# START
# ============================================================
root.mainloop()
