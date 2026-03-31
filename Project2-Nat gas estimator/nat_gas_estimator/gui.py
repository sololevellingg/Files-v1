# =============================================================================
# gui.py  —  Natural Gas Price Estimator  |  Interactive Desktop App
# Features: OLS model, confidence intervals, forecast, CSV export
# =============================================================================

import os, sys, csv
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from datetime import datetime, date
from model import (RAW_DATA, T0, DATA_END, FORECAST_END,
                   BETA, R2, MAE, RMSE, CI_95, AMPLITUDE, TREND_YR,
                   _ts, _prices, to_years, build_row, predict)

# ── Palette ───────────────────────────────────────────────────────────────────
BG       = "#0D1B2A"
PANEL    = "#1B2838"
BORDER   = "#1E3A5F"
ACCENT   = "#00B4D8"
ACCENT2  = "#FF6B6B"
SUCCESS  = "#06D6A0"
WARNING  = "#FFD166"
TEXT     = "#E0F2FE"
MUTED    = "#4A7C9E"
HIST_CLR = "#00B4D8"
FORE_CLR = "#FF6B6B"
CI_CLR   = "#FF6B6B"

# ── Spark chart ───────────────────────────────────────────────────────────────
class SparkChart(tk.Canvas):
    def __init__(self, parent, **kw):
        super().__init__(parent, bg=BG, highlightthickness=0, **kw)
        self._marker = None
        self.bind("<Configure>", lambda e: self._draw_base())

    def _draw_base(self):
        self.delete("all")
        w = self.winfo_width()  or 600
        h = self.winfo_height() or 180
        pad_l, pad_r, pad_t, pad_b = 50, 20, 16, 28

        min_x = min(_ts)
        max_x = to_years(FORECAST_END)
        min_y, max_y = 9.0, 14.5

        def sx(x): return pad_l + (x - min_x) / (max_x - min_x) * (w - pad_l - pad_r)
        def sy(y): return h - pad_b - (y - min_y) / (max_y - min_y) * (h - pad_t - pad_b)

        self._sx, self._sy = sx, sy
        self._min_x, self._max_x = min_x, max_x

        # Grid lines
        for yv in [10.0, 11.0, 12.0, 13.0, 14.0]:
            yc = sy(yv)
            self.create_line(pad_l, yc, w - pad_r, yc, fill=BORDER, dash=(4, 4))
            self.create_text(pad_l - 6, yc, text=f"${yv:.0f}",
                             anchor="e", fill=MUTED, font=("Courier", 8))

        # CI band (fitted)
        import math
        steps = 200
        band_pts = []
        for direction in [1, -1]:
            pts = []
            r = range(steps + 1) if direction == 1 else range(steps, -1, -1)
            for i in r:
                t  = min_x + (max_x - min_x) * i / steps
                dt = datetime(2020, 10, 31) + __import__('datetime').timedelta(days=t * 365.25)
                if dt <= DATA_END:
                    row   = build_row(t)
                    price = float(np.array(row) @ BETA)
                    pts.extend([sx(t), sy(price + direction * CI_95)])
            band_pts.extend(pts)
        if len(band_pts) >= 4:
            self.create_polygon(*band_pts, fill="#FF6B6B22", outline="", smooth=False)

        # Forecast CI band
        fc_upper, fc_lower = [], []
        for i in range(steps + 1):
            t  = to_years(DATA_END) + (to_years(FORECAST_END) - to_years(DATA_END)) * i / steps
            dt = T0.__class__(2020, 10, 31) + __import__('datetime').timedelta(days=t * 365.25)
            extra = max(0, t - to_years(DATA_END))
            ci    = CI_95 * (1 + extra * 0.5)
            row   = build_row(t)
            price = float(np.array(row) @ BETA)
            fc_upper.extend([sx(t), sy(price + ci)])
            fc_lower.insert(0, sy(price - ci))
            fc_lower.insert(0, sx(t))
        fc_poly = fc_upper + fc_lower
        if len(fc_poly) >= 4:
            self.create_polygon(*fc_poly, fill="#FF6B6B15", outline="", smooth=False)

        # Fitted + forecast curves
        pts_hist, pts_fore = [], []
        for i in range(steps + 1):
            t  = min_x + (max_x - min_x) * i / steps
            dt = T0.__class__(2020, 10, 31) + __import__('datetime').timedelta(days=t * 365.25)
            price = float(np.array(build_row(t)) @ BETA)
            xc, yc = sx(t), sy(price)
            if dt <= DATA_END:
                pts_hist.extend([xc, yc])
            else:
                if not pts_fore:
                    pts_fore.extend([xc, yc])
                pts_fore.extend([xc, yc])

        if len(pts_hist) >= 4:
            self.create_line(*pts_hist, fill=HIST_CLR, width=2, smooth=True)
        if len(pts_fore) >= 4:
            self.create_line(*pts_fore, fill=FORE_CLR, width=2.5,
                             smooth=True, dash=(8, 3))

        # Actual data dots
        for t, p in zip(_ts, _prices):
            xc, yc = sx(t), sy(p)
            self.create_oval(xc - 3, yc - 3, xc + 3, yc + 3,
                             fill=HIST_CLR, outline=BG, width=1)

        # Data-end divider
        xd = sx(to_years(DATA_END))
        self.create_line(xd, pad_t, xd, h - pad_b, fill=MUTED,
                         dash=(3, 3), width=1)
        self.create_text(xd + 4, pad_t + 2, text="data end ▶",
                         anchor="w", fill=MUTED, font=("Courier", 7))

        # Axis labels
        self.create_text(pad_l, h - 6, text="Oct 2020", anchor="w",
                         fill=MUTED, font=("Courier", 7))
        self.create_text(w - pad_r, h - 6, text="Sep 2025", anchor="e",
                         fill=MUTED, font=("Courier", 7))

        # Legend
        lx, ly = w - pad_r - 150, pad_t + 4
        self.create_line(lx, ly + 6,  lx + 20, ly + 6,  fill=HIST_CLR, width=2)
        self.create_text(lx + 24, ly + 6,  text="fitted",   anchor="w",
                         fill=HIST_CLR, font=("Courier", 8))
        self.create_line(lx, ly + 20, lx + 20, ly + 20, fill=FORE_CLR,
                         width=2.5, dash=(6, 2))
        self.create_text(lx + 24, ly + 20, text="forecast", anchor="w",
                         fill=FORE_CLR, font=("Courier", 8))
        self.create_rectangle(lx, ly + 32, lx + 12, ly + 40,
                              fill="#FF6B6B33", outline="")
        self.create_text(lx + 24, ly + 36, text="95% CI",   anchor="w",
                         fill=MUTED, font=("Courier", 8))

    def mark(self, query_dt, price, lo, hi):
        """Mark a queried date on the chart."""
        if self._marker:
            for item in self._marker:
                self.delete(item)
        t  = to_years(query_dt)
        xc = self._sx(t)
        yc = self._sy(price)
        ylo = self._sy(lo)
        yhi = self._sy(hi)
        color = FORE_CLR if query_dt > DATA_END else SUCCESS
        items = []
        items.append(self.create_line(xc, 0, xc, 9999,
                                      fill=color, dash=(4, 2), width=1))
        items.append(self.create_line(xc - 6, ylo, xc + 6, ylo,
                                      fill=WARNING, width=1.5))
        items.append(self.create_line(xc - 6, yhi, xc + 6, yhi,
                                      fill=WARNING, width=1.5))
        items.append(self.create_line(xc, ylo, xc, yhi,
                                      fill=WARNING, width=1, dash=(2, 2)))
        items.append(self.create_oval(xc - 7, yc - 7, xc + 7, yc + 7,
                                      fill=color, outline=BG, width=2))
        self._marker = items

    def clear_marker(self):
        if self._marker:
            for item in self._marker:
                self.delete(item)
            self._marker = None


# ── Main App ──────────────────────────────────────────────────────────────────
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Natural Gas Price Estimator")
        self.configure(bg=BG)
        self.resizable(True, True)
        self.minsize(680, 700)
        self._last_result = None
        self._build_ui()

    def _build_ui(self):
        # ── Header ────────────────────────────────────────────────────────────
        hdr = tk.Frame(self, bg=BG)
        hdr.pack(fill="x", padx=28, pady=(24, 0))

        tk.Label(hdr, text="NAT GAS", bg=BG, fg=ACCENT,
                 font=("Courier", 9, "bold")).pack(anchor="w")
        tk.Label(hdr, text="Price Estimator", bg=BG, fg=TEXT,
                 font=("Georgia", 24, "bold")).pack(anchor="w")
        tk.Label(hdr,
                 text=f"OLS trend + seasonality model  ·  Oct 2020 – Sep 2024  ·  R²={R2:.4f}",
                 bg=BG, fg=MUTED, font=("Courier", 9)).pack(anchor="w", pady=(2, 0))

        self._sep(self)

        # ── Chart ─────────────────────────────────────────────────────────────
        cf = tk.Frame(self, bg=PANEL, highlightbackground=BORDER,
                      highlightthickness=1)
        cf.pack(fill="x", padx=28, pady=(0, 12))
        self.chart = SparkChart(cf, width=620, height=180)
        self.chart.pack(fill="x", padx=2, pady=2)
        self.after(80, self.chart._draw_base)

        # ── Input row ─────────────────────────────────────────────────────────
        inp = tk.Frame(self, bg=BG)
        inp.pack(fill="x", padx=28)
        tk.Label(inp, text="ENTER DATE", bg=BG, fg=MUTED,
                 font=("Courier", 8, "bold")).pack(anchor="w")

        row = tk.Frame(inp, bg=BG)
        row.pack(fill="x", pady=(6, 0))

        spin_kw = dict(bg=PANEL, fg=TEXT, insertbackground=ACCENT,
                       relief="flat", font=("Courier", 14, "bold"),
                       highlightbackground=BORDER, highlightthickness=1,
                       buttonbackground=PANEL, bd=0)

        self._year  = tk.StringVar(value="2025")
        self._month = tk.StringVar(value="06")
        self._day   = tk.StringVar(value="15")

        for lbl, var, lo, hi, w in [
            ("YYYY", self._year,  2020, 2025, 5),
            ("MM",   self._month, 1,    12,   3),
            ("DD",   self._day,   1,    31,   3),
        ]:
            tk.Label(row, text=lbl, bg=BG, fg=MUTED,
                     font=("Courier", 7)).pack(side="left", padx=(0, 2))
            sb = tk.Spinbox(row, from_=lo, to=hi, width=w,
                            textvariable=var, **spin_kw,
                            command=self._on_spin)
            sb.pack(side="left")
            sb.bind("<KeyRelease>", lambda e: self._on_spin())
            if lbl != "DD":
                tk.Label(row, text=" – ", bg=BG, fg=MUTED,
                         font=("Courier", 14)).pack(side="left")

        self.btn = tk.Button(row, text="ESTIMATE  →",
                             bg=ACCENT, fg=BG,
                             font=("Courier", 11, "bold"),
                             relief="flat", cursor="hand2",
                             padx=16, pady=6,
                             activebackground=ACCENT2,
                             activeforeground="#fff",
                             command=self._estimate)
        self.btn.pack(side="left", padx=(18, 0))
        self.btn.bind("<Enter>", lambda e: self.btn.config(bg=ACCENT2, fg="#fff"))
        self.btn.bind("<Leave>", lambda e: self.btn.config(bg=ACCENT,  fg=BG))

        self._sep(self)

        # ── Result panel ──────────────────────────────────────────────────────
        res = tk.Frame(self, bg=PANEL, highlightbackground=BORDER,
                       highlightthickness=1)
        res.pack(fill="x", padx=28)
        inner = tk.Frame(res, bg=PANEL)
        inner.pack(padx=20, pady=14, fill="x")

        self.lbl_tag   = tk.Label(inner, text="AWAITING INPUT", bg=PANEL,
                                  fg=MUTED, font=("Courier", 9, "bold"))
        self.lbl_tag.pack(anchor="w")

        price_row = tk.Frame(inner, bg=PANEL)
        price_row.pack(anchor="w", pady=(2, 0))
        self.lbl_price = tk.Label(price_row, text="—", bg=PANEL, fg=TEXT,
                                  font=("Georgia", 40, "bold"))
        self.lbl_price.pack(side="left")
        tk.Label(price_row, text="/ MMBtu", bg=PANEL, fg=MUTED,
                 font=("Courier", 11)).pack(side="left", padx=(8, 0),
                                             anchor="s", pady=(0, 8))

        self.lbl_ci   = tk.Label(inner, text="", bg=PANEL, fg=WARNING,
                                 font=("Courier", 10))
        self.lbl_ci.pack(anchor="w")
        self.lbl_note = tk.Label(inner, text="", bg=PANEL, fg=MUTED,
                                 font=("Courier", 9), wraplength=520,
                                 justify="left")
        self.lbl_note.pack(anchor="w", pady=(4, 0))

        # Export button
        self.exp_btn = tk.Button(inner, text="⬇  Export to CSV",
                                 bg=PANEL, fg=MUTED,
                                 font=("Courier", 9),
                                 relief="flat", cursor="hand2",
                                 padx=10, pady=4,
                                 highlightbackground=BORDER,
                                 highlightthickness=1,
                                 state="disabled",
                                 command=self._export_csv)
        self.exp_btn.pack(anchor="w", pady=(10, 0))

        self._sep(self)

        # ── Model coefficients ────────────────────────────────────────────────
        coef_frame = tk.Frame(self, bg=BG)
        coef_frame.pack(padx=28, pady=(0, 20), fill="x")
        tk.Label(coef_frame, text="MODEL COEFFICIENTS & PERFORMANCE",
                 bg=BG, fg=MUTED, font=("Courier", 8, "bold")).pack(anchor="w",
                                                                      pady=(0, 8))
        cf_row = tk.Frame(coef_frame, bg=BG)
        cf_row.pack(fill="x")
        metrics = [
            ("INTERCEPT",  f"{BETA[0]:.4f}"),
            ("TREND /YR",  f"+${TREND_YR:.4f}"),
            ("AMPLITUDE",  f"±${AMPLITUDE:.4f}"),
            ("R²",         f"{R2:.4f}"),
            ("MAE",        f"${MAE:.4f}"),
            ("RMSE",       f"${RMSE:.4f}"),
            ("95% CI",     f"±${CI_95:.4f}"),
        ]
        for label, val in metrics:
            box = tk.Frame(cf_row, bg=PANEL, highlightbackground=BORDER,
                           highlightthickness=1)
            box.pack(side="left", padx=(0, 6))
            tk.Label(box, text=label, bg=PANEL, fg=MUTED,
                     font=("Courier", 7), padx=10, pady=4).pack(anchor="w")
            tk.Label(box, text=val, bg=PANEL, fg=ACCENT,
                     font=("Courier", 12, "bold"), padx=10).pack(anchor="w",
                                                                   pady=(0, 6))

        self.geometry("700x720")

    def _sep(self, parent):
        tk.Frame(parent, bg=BORDER, height=1).pack(fill="x", padx=28, pady=12)

    def _date_str(self):
        y = self._year.get().zfill(4)
        m = self._month.get().zfill(2)
        d = self._day.get().zfill(2)
        return f"{y}-{m}-{d}"

    def _on_spin(self):
        self.chart.clear_marker()
        self.lbl_tag.config(text="AWAITING INPUT", fg=MUTED)
        self.lbl_price.config(text="—", fg=TEXT)
        self.lbl_ci.config(text="")
        self.lbl_note.config(text="")
        self.exp_btn.config(state="disabled")
        self._last_result = None

    def _estimate(self):
        try:
            dt = datetime.strptime(self._date_str(), "%Y-%m-%d")
        except ValueError:
            messagebox.showerror("Invalid Date", "Please enter a valid date.")
            return
        try:
            price, lo, hi, is_fc = predict(dt)
        except ValueError as e:
            messagebox.showerror("Out of Range", str(e))
            return

        color = FORE_CLR if is_fc else SUCCESS
        tag   = "FORECAST  ·  beyond training data" if is_fc else "HISTORICAL ESTIMATE"
        note  = ("Extrapolated beyond Sep 2024. Confidence interval widens "
                 "with distance from training data."
                 if is_fc else
                 "Interpolated from fitted trend + seasonality model. "
                 "Within training window.")

        self.lbl_tag.config(text=tag, fg=color)
        self.lbl_price.config(text=f"${price:.2f}", fg=color)
        self.lbl_ci.config(text=f"95% CI:  ${lo:.2f}  –  ${hi:.2f}")
        self.lbl_note.config(text=note)
        self.chart.mark(dt, price, lo, hi)
        self.exp_btn.config(state="normal")
        self._last_result = {"date": dt, "price": price, "lo": lo,
                             "hi": hi, "is_forecast": is_fc}

    def _export_csv(self):
        if not self._last_result:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            initialfile=f"nat_gas_{self._last_result['date'].strftime('%Y%m%d')}.csv"
        )
        if not path:
            return
        r = self._last_result
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["date", "estimated_price", "lower_95",
                             "upper_95", "is_forecast", "model_r2",
                             "model_mae", "model_rmse"])
            writer.writerow([r["date"].date(), f"{r['price']:.4f}",
                             f"{r['lo']:.4f}", f"{r['hi']:.4f}",
                             r["is_forecast"], f"{R2:.4f}",
                             f"{MAE:.4f}", f"{RMSE:.4f}"])
        messagebox.showinfo("Exported", f"Saved to:\n{path}")


if __name__ == "__main__":
    App().mainloop()
