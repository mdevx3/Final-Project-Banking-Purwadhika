"""Grafik Plotly dengan gaya yang seragam di seluruh aplikasi."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from . import config as C

TATA_LETAK = dict(
    margin=dict(l=10, r=10, t=50, b=10),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    hoverlabel=dict(font_size=13),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)


def _rapikan(fig: go.Figure, judul: str = "", tinggi: int = 380) -> go.Figure:
    fig.update_layout(title=judul, height=tinggi, **TATA_LETAK)
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.20)", zeroline=False)
    return fig


def gauge_probabilitas(prob: float, ambang: float) -> go.Figure:
    """Meteran probabilitas dengan ambang operasi sebagai garis acuan."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={"suffix": "%", "font": {"size": 40}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": C.HIJAU if prob >= ambang else C.MERAH, "thickness": 0.75},
            "steps": [
                {"range": [0, ambang * 100], "color": "rgba(178,34,34,0.12)"},
                {"range": [ambang * 100, 100], "color": "rgba(46,139,87,0.12)"},
            ],
            "threshold": {"line": {"color": "black", "width": 3},
                          "thickness": 0.9, "value": ambang * 100},
        },
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=10),
                      paper_bgcolor="rgba(0,0,0,0)")
    return fig


def bar_konversi(tabel: pd.DataFrame, kolom_x: str, baseline: float,
                 judul: str, label_x: str) -> go.Figure:
    """Conversion rate per kategori, dibandingkan garis baseline populasi."""
    warna = [C.HIJAU if v > baseline else C.MERAH for v in tabel["conv_rate"]]
    fig = go.Figure()
    fig.add_bar(
        x=tabel[kolom_x].astype(str), y=tabel["conv_rate"], marker_color=warna,
        text=[f"{v:.1f}%" for v in tabel["conv_rate"]], textposition="outside",
        customdata=np.stack([tabel["volume"], tabel["deposan"], tabel["lift"]], axis=-1),
        hovertemplate=("<b>%{x}</b><br>Conversion rate: %{y:.2f}%<br>"
                       "Volume panggilan: %{customdata[0]:,}<br>"
                       "Deposan: %{customdata[1]:,}<br>"
                       "Lift: %{customdata[2]:.2f}x<extra></extra>"),
        name="Conversion rate",
    )
    fig.add_hline(y=baseline, line_dash="dash", line_color=C.BIRU,
                  annotation_text=f"Baseline {baseline:.2f}%", annotation_position="top left")
    fig.update_yaxes(title="Conversion rate (%)",
                     range=[0, max(tabel["conv_rate"].max(), baseline) * 1.25])
    fig.update_xaxes(title=label_x)
    return _rapikan(fig, judul, 420)


def bar_volume_vs_hasil(tabel: pd.DataFrame, kolom_x: str, judul: str) -> go.Figure:
    """Bandingkan porsi beban panggilan dengan porsi deposan yang dihasilkan."""
    fig = go.Figure()
    fig.add_bar(x=tabel[kolom_x].astype(str), y=tabel["pct_volume"],
                name="% dari total panggilan", marker_color=C.BIRU)
    fig.add_bar(x=tabel[kolom_x].astype(str), y=tabel["pct_deposan"],
                name="% dari total deposan", marker_color=C.HIJAU)
    fig.update_yaxes(title="Persentase (%)")
    fig.update_layout(barmode="group")
    return _rapikan(fig, judul, 420)


def kurva_gain_plot(seri: dict[str, tuple[np.ndarray, np.ndarray]],
                    penanda: list[int] | None = None) -> go.Figure:
    """Cumulative gain beberapa strategi dalam satu bidang."""
    warna = {"Model machine learning": C.HIJAU, "Skor prioritas manual": C.BIRU}
    fig = go.Figure()
    for nama, (x, y) in seri.items():
        # 400 titik sudah cukup mulus dan jauh lebih ringan digambar
        langkah = max(len(x) // 400, 1)
        fig.add_scatter(x=x[::langkah], y=y[::langkah], mode="lines", name=nama,
                        line=dict(color=warna.get(nama, C.ABU), width=3),
                        hovertemplate="%{x:.0f}% dihubungi -> %{y:.1f}% deposan<extra>"
                                      + nama + "</extra>")
    fig.add_scatter(x=[0, 100], y=[0, 100], mode="lines", name="Tanpa seleksi (acak)",
                    line=dict(color=C.ABU, width=2, dash="dash"),
                    hovertemplate="%{x:.0f}% dihubungi -> %{y:.1f}% deposan<extra>Acak</extra>")

    if penanda:
        x_m, y_m = seri["Model machine learning"]
        for p in penanda:
            k = min(int(len(x_m) * p / 100), len(x_m) - 1)
            fig.add_scatter(x=[p], y=[y_m[k]], mode="markers",
                            marker=dict(size=9, color="black", symbol="circle"),
                            showlegend=False, hoverinfo="skip")
            # Label ditaruh di atas titik dengan latar putih supaya tidak menimpa kurva
            fig.add_annotation(x=p, y=y_m[k], text=f"<b>{y_m[k]:.0f}%</b>",
                               showarrow=False, yshift=16, font=dict(size=12, color="#1a1a1a"),
                               bgcolor="rgba(255,255,255,.85)", borderpad=2)
    fig.update_xaxes(title="Persentase nasabah dihubungi (diurutkan dari skor tertinggi)",
                     range=[0, 100])
    fig.update_yaxes(title="Persentase deposan tertangkap (%)", range=[0, 101])
    return _rapikan(fig, "Cumulative Gain — berapa deposan tertangkap per kapasitas", 460)


def kurva_roc(fpr, tpr, auc: float) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(x=fpr, y=tpr, mode="lines", name=f"Model (AUC {auc:.4f})",
                    line=dict(color=C.HIJAU, width=3))
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Acak (AUC 0,5)",
                    line=dict(color=C.ABU, width=2, dash="dash"))
    fig.update_xaxes(title="False Positive Rate")
    fig.update_yaxes(title="True Positive Rate (Recall)")
    return _rapikan(fig, "ROC Curve", 400)


def kurva_pr(rec, prec, ap: float, base_rate: float) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(x=rec, y=prec, mode="lines", name=f"Model (PR-AUC {ap:.4f})",
                    line=dict(color=C.BIRU, width=3))
    fig.add_hline(y=base_rate, line_dash="dash", line_color=C.ABU,
                  annotation_text=f"Acak = base rate ({base_rate:.4f})")
    fig.update_xaxes(title="Recall")
    fig.update_yaxes(title="Precision")
    return _rapikan(fig, "Precision-Recall Curve", 400)


def bar_kepentingan(tabel: pd.DataFrame) -> go.Figure:
    t = tabel.sort_values("importance")
    warna = [C.HIJAU if v > 0.001 else C.MERAH for v in t["importance"]]
    fig = go.Figure(go.Bar(
        x=t["importance"], y=t["fitur"], orientation="h", marker_color=warna,
        error_x=dict(type="data", array=t["std"], color="rgba(80,80,80,0.6)"),
        hovertemplate="<b>%{y}</b><br>Penurunan F2: %{x:.5f}<extra></extra>"))
    fig.update_xaxes(title="Penurunan F2 ketika kolom diacak")
    return _rapikan(fig, "Permutation Importance pada data uji", 520)


def garis_ekonomi(ekonomi: pd.DataFrame, nilai_hubungi_semua: float,
                  ambang_f2: float, ambang_terbaik: float) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(x=ekonomi["ambang"], y=ekonomi["nilai_bersih"], mode="lines+markers",
                    name="Nilai bersih model", line=dict(color=C.HIJAU, width=3),
                    marker=dict(size=5),
                    hovertemplate="Ambang %{x:.2f}<br>Nilai bersih EUR %{y:,.0f}<extra></extra>")
    fig.add_hline(y=nilai_hubungi_semua, line_dash="dash", line_color=C.MERAH,
                  annotation_text=f"Hubungi semua: EUR {nilai_hubungi_semua:,.0f}")
    fig.add_vline(x=ambang_f2, line_dash="dot", line_color=C.BIRU,
                  annotation_text=f"Ambang F2 {ambang_f2:.2f}", annotation_position="top right")
    fig.add_vline(x=ambang_terbaik, line_dash="dot", line_color="black",
                  annotation_text=f"Nilai tertinggi {ambang_terbaik:.2f}",
                  annotation_position="bottom right")
    fig.update_xaxes(title="Ambang probabilitas")
    fig.update_yaxes(title="Nilai bersih (EUR)")
    return _rapikan(fig, "Nilai bersih per ambang vs strategi hubungi semua", 430)


def garis_f2(kurva: pd.DataFrame, ambang_dipakai: float) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(x=kurva["ambang"], y=kurva["f2"], mode="lines",
                    line=dict(color=C.HIJAU, width=3), name="F2",
                    hovertemplate="Ambang %{x:.2f} -> F2 %{y:.4f}<extra></extra>")
    puncak = kurva.loc[kurva["f2"].idxmax()]
    fig.add_scatter(x=[puncak["ambang"]], y=[puncak["f2"]], mode="markers",
                    marker=dict(size=12, color="black"), name="F2 tertinggi",
                    hovertemplate="Puncak: ambang %{x:.2f}, F2 %{y:.4f}<extra></extra>")
    fig.add_vline(x=ambang_dipakai, line_dash="dot", line_color=C.BIRU,
                  annotation_text=f"Ambang dipakai {ambang_dipakai:.2f}")
    fig.update_xaxes(title="Ambang probabilitas")
    fig.update_yaxes(title="F2 pada data uji")
    return _rapikan(fig, "F2 di sepanjang ambang", 400)
