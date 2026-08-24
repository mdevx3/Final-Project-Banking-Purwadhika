"""Grafik Plotly dengan gaya yang seragam di seluruh aplikasi."""
from __future__ import annotations

import numpy as np
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
