import pandas as pd
import streamlit as st
from difflib import get_close_matches
from pathlib import Path

# Optional (fallback vẽ sankey nếu không có file html)
try:
    import plotly.graph_objects as go
    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False

import numpy as np
import matplotlib.pyplot as plt

# =========================
# Page config
# =========================
st.set_page_config(
    page_title="Financial State Tracker",
    page_icon="📈",
    layout="wide"
)

# =========================
# Styling
# =========================
st.markdown("""
<style>
.small-note {color:#64748b; font-size: 0.92rem;}
.badge {display:inline-block; padding:4px 10px; border-radius:999px; font-weight:600; font-size:0.9rem;}
.badge-weak {background:#fee2e2; color:#991b1b;}
.badge-mid  {background:#fef3c7; color:#92400e;}
.badge-good {background:#dcfce7; color:#166534;}
.card {border:1px solid #e2e8f0; border-radius:14px; padding:14px 16px; background:#ffffff;}
hr {border:none; border-top:1px solid #e2e8f0; margin: 12px 0;}
</style>
""", unsafe_allow_html=True)

STATE_LABEL = {0: "Yếu", 1: "Trung bình", 2: "Tốt"}
STATE_BADGE_CLASS = {0: "badge-weak", 1: "badge-mid", 2: "badge-good"}

FIG_DIR = Path("outputs/figures")
TABLE_DIR = Path("outputs/tables")

# =========================
# Load data
# =========================
@st.cache_data
def load_labels():
    df = pd.read_csv(TABLE_DIR / "cluster_labels.csv")

    # Industry stats
    stats = (
        df.groupby(["Ngành ICB - cấp 1", "Năm"])["Composite_Score"]
          .agg(industry_mean="mean", industry_median="median")
          .reset_index()
    )
    df = df.merge(stats, on=["Ngành ICB - cấp 1", "Năm"], how="left")
    df["gap_vs_industry_median"] = df["Composite_Score"] - df["industry_median"]

    # Rank in industry per year
    df["rank_in_industry"] = (
        df.groupby(["Ngành ICB - cấp 1", "Năm"])["Composite_Score"]
          .rank(method="dense", ascending=False)
          .astype(int)
    )

    return df

@st.cache_data
def load_migration():
    path = TABLE_DIR / "migration_records.csv"
    if not path.exists():
        return None
    m = pd.read_csv(path)
    # kỳ vọng có: Mã, Năm, Ngành ICB - cấp 1, cluster, cluster_next
    return m

def build_transition_matrix(mdf: pd.DataFrame, industry: str, year_t: int, normalize="row"):
    """
    normalize:
      - "row": mỗi hàng sum=1 (xác suất chuyển từ trạng thái t sang t+1)
      - "none": số lượng tuyệt đối
    """
    tmp = mdf[(mdf["Ngành ICB - cấp 1"] == industry) & (mdf["Năm"] == year_t)].copy()
    if tmp.empty:
        return None, tmp

    flow = (
        tmp.groupby(["cluster", "cluster_next"])
           .size()
           .reset_index(name="value")
    )

    mat = (
        flow.pivot(index="cluster", columns="cluster_next", values="value")
            .fillna(0.0)
            .reindex(index=[0,1,2], columns=[0,1,2], fill_value=0.0)
    )

    if normalize == "row":
        row_sum = mat.sum(axis=1).replace(0, np.nan)
        mat = mat.div(row_sum, axis=0).fillna(0.0)

    return mat, flow

def render_heatmap(mat: pd.DataFrame, title: str):
    fig = plt.figure(figsize=(6.5, 4.8))
    ax = plt.gca()
    im = ax.imshow(mat.values, aspect="auto")
    ax.set_xticks([0,1,2])
    ax.set_yticks([0,1,2])
    ax.set_xticklabels(["0-Yếu", "1-TB", "2-Tốt"])
    ax.set_yticklabels(["0-Yếu", "1-TB", "2-Tốt"])
    ax.set_xlabel("Trạng thái năm t+1")
    ax.set_ylabel("Trạng thái năm t")
    ax.set_title(title)

    # annotate
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{mat.values[i,j]:.2f}", ha="center", va="center", color="black")

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    st.pyplot(fig)

def try_embed_sankey_html(industry: str, year_t: int, year_t1: int):
    """
    Ưu tiên embed file html đã export sẵn: outputs/figures/sankey_<industry>_<y0>_<y1>.html
    """
    safe_ind = industry.replace(" ", "_")
    # bạn có file kiểu sankey_Công_nghiệp_2022_2023.html -> giữ đúng theo pattern bạn đã dùng
    candidates = [
        FIG_DIR / f"sankey_{industry}_{year_t}_{year_t1}.html",
        FIG_DIR / f"sankey_{safe_ind}_{year_t}_{year_t1}.html",
    ]

    for p in candidates:
        if p.exists():
            html = p.read_text(encoding="utf-8", errors="ignore")
            st.components.v1.html(html, height=650, scrolling=True)
            return True, str(p)

    # nếu tên file của bạn có dấu, đôi khi windows/path khác; fallback: tìm chứa năm
    if FIG_DIR.exists():
        hits = list(FIG_DIR.glob(f"*{year_t}_{year_t1}*.html"))
        # ưu tiên file có "sankey"
        hits = sorted(hits, key=lambda x: ("sankey" not in x.name.lower(), x.name))
        for p in hits[:3]:
            html = p.read_text(encoding="utf-8", errors="ignore")
            st.components.v1.html(html, height=650, scrolling=True)
            return True, str(p)

    return False, None

def render_sankey_plotly(flow: pd.DataFrame, industry: str, year_t: int, year_t1: int):
    """
    Vẽ sankey trực tiếp (fallback khi không có HTML).
    flow: columns ['cluster','cluster_next','value'] (count)
    """
    if not PLOTLY_OK:
        st.warning("Không thể vẽ Sankey trực tiếp vì thiếu plotly. Hãy cài: pip install plotly")
        return

    labels = ["Yếu (t)", "TB (t)", "Tốt (t)", "Yếu (t+1)", "TB (t+1)", "Tốt (t+1)"]
    source_map = {0: 0, 1: 1, 2: 2}
    target_map = {0: 3, 1: 4, 2: 5}

    sources = flow["cluster"].map(source_map).astype(int).tolist()
    targets = flow["cluster_next"].map(target_map).astype(int).tolist()
    values  = flow["value"].astype(int).tolist()

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=14,
            thickness=18,
            label=labels,
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values
        )
    )])

    fig.update_layout(
        title_text=f"Dòng dịch chuyển trạng thái tài chính ({industry}) {year_t} → {year_t1}",
        font_size=12,
        height=650
    )
    st.plotly_chart(fig, use_container_width=True)

# =========================
# App header
# =========================
df = load_labels()
migrate = load_migration()

industries = sorted(df["Ngành ICB - cấp 1"].dropna().unique().tolist())
all_years = sorted(df["Năm"].dropna().unique().tolist())

st.title("Financial State Tracker — Theo dõi trạng thái tài chính nội ngành")
st.caption("Mục tiêu: so sánh vị thế tài chính *tương đối* của doanh nghiệp với các doanh nghiệp **cùng ngành**, theo từng năm, và theo dõi dịch chuyển trạng thái.")

with st.expander("📌 Cách dùng nhanh (30 giây)", expanded=True):
    st.markdown("""
1) **Chọn ngành** ở sidebar  
2) **Nhập mã** (Search) hoặc chọn mã từ danh sách  
3) Xem **Trạng thái năm gần nhất**, **Rank nội ngành**, **Composite vs Median**, và tab **Dịch chuyển** để xem ma trận chuyển trạng thái.
""")

# =========================
# Sidebar
# =========================
st.sidebar.header("Bộ lọc")

industry = st.sidebar.selectbox("Chọn ngành", industries)

tickers = sorted(df[df["Ngành ICB - cấp 1"] == industry]["Mã"].dropna().unique().tolist())
default_ticker = tickers[0] if tickers else ""

search = st.sidebar.text_input(
    "Nhập mã (Search)",
    value=default_ticker,
    help="Gõ mã cổ phiếu (VD: BMP, NAV...). Nếu sai, hệ thống gợi ý mã gần giống."
).strip().upper()

if search and search not in tickers:
    suggestion = get_close_matches(search, tickers, n=5, cutoff=0.4)
    if suggestion:
        st.sidebar.warning(f"Không thấy mã **{search}** trong ngành này. Gợi ý: {', '.join(suggestion)}")
    else:
        st.sidebar.warning(f"Không thấy mã **{search}** trong ngành này.")
    search = default_ticker

ticker = st.sidebar.selectbox("Chọn mã", tickers, index=(tickers.index(search) if search in tickers else 0))

st.sidebar.markdown("---")
with st.sidebar.expander("🧾 Giải thích chỉ số", expanded=False):
    st.markdown("""
- **Composite_Score**: điểm tổng hợp (chuẩn hoá nội ngành theo năm).  
- **Rank nội ngành**: 1 là tốt nhất trong ngành năm đó.  
- **Median ngành**: trung vị điểm trong ngành năm đó.  
- **Gap = Composite − Median**: dương → cao hơn mặt bằng ngành; âm → thấp hơn.  
- **Trạng thái (cluster)**: 0 (Yếu), 1 (Trung bình), 2 (Tốt) dựa trên phân cụm nội ngành.
""")

with st.sidebar.expander("⚠️ Lưu ý diễn giải", expanded=False):
    st.markdown("""
- So sánh **chỉ hợp lệ trong cùng ngành, cùng năm** (vì đã chuẩn hoá).  
- Thiếu năm = DN thiếu đủ chỉ số để phân cụm năm đó (không phải lỗi).  
""")

# =========================
# Slice company data
# =========================
d = df[(df["Ngành ICB - cấp 1"] == industry) & (df["Mã"] == ticker)].copy().sort_values("Năm")
last = d.iloc[-1]

company_name = d["Tên công ty"].iloc[-1] if "Tên công ty" in d.columns else ""
exchange = d["Sàn"].iloc[-1] if "Sàn" in d.columns else ""

years_present = d["Năm"].astype(int).tolist()
missing_years = [int(y) for y in all_years if int(y) not in years_present]

cluster = int(last["cluster"])
state_text = STATE_LABEL.get(cluster, str(cluster))
badge_class = STATE_BADGE_CLASS.get(cluster, "badge-mid")

# =========================
# Layout
# =========================
left, right = st.columns([1.15, 2.25], gap="large")

with left:
    st.markdown(f"""
<div class="card">
  <h3 style="margin:0;">{ticker} — {company_name}</h3>
  <div class="small-note">Sàn: {exchange} • Ngành: {industry}</div>
  <hr/>
  <div><b>Năm gần nhất:</b> {int(last['Năm'])}</div>
  <div style="margin-top:6px;"><b>Trạng thái:</b> <span class="badge {badge_class}">{cluster} — {state_text}</span></div>
  <div style="margin-top:6px;"><b>Rank nội ngành:</b> {int(last['rank_in_industry'])}</div>
  <div style="margin-top:6px;"><b>Composite:</b> {last['Composite_Score']:.3f}</div>
  <div style="margin-top:6px;"><b>Median ngành:</b> {last['industry_median']:.3f}</div>
  <div style="margin-top:6px;"><b>Gap:</b> {last['gap_vs_industry_median']:.3f}</div>
</div>
""", unsafe_allow_html=True)

    if missing_years:
        st.info(f"DN thiếu dữ liệu các năm: {missing_years} (không đủ chỉ số để phân cụm).")

with right:
    tabs = st.tabs(["📌 Doanh nghiệp", "🏭 So sánh nội ngành", "🔁 Dịch chuyển (Sankey/Heatmap)", "⬇️ Tải dữ liệu"])

    # -------------------------
    # Tab 1: Company
    # -------------------------
    with tabs[0]:
        st.subheader("Bảng theo năm (trajectory)")
        d_show = d[["Năm", "cluster", "rank_in_industry", "Composite_Score", "industry_median", "gap_vs_industry_median"]].copy()
        d_show["state"] = d_show["cluster"].map(STATE_LABEL)
        d_show = d_show[["Năm", "state", "rank_in_industry", "Composite_Score", "industry_median", "gap_vs_industry_median"]]
        st.dataframe(d_show, use_container_width=True)

        st.subheader("Quỹ đạo vị thế nội ngành (Composite vs Median ngành)")
        st.line_chart(d.set_index("Năm")[["Composite_Score", "industry_median"]], height=320)
        st.caption("Composite > Median: doanh nghiệp vượt mặt bằng ngành ở năm đó. Composite < Median: thấp hơn mặt bằng ngành.")

    # -------------------------
    # Tab 2: Industry comparison
    # -------------------------
    with tabs[1]:
        last_year = int(last["Năm"])
        st.subheader(f"Top 5 DN trong ngành (năm {last_year})")
        top5 = (
            df[(df["Ngành ICB - cấp 1"] == industry) & (df["Năm"] == last_year)]
            .sort_values("Composite_Score", ascending=False)
            .head(5)[["Mã", "Tên công ty", "Composite_Score", "rank_in_industry", "cluster"]]
            .copy()
        )
        top5["state"] = top5["cluster"].map(STATE_LABEL)
        top5 = top5[["Mã", "Tên công ty", "Composite_Score", "rank_in_industry", "state"]]
        st.dataframe(top5, use_container_width=True)

        total = len(df[(df["Ngành ICB - cấp 1"] == industry) & (df["Năm"] == last_year)])
        rank = int(last["rank_in_industry"])
        st.markdown(f"**Vị thế hiện tại:** Doanh nghiệp đang đứng **hạng {rank}/{total}** trong ngành năm **{last_year}**.")

    # -------------------------
    # Tab 3: Migration
    # -------------------------
    with tabs[2]:
        st.subheader("Dịch chuyển trạng thái tài chính theo ngành (t → t+1)")
        st.markdown("""
**Cách đọc nhanh:**
- **Ma trận/Heatmap:** Hàng = trạng thái năm *t*, cột = trạng thái năm *t+1*  
- **Đường chéo** (0→0, 1→1, 2→2): ổn định  
- **Trên đường chéo** (0→1/2, 1→2): cải thiện  
- **Dưới đường chéo** (2→1/0, 1→0): suy giảm
""")

        if migrate is None:
            st.error("Không tìm thấy outputs/tables/migration_records.csv. Hãy export file này từ notebook migration.")
        else:
            years_t = sorted(migrate[migrate["Ngành ICB - cấp 1"] == industry]["Năm"].dropna().unique().tolist())
            years_t = [int(y) for y in years_t]
            if not years_t:
                st.warning("Ngành này chưa có migration records.")
            else:
                colA, colB, colC = st.columns([1,1,1.2])
                with colA:
                    year_t = st.selectbox("Chọn năm t", years_t, index=len(years_t)-1)
                with colB:
                    year_t1 = year_t + 1
                    st.text_input("Năm t+1", value=str(year_t1), disabled=True)
                with colC:
                    mode = st.selectbox("Kiểu hiển thị", ["Chuẩn hoá theo hàng (xác suất)", "Số lượng tuyệt đối"])

                normalize = "row" if mode.startswith("Chuẩn") else "none"
                mat, flow = build_transition_matrix(migrate, industry, year_t, normalize=normalize)

                if mat is None:
                    st.warning("Không có dữ liệu migration cho cặp năm đã chọn.")
                else:
                    st.markdown("### 1) Heatmap ma trận chuyển trạng thái")
                    title = f"Ma trận chuyển trạng thái — {industry} ({year_t} → {year_t1})"
                    render_heatmap(mat, title=title)

                    st.markdown("### 2) Sankey (luồng dịch chuyển)")
                    ok, used_file = try_embed_sankey_html(industry, year_t, year_t1)
                    if ok:
                        st.caption(f"Đang dùng Sankey HTML đã export: {used_file}")
                    else:
                        st.caption("Không tìm thấy file Sankey HTML phù hợp → vẽ trực tiếp (fallback).")
                        render_sankey_plotly(flow, industry, year_t, year_t1)

                    st.markdown("### 3) Bảng luồng (flow table)")
                    flow_show = flow.copy()
                    flow_show["from_state"] = flow_show["cluster"].map(STATE_LABEL)
                    flow_show["to_state"] = flow_show["cluster_next"].map(STATE_LABEL)
                    flow_show = flow_show[["cluster","from_state","cluster_next","to_state","value"]].sort_values(["cluster","cluster_next"])
                    st.dataframe(flow_show, use_container_width=True)

                    st.markdown("### 4) Tải dữ liệu migration (CSV)")
                    st.download_button(
                        "Tải ma trận (CSV)",
                        data=mat.reset_index().to_csv(index=False).encode("utf-8-sig"),
                        file_name=f"transition_matrix_{industry}_{year_t}_{year_t1}.csv",
                        mime="text/csv"
                    )
                    st.download_button(
                        "Tải flow table (CSV)",
                        data=flow_show.to_csv(index=False).encode("utf-8-sig"),
                        file_name=f"flow_{industry}_{year_t}_{year_t1}.csv",
                        mime="text/csv"
                    )

    # -------------------------
    # Tab 4: Download
    # -------------------------
    with tabs[3]:
        st.subheader("Tải dữ liệu doanh nghiệp đang xem")
        d_show = d[["Năm", "cluster", "rank_in_industry", "Composite_Score", "industry_median", "gap_vs_industry_median"]].copy()
        d_show["state"] = d_show["cluster"].map(STATE_LABEL)
        d_show = d_show[["Năm", "state", "rank_in_industry", "Composite_Score", "industry_median", "gap_vs_industry_median"]]

        st.download_button(
            "Tải trajectory doanh nghiệp (CSV)",
            data=d_show.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"{ticker}_{industry}_trajectory.csv",
            mime="text/csv"
        )

        st.subheader("Tải dataset tổng (cho người dùng nghiên cứu)")
        st.download_button(
            "Tải cluster_labels.csv",
            data=df.to_csv(index=False).encode("utf-8-sig"),
            file_name="cluster_labels_enriched.csv",
            mime="text/csv"
        )

        if migrate is not None:
            st.download_button(
                "Tải migration_records.csv",
                data=migrate.to_csv(index=False).encode("utf-8-sig"),
                file_name="migration_records.csv",
                mime="text/csv"
            )

st.markdown("---")
st.caption("Gợi ý diễn giải: Composite ~ 0 ≈ gần trung vị ngành; Composite > 0 vượt ngành; Composite < 0 kém ngành. Rank=1 là tốt nhất nội ngành năm đó.")