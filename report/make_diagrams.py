"""Draw the two explanatory diagrams of the report (no data needed):

figures/diagram_pipeline.png  - how the project works, from raw data to recommendations
figures/diagram_timeline.png  - which dates each cohort may use (observation cutoff)
"""
import os
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

R = Path(os.environ.get("REPORT_DIR", Path(__file__).resolve().parent))
F = R / "figures"
F.mkdir(parents=True, exist_ok=True)

INK, MUTED = "#1F2933", "#52606D"
STAGES = [
    ("1. Dữ liệu gốc", "#E8EEF7", "#3B6FB6", [
        "Thông tin thành viên", "Lịch sử giao dịch", "Nhật ký nghe nhạc\n(392 triệu dòng, 28 GB)",
        "Nhãn rời bỏ\ntháng 2 và tháng 3"]),
    ("2. Chuẩn bị dữ liệu", "#E7F4EC", "#2F855A", [
        "Cắt dữ liệu tại\nmốc quan sát", "Làm sạch\n(tuổi, giới tính, giờ nghe)",
        "Tạo 41 đặc trưng\ncho mỗi người dùng", "Tạo chuỗi giao dịch\ncho mỗi người dùng"]),
    ("3. Khai phá dữ liệu", "#FDF1E3", "#C05621", [
        "Phân tích khám phá", "Phân cụm (K-Means)", "Luật kết hợp (FP-Growth)",
        "Mẫu tuần tự (PrefixSpan)", "Phân lớp (LightGBM...)", "Phát hiện bất thường"]),
    ("4. Kiểm chứng", "#F3E8F7", "#805AD5", [
        "Đo lại trên 30% dữ liệu\nkhông dùng để khai phá", "Lặp lại trên tháng 2",
        "Kiểm tra rò rỉ dữ liệu", "So sánh các thuật toán"]),
    ("5. Kết quả", "#FDEBEC", "#C53030", [
        "7 nhóm khách hàng", "Các tổ hợp và chuỗi\ndẫn tới rời bỏ",
        "Danh sách khách\nrủi ro cao", "Khuyến nghị\ngiữ chân khách hàng"]),
]


def pipeline():
    fig, ax = plt.subplots(figsize=(15, 7.2))
    ax.set_xlim(0, 15); ax.set_ylim(0, 7.2); ax.axis("off")
    w, gap, x0 = 2.55, 0.42, 0.25
    for i, (title, fill, edge, items) in enumerate(STAGES):
        x = x0 + i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, 0.35), w, 6.1, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc=fill, ec=edge, lw=1.6))
        ax.text(x + w / 2, 6.1, title, ha="center", va="center", fontsize=13, weight="bold", color=edge)
        n = len(items)
        top, bottom = 5.55, 0.65
        h = min(0.82, (top - bottom - 0.12 * (n - 1)) / n)
        block = n * h + (n - 1) * 0.12
        start = (top + bottom) / 2 + block / 2          # centre the stack vertically
        for k, it in enumerate(items):
            y = start - (k + 1) * h - k * 0.12
            ax.add_patch(FancyBboxPatch((x + 0.15, y), w - 0.3, h, boxstyle="round,pad=0.01,rounding_size=0.08",
                                        fc="white", ec=edge, lw=0.9))
            ax.text(x + w / 2, y + h / 2, it, ha="center", va="center", fontsize=9.5, color=INK)
        if i < len(STAGES) - 1:
            ax.add_patch(FancyArrowPatch((x + w + 0.03, 3.4), (x + w + gap - 0.03, 3.4), arrowstyle="-|>",
                                         mutation_scale=22, lw=2, color=MUTED))
    ax.text(7.5, 6.95, "Quy trình thực hiện dự án", ha="center", va="center", fontsize=16, weight="bold", color=INK)
    fig.savefig(F / "diagram_pipeline.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def timeline():
    fig, ax = plt.subplots(figsize=(14, 4.8))
    rows = [
        ("Tháng 2\n(dùng để huấn luyện)", 1.0, date(2016, 12, 3), date(2017, 1, 31), date(2017, 2, 1), date(2017, 3, 31)),
        ("Tháng 3\n(dùng để kiểm tra)", 0.0, date(2016, 12, 31), date(2017, 2, 28), date(2017, 3, 1), date(2017, 4, 30)),
    ]
    for label, y, ws, cut, es, ee in rows:
        ax.barh(y, mdates.date2num(cut) - mdates.date2num(date(2016, 11, 1)), left=mdates.date2num(date(2016, 11, 1)),
                height=0.5, color="#C6DBF2", edgecolor="#3B6FB6")
        ax.barh(y, mdates.date2num(cut) - mdates.date2num(ws), left=mdates.date2num(ws), height=0.5,
                color="#3B6FB6", edgecolor="#3B6FB6")
        ax.barh(y, mdates.date2num(ee) - mdates.date2num(es), left=mdates.date2num(es), height=0.5,
                color="#FBD5D5", edgecolor="#C53030", hatch="//")
        ax.axvline(mdates.date2num(cut), ymin=0, ymax=1, color="#C53030", lw=0)
        ax.plot([mdates.date2num(cut)] * 2, [y - 0.38, y + 0.38], color="#C53030", lw=2.5)
        ax.text(mdates.date2num(cut), y + 0.45, f"mốc quan sát {cut:%d/%m/%Y}", ha="center", fontsize=9.5,
                color="#C53030", weight="bold")
        ax.text(mdates.date2num(ws) + 30, y, "60 ngày nghe nhạc", ha="center", va="center", fontsize=9, color="white",
                weight="bold")
        ax.text((mdates.date2num(es) + mdates.date2num(ee)) / 2, y,
                "thuê bao hết hạn, chờ 30 ngày\nxem có gia hạn không → nhãn", ha="center", va="center", fontsize=8.5,
                color="#7B1E1E", bbox=dict(fc="white", ec="none", pad=2))
        ax.text(mdates.date2num(date(2016, 11, 3)), y, "← lịch sử từ 2015", ha="left", va="center",
                fontsize=8.5, color="#3B6FB6")
    ax.set_yticks([1, 0], [r[0] for r in rows], fontsize=10)
    ax.set_ylim(-0.7, 1.75)
    ax.xaxis.set_major_locator(mdates.MonthLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%Y"))
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, fc="#C6DBF2", ec="#3B6FB6"),
               plt.Rectangle((0, 0), 1, 1, fc="#3B6FB6"),
               plt.Rectangle((0, 0), 1, 1, fc="#FBD5D5", ec="#C53030", hatch="//")]
    ax.legend(handles, ["lịch sử giao dịch và hồ sơ (được dùng)", "nhật ký nghe nhạc (được dùng)",
                        "sau mốc quan sát (KHÔNG được dùng để tạo đặc trưng)"],
              loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False, fontsize=9)
    ax.set_title("Mỗi nhóm khách hàng chỉ được dùng dữ liệu trước mốc quan sát của mình", fontsize=13,
                 weight="bold", color=INK)
    fig.savefig(F / "diagram_timeline.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    pipeline()
    timeline()
    print("diagrams written to", F)
