import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================
# 1. 파일 업로드
# =========================
from google.colab import files
uploaded = files.upload()

# 업로드한 파일명 입력
file_name = list(uploaded.keys())[0]   # 자동으로 첫 파일 사용
# file_name = '260405_ramsey_ch2_A_0.3_1000rep_summary_results'   # 직접 적고 싶으면 이 줄 사용

# =========================
# 2. 데이터 읽기
# =========================
df = pd.read_excel(file_name)

# 필요한 열만 확인
print(df.columns)

# =========================
# 3. 시간 형식 변환
# 현재 형식 예: 03-4-2026 17:46:24
# dayfirst=True 꼭 넣기
# =========================
df['ModifiedDate'] = pd.to_datetime(df['ModifiedDate'], dayfirst=True, errors='coerce')

# 숫자형 변환
df['FFT_Peak_MHz'] = pd.to_numeric(df['FFT_Peak_MHz'], errors='coerce')

# 결측 제거
df = df.dropna(subset=['ModifiedDate', 'FFT_Peak_MHz']).copy()

# 시간순 정렬
df = df.sort_values('ModifiedDate').reset_index(drop=True)

print(df[['ModifiedDate', 'FFT_Peak_MHz']].head())
print(df[['ModifiedDate', 'FFT_Peak_MHz']].tail())

# =========================
# 4. 이동평균 추가
# window는 적당히 조절
# =========================
window = 30
df['FFT_MA'] = df['FFT_Peak_MHz'].rolling(window=window, center=True).mean()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap

# =========================
# 5. 전체 시간 흐름 plot
# 시간 흐름 배경 그라데이션
# =========================

fig, ax = plt.subplots(figsize=(16, 6))

# -------------------------
# 1) 사용자 지정 색상
# -------------------------
morning_color = '#FFF19B'   # 아침
noon_color    = '#FFF19B'   # 점심
evening_color = '#3D45AA'   # 저녁
dawn_color    = '#3D45AA'   # 새벽

dot_color  = '#285A48'      # 도트
line_color = '#000000'      # 선

# -------------------------
# 2) 시간축 전체 범위
# -------------------------
xmin = df['ModifiedDate'].min()
xmax = df['ModifiedDate'].max()

ymin = df['FFT_Peak_MHz'].min()
ymax = df['FFT_Peak_MHz'].max()

# 여백 조금
ypad = 0.03 * (ymax - ymin) if ymax > ymin else 0.1
ymin_plot = ymin - ypad
ymax_plot = ymax + ypad

# -------------------------
# 3) 하루 24시간용 그라데이션 colormap
#    0시(새벽) -> 6시(아침) -> 12시(점심) -> 18시(저녁) -> 24시(새벽)
# -------------------------
cmap = LinearSegmentedColormap.from_list(
    'day_cycle',
    [
        (0.00, dawn_color),      # 00:00
        (0.25, morning_color),   # 06:00
        (0.50, noon_color),      # 12:00
        (0.75, evening_color),   # 18:00
        (1.00, dawn_color)       # 24:00
    ]
)

# -------------------------
# 4) 날짜별로 배경 그라데이션 깔기
# -------------------------
start_day = df['ModifiedDate'].min().normalize()
end_day   = df['ModifiedDate'].max().normalize() + pd.Timedelta(days=1)

current_day = start_day
while current_day < end_day:
    next_day = current_day + pd.Timedelta(days=1)

    # x축 숫자값으로 변환
    x0 = mdates.date2num(current_day)
    x1 = mdates.date2num(next_day)

    # 가로 방향 gradient 이미지 생성
    grad = np.linspace(0, 1, 600).reshape(1, -1)

    ax.imshow(
        grad,
        extent=[x0, x1, ymin_plot, ymax_plot],
        aspect='auto',
        cmap=cmap,
        alpha=0.22
    )

    current_day = next_day

# -------------------------
# 5) 데이터 plot
# -------------------------
ax.plot(
    df['ModifiedDate'],
    df['FFT_Peak_MHz'],
    linestyle='None',
    marker='o',
    markersize=3,
    color=dot_color,
    label='FFT Peak',
    zorder=3
)

ax.plot(
    df['ModifiedDate'],
    df['FFT_MA'],
    color=line_color,
    linewidth=2,
    label=f'Moving Average ({window})',
    zorder=4
)

# -------------------------
# 6) 축/스타일
# -------------------------
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin_plot, ymax_plot)

ax.set_xlabel('Time')
ax.set_ylabel('FFT Peak (MHz)')
ax.set_title('FFT Peak flow over time')

ax.grid(True, alpha=0.2)
ax.legend()

ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# =========================
# 6. 시간대(hour)별 평균
# 밤/낮 경향 확인용
# =========================
df['Hour'] = df['ModifiedDate'].dt.hour

hourly = df.groupby('Hour')['FFT_Peak_MHz'].agg(['mean', 'std', 'count']).reset_index()

plt.figure(figsize=(10, 5))
plt.errorbar(hourly['Hour'], hourly['mean'], yerr=hourly['std'], fmt='o-', capsize=4)
plt.xlabel('Hour of day')
plt.ylabel('Mean FFT Peak (MHz)')
plt.title('Hourly mean FFT Peak')
plt.grid(True, alpha=0.3)
plt.xticks(range(24))
plt.tight_layout()
plt.show()

print(hourly)

# =========================
# 7. 날짜별 평균
# =========================
df['DateOnly'] = df['ModifiedDate'].dt.date
daily = df.groupby('DateOnly')['FFT_Peak_MHz'].agg(['mean', 'std', 'count']).reset_index()

plt.figure(figsize=(12, 5))
plt.plot(daily['DateOnly'], daily['mean'], 'o-')
plt.xlabel('Date')
plt.ylabel('Mean FFT Peak (MHz)')
plt.title('Daily mean FFT Peak')
plt.grid(True, alpha=0.3)
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

print(daily)

# =========================
# 8. 날짜-시간 히트맵
# 어느 시간대에 작아지고 커지는지 보기 좋음
# =========================
df['DateStr'] = df['ModifiedDate'].dt.strftime('%Y-%m-%d')

pivot = df.pivot_table(
    index='Hour',
    columns='DateStr',
    values='FFT_Peak_MHz',
    aggfunc='mean'
)

plt.figure(figsize=(12, 8))
plt.imshow(pivot, aspect='auto', origin='lower')
plt.colorbar(label='FFT Peak (MHz)')
plt.yticks(range(len(pivot.index)), pivot.index)
plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=45)
plt.xlabel('Date')
plt.ylabel('Hour')
plt.title('FFT Peak heatmap by date and hour')
plt.tight_layout()
plt.show()

# =========================
# 9. 낮/밤 비교 예시
# 낮: 9~18시 / 밤: 21~06시
# =========================
day_data = df[(df['Hour'] >= 9) & (df['Hour'] < 18)]['FFT_Peak_MHz']
night_data = df[(df['Hour'] >= 21) | (df['Hour'] < 6)]['FFT_Peak_MHz']

print('낮 평균 FFT Peak:', day_data.mean())
print('밤 평균 FFT Peak:', night_data.mean())
print('낮 표준편차:', day_data.std())
print('밤 표준편차:', night_data.std())
print('낮 샘플 수:', len(day_data))
print('밤 샘플 수:', len(night_data))
