import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.facecolor'] = 'white'

# 데이터 생성
data = {
    'threshold_miles': [50, 50, 50, 100, 100, 100, 200, 200, 200, 300, 300, 300, 400, 400, 400],
    'p': [5, 8, 10, 5, 8, 10, 5, 8, 10, 5, 8, 10, 5, 8, 10],
    'covered_states': [51]*15,
    'cover_rate': [1.0]*15,
    'objective': [9.93937]*15,
    'chosen_hubs': [
        [1, 16, 19, 29, 31],
        [1, 13, 16, 19, 27, 29, 31, 41],
        [1, 13, 16, 18, 19, 26, 27, 29, 31, 41],
        [17, 40, 46, 48, 5],
        [1, 17, 29, 31, 40, 46, 48, 5],
        [1, 17, 18, 29, 31, 40, 46, 48, 5, 55],
        [19, 29, 5, 55, 56],
        [1, 10, 11, 12, 19, 29, 5, 55],
        [1, 10, 11, 12, 13, 15, 19, 29, 5, 55],
        [1, 10, 11, 5, 55],
        [1, 10, 11, 12, 13, 15, 5, 55],
        [1, 10, 11, 12, 13, 15, 16, 17, 5, 55],
        [1, 10, 11, 12, 5],
        [1, 10, 11, 12, 13, 15, 16, 5],
        [1, 10, 11, 12, 13, 15, 16, 17, 18, 5]
    ]
}

df = pd.DataFrame(data)

# 미국 주 번호 -> 이름 매핑
state_names = {
    1: 'AL', 5: 'CA', 10: 'DE', 11: 'DC', 12: 'FL', 13: 'GA',
    15: 'HI', 16: 'ID', 17: 'IL', 18: 'IN', 19: 'IA', 26: 'MI',
    27: 'MN', 29: 'MO', 31: 'NE', 40: 'OK', 41: 'OR', 46: 'SD',
    48: 'TX', 55: 'WI', 56: 'WY'
}

# 큰 그림 생성
fig = plt.figure(figsize=(20, 12))

# 1. 허브 빈도 분석
print("=== Hub Frequency Analysis ===")
all_hubs = []
for hubs in df['chosen_hubs']:
    all_hubs.extend(hubs)

hub_freq = Counter(all_hubs)
hub_freq_sorted = sorted(hub_freq.items(), key=lambda x: x[1], reverse=True)

print("\nTop 10 Most Selected Hubs:")
for state_num, count in hub_freq_sorted[:10]:
    state_name = state_names.get(state_num, f'State {state_num}')
    print(f"  {state_name} (#{state_num}): {count}/15 scenarios ({count/15*100:.1f}%)")

# 2. 시각화 1: 허브 선택 빈도
ax1 = plt.subplot(2, 3, 1)
states = [state_names.get(s, f'{s}') for s, _ in hub_freq_sorted[:15]]
counts = [c for _, c in hub_freq_sorted[:15]]
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(states)))

bars = ax1.barh(states, counts, color=colors)
ax1.set_xlabel('Selection Frequency', fontsize=12, fontweight='bold')
ax1.set_title('Top 15 Most Selected Hub States\n(across all scenarios)', 
              fontsize=13, fontweight='bold', pad=15)
ax1.set_xlim(0, 16)
ax1.grid(axis='x', alpha=0.3, linestyle='--')

for i, (bar, count) in enumerate(zip(bars, counts)):
    ax1.text(count + 0.3, bar.get_y() + bar.get_height()/2, 
             f'{count}', va='center', fontweight='bold')

# 3. 시각화 2: 거리별 허브 수
ax2 = plt.subplot(2, 3, 2)
pivot = df.pivot(index='threshold_miles', columns='p', values='p')
pivot.index = [f'{int(x)}mi' for x in pivot.index]

x = np.arange(len(pivot.index))
width = 0.25
p_values = [5, 8, 10]
colors_p = ['#FF6B6B', '#4ECDC4', '#45B7D1']

for i, p in enumerate(p_values):
    counts = [len(df[(df['threshold_miles']==int(t[:-2])) & (df['p']==p)]['chosen_hubs'].values[0]) 
              for t in pivot.index]
    ax2.bar(x + i*width, counts, width, label=f'p={p}', color=colors_p[i], alpha=0.8)

ax2.set_xlabel('Threshold Distance', fontsize=12, fontweight='bold')
ax2.set_ylabel('Number of Hubs', fontsize=12, fontweight='bold')
ax2.set_title('Hub Count by Distance Threshold\nand Number of Facilities', 
              fontsize=13, fontweight='bold', pad=15)
ax2.set_xticks(x + width)
ax2.set_xticklabels(pivot.index)
ax2.legend(title='Facilities', fontsize=10)
ax2.grid(axis='y', alpha=0.3, linestyle='--')

# 4. 시각화 3: 거리별 허브 패턴
ax3 = plt.subplot(2, 3, 3)

threshold_groups = df.groupby('threshold_miles')
for threshold in [50, 100, 200, 300, 400]:
    group = df[df['threshold_miles'] == threshold]
    hub_set = set()
    for hubs in group['chosen_hubs']:
        hub_set.update(hubs)
    ax3.scatter([threshold]*len(hub_set), list(hub_set), 
                s=100, alpha=0.6, label=f'{threshold}mi')

ax3.set_xlabel('Threshold Distance (miles)', fontsize=12, fontweight='bold')
ax3.set_ylabel('State FIPS Code', fontsize=12, fontweight='bold')
ax3.set_title('Hub Distribution by Distance Threshold', 
              fontsize=13, fontweight='bold', pad=15)
ax3.grid(True, alpha=0.3, linestyle='--')
ax3.legend(title='Distance', ncol=5, loc='upper center', bbox_to_anchor=(0.5, -0.1))

# 5. 시각화 4: 히트맵 - 주별 시나리오별 선택
ax4 = plt.subplot(2, 3, 4)

# 히트맵 데이터 생성
unique_states = sorted(set(all_hubs))
scenario_labels = [f'{t}mi-p{p}' for t, p in zip(df['threshold_miles'], df['p'])]
heatmap_data = np.zeros((len(unique_states), len(scenario_labels)))

for j, (scenario, hubs) in enumerate(zip(scenario_labels, df['chosen_hubs'])):
    for i, state in enumerate(unique_states):
        if state in hubs:
            heatmap_data[i, j] = 1

im = ax4.imshow(heatmap_data, aspect='auto', cmap='RdYlGn', interpolation='nearest')
ax4.set_xticks(range(len(scenario_labels)))
ax4.set_xticklabels(scenario_labels, rotation=45, ha='right', fontsize=8)
ax4.set_yticks(range(len(unique_states)))
ax4.set_yticklabels([state_names.get(s, f'{s}') for s in unique_states], fontsize=9)
ax4.set_title('Hub Selection Heatmap\n(Green = Selected)', 
              fontsize=13, fontweight='bold', pad=15)
ax4.set_xlabel('Scenario', fontsize=11, fontweight='bold')
ax4.set_ylabel('State', fontsize=11, fontweight='bold')

# 6. 시각화 5: 거리별 핵심 허브
ax5 = plt.subplot(2, 3, 5)

distance_core_hubs = {}
for threshold in [50, 100, 200, 300, 400]:
    group = df[df['threshold_miles'] == threshold]
    hub_counter = Counter()
    for hubs in group['chosen_hubs']:
        hub_counter.update(hubs)
    distance_core_hubs[threshold] = hub_counter

threshold_list = [50, 100, 200, 300, 400]
top_states = list(set([s for _, counter in distance_core_hubs.items() 
                       for s, _ in counter.most_common(5)]))[:10]

heatmap_core = np.zeros((len(top_states), len(threshold_list)))
for i, state in enumerate(top_states):
    for j, threshold in enumerate(threshold_list):
        heatmap_core[i, j] = distance_core_hubs[threshold].get(state, 0)

im2 = ax5.imshow(heatmap_core, aspect='auto', cmap='YlOrRd', interpolation='nearest')
ax5.set_xticks(range(len(threshold_list)))
ax5.set_xticklabels([f'{t}mi' for t in threshold_list])
ax5.set_yticks(range(len(top_states)))
ax5.set_yticklabels([state_names.get(s, f'{s}') for s in top_states], fontsize=10)
ax5.set_title('Core Hub Frequency by Distance\n(Top 10 States)', 
              fontsize=13, fontweight='bold', pad=15)
ax5.set_xlabel('Threshold Distance', fontsize=11, fontweight='bold')
ax5.set_ylabel('State', fontsize=11, fontweight='bold')

# 컬러바
cbar2 = plt.colorbar(im2, ax=ax5)
cbar2.set_label('Selection Count', rotation=270, labelpad=20, fontweight='bold')

# 7. 시각화 6: 허브 수에 따른 패턴
ax6 = plt.subplot(2, 3, 6)

p_groups = df.groupby('p')
categories = ['50mi', '100mi', '200mi', '300mi', '400mi']

for p in [5, 8, 10]:
    group = df[df['p'] == p]
    unique_hubs = [len(set([h for hubs in group[group['threshold_miles']==t]['chosen_hubs'].values 
                            for h in hubs])) 
                   for t in [50, 100, 200, 300, 400]]
    ax6.plot(categories, unique_hubs, marker='o', linewidth=2.5, 
             markersize=10, label=f'p={p}', alpha=0.8)

ax6.set_xlabel('Threshold Distance', fontsize=12, fontweight='bold')
ax6.set_ylabel('Unique Hub States Used', fontsize=12, fontweight='bold')
ax6.set_title('Hub Diversity Across Distance Thresholds', 
              fontsize=13, fontweight='bold', pad=15)
ax6.legend(title='Facilities', fontsize=11)
ax6.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout(pad=3.0)
plt.savefig('/home/claude/mclp_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
print("\n==> Visualization saved: mclp_analysis.png")

# 추가 분석: 전략적 인사이트
print("\n=== Strategic Insights ===")
print("\n1. Core Strategic Hubs (selected in 80%+ scenarios):")
for state_num, count in hub_freq_sorted:
    if count >= 12:  # 80% of 15
        state_name = state_names.get(state_num, f'State {state_num}')
        print(f"   - {state_name}: {count}/15 ({count/15*100:.0f}%)")

print("\n2. Distance-Specific Patterns:")
for threshold in [50, 100, 200, 300, 400]:
    group = df[df['threshold_miles'] == threshold]
    all_hubs_dist = []
    for hubs in group['chosen_hubs']:
        all_hubs_dist.extend(hubs)
    unique_count = len(set(all_hubs_dist))
    most_common = Counter(all_hubs_dist).most_common(3)
    print(f"\n   {threshold}mi threshold:")
    print(f"   - Unique hubs used: {unique_count}")
    print(f"   - Most frequent: {', '.join([state_names.get(s, str(s)) for s, _ in most_common])}")

plt.close()
