# utils.py

__all__ = [
    'filter_year','remove_invalid_values','calc_value_density','calc_hub_index', 'top_value_density',
'top_value_density', 'top_total_value', 'top_total_tons','top_hub_states','plot_value_per_ton_top10',
'plot_value_vs_tons_scatter','plot_value_per_ton_trend', 'plot_value_ton_comparison'
   
]

# ============================
# 1. 공통 전처리
# ============================

def filter_year(df: pd.DataFrame, year: int):
    """연도 필터"""
    return df[df["year"] == year].copy()


def remove_invalid_values(df: pd.DataFrame, col: str):
    """NaN, inf 제거"""
    return df[df[col].notna() & ~df[col].isin([np.inf, -np.inf])]


# ============================
# 2. 집계 & 파생변수
# ============================

def calc_value_density(df: pd.DataFrame):
    """
    state, year 단위
    value / tons 계산
    """
    agg = (
        df.groupby(["state", "year"])[["value", "tons"]]
        .sum()
        .reset_index()
    )

    agg["value_per_ton"] = agg["value"] / agg["tons"]
    return agg


def calc_hub_index(df: pd.DataFrame):
    """
    Outbound + Inbound Tons
    """
    outb = (
        df.groupby(["state_orig_nm", "year"])["tons"]
        .sum()
        .reset_index()
        .rename(columns={"state_orig_nm": "state",
                         "tons": "outbound_tons"})
    )

    inb = (
        df.groupby(["state_dest_nm", "year"])["tons"]
        .sum()
        .reset_index()
        .rename(columns={"state_dest_nm": "state",
                         "tons": "inbound_tons"})
    )

    hub = pd.merge(outb, inb, on=["state", "year"], how="outer").fillna(0)
    hub["hub_index"] = hub["outbound_tons"] + hub["inbound_tons"]

    return hub


# ============================
# 3. 분석 함수
# ============================

def top_value_density(value_state_long, year=2024, top_n=10):
    df = filter_year(value_state_long, year)
    df = remove_invalid_values(df, "value_per_ton")

    return (
        df.sort_values("value_per_ton", ascending=False)
          .head(top_n)
          .reset_index(drop=True)
    )


def top_total_value(value_state_long, year=2024, top_n=10):
    df = filter_year(value_state_long, year)

    return (
        df.sort_values("value", ascending=False)
          .head(top_n)
          .reset_index(drop=True)
    )

def top_total_tons(value_state_long, year=2024, top_n=10):
    df = filter_year(value_state_long, year)

    return (
        df.sort_values("tons", ascending=False)
          .head(top_n)
          .reset_index(drop=True)
    )



def top_hub_states(hub_long, year=2024, top_n=10):
    df = filter_year(hub_long, year)

    return (
        df.sort_values("hub_index", ascending=False)
          .head(top_n)
          .reset_index(drop=True)
    )


def plot_value_per_ton_top10(value_state_long, year=2024):
    """
    Value per Ton Top 10 가로 막대 그래프
    """
    year_data = value_state_long[value_state_long['year'] == year].copy()
    
    # 결측치 제거
    year_data = year_data[year_data['value_per_ton'].notna()]
    year_data = year_data[~year_data['value_per_ton'].isin([np.inf, -np.inf])]
    
    top10 = year_data.nlargest(10, 'value_per_ton')
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 가로 막대 그래프 (역순으로 표시해서 1위가 위에 오도록)
    y_pos = np.arange(len(top10))
    ax.barh(y_pos, top10['value_per_ton'].values[::-1], alpha=0.8, color='steelblue')
    
    ax.set_xlabel('Value per Ton ($ thousands)', fontsize=12, fontweight='bold')
    ax.set_ylabel('State', fontsize=12, fontweight='bold')
    ax.set_title(f'{year} Top 10 States by Value Density', fontsize=14, fontweight='bold')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top10['state'].values[::-1])
    ax.grid(axis='x', alpha=0.3)
    
    # 값 표시
    for i, v in enumerate(top10['value_per_ton'].values[::-1]):
        ax.text(v + 0.02, i, f'{v:.2f}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/value_per_ton_top10.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n차트 저장 완료: value_per_ton_top10.png")


def plot_value_vs_tons_scatter(value_state_long, year=2024, top_n=20):
    """
    Value vs Tons 산점도 (Top N 주)
    """
    year_data = value_state_long[value_state_long['year'] == year].copy()
    
    # Top N by value
    top_states = year_data.nlargest(top_n, 'value')
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    scatter = ax.scatter(
        top_states['tons'], 
        top_states['value'],
        s=100,
        alpha=0.6,
        c=top_states['value_per_ton'],
        cmap='viridis'
    )
    
    # 주 이름 표시
    for idx, row in top_states.iterrows():
        ax.annotate(
            row['state'],
            (row['tons'], row['value']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=8
        )
    
    ax.set_xlabel('Total Tons (thousands)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Total Value ($ millions)', fontsize=12, fontweight='bold')
    ax.set_title(f'{year} Value vs Tons - Top {top_n} States', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    
    # 컬러바
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Value per Ton', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/value_vs_tons_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"차트 저장 완료: value_vs_tons_scatter.png")


def plot_value_per_ton_trend(value_state_long, top_n=10):
    """
    Top N 주의 Value per Ton 연도별 추이
    """
    # 2024년 기준 Top N 선정
    top_states = (
        value_state_long[value_state_long['year'] == 2024]
        .nlargest(top_n, 'value_per_ton')['state']
        .tolist()
    )
    
    # Top N 주 데이터만 필터링
    trend_data = value_state_long[value_state_long['state'].isin(top_states)].copy()
    
    # 결측치 제거
    trend_data = trend_data[trend_data['value_per_ton'].notna()]
    trend_data = trend_data[~trend_data['value_per_ton'].isin([np.inf, -np.inf])]
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    for state in top_states:
        state_data = trend_data[trend_data['state'] == state].sort_values('year')
        ax.plot(
            state_data['year'], 
            state_data['value_per_ton'],
            marker='o',
            label=state,
            linewidth=2
        )
    
    ax.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax.set_ylabel('Value per Ton ($ thousands)', fontsize=12, fontweight='bold')
    ax.set_title(f'Value per Ton Trend - Top {top_n} States (2024 basis)', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/value_per_ton_trend.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"차트 저장 완료: value_per_ton_trend.png")


def plot_value_ton_comparison(value_state_long, year=2024, top_n=15):
    """
    Value와 Tons Top N 비교 (Dual Bar Chart)
    """
    year_data = value_state_long[value_state_long['year'] == year].copy()
    
    # Value 기준 Top N
    top_states = year_data.nlargest(top_n, 'value')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Value Top N
    ax1.barh(range(len(top_states)), top_states['value'].values[::-1], alpha=0.8, color='steelblue')
    ax1.set_xlabel('Value ($ millions)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('State', fontsize=12, fontweight='bold')
    ax1.set_title(f'{year} Top {top_n} by Total Value', fontsize=13, fontweight='bold')
    ax1.set_yticks(range(len(top_states)))
    ax1.set_yticklabels(top_states['state'].values[::-1])
    ax1.grid(axis='x', alpha=0.3)
    
    # Tons Top N
    top_tons = year_data.nlargest(top_n, 'tons')
    ax2.barh(range(len(top_tons)), top_tons['tons'].values[::-1], alpha=0.8, color='coral')
    ax2.set_xlabel('Tons (thousands)', fontsize=12, fontweight='bold')
    ax2.set_title(f'{year} Top {top_n} by Total Tons', fontsize=13, fontweight='bold')
    ax2.set_yticks(range(len(top_tons)))
    ax2.set_yticklabels(top_tons['state'].values[::-1])
    ax2.grid(axis='x', alpha=0.3)
    
    #plt.tight_layout()
    #plt.savefig('/mnt/user-data/outputs/value_ton_comparison.png', dpi=300, bbox_inches='tight')
    #plt.close()
    
    print(f"차트 저장 완료: value_ton_comparison.png")