"""
MCLP 분석 사용 예제 및 유틸리티 함수들
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mclp_analysis_code import *


# ============================================================================
# 추가 유틸리티 함수들
# ============================================================================

def create_state_mapping():
    """
    미국 주 FIPS 코드 -> 주 이름 매핑
    """
    state_map = {
        1: 'Alabama', 2: 'Alaska', 4: 'Arizona', 5: 'Arkansas', 6: 'California',
        8: 'Colorado', 9: 'Connecticut', 10: 'Delaware', 11: 'District of Columbia',
        12: 'Florida', 13: 'Georgia', 15: 'Hawaii', 16: 'Idaho', 17: 'Illinois',
        18: 'Indiana', 19: 'Iowa', 20: 'Kansas', 21: 'Kentucky', 22: 'Louisiana',
        23: 'Maine', 24: 'Maryland', 25: 'Massachusetts', 26: 'Michigan',
        27: 'Minnesota', 28: 'Mississippi', 29: 'Missouri', 30: 'Montana',
        31: 'Nebraska', 32: 'Nevada', 33: 'New Hampshire', 34: 'New Jersey',
        35: 'New Mexico', 36: 'New York', 37: 'North Carolina', 38: 'North Dakota',
        39: 'Ohio', 40: 'Oklahoma', 41: 'Oregon', 42: 'Pennsylvania',
        44: 'Rhode Island', 45: 'South Carolina', 46: 'South Dakota',
        47: 'Tennessee', 48: 'Texas', 49: 'Utah', 50: 'Vermont', 51: 'Virginia',
        53: 'Washington', 54: 'West Virginia', 55: 'Wisconsin', 56: 'Wyoming'
    }
    
    # 약어 버전
    abbr_map = {
        1: 'AL', 2: 'AK', 4: 'AZ', 5: 'AR', 6: 'CA', 8: 'CO', 9: 'CT',
        10: 'DE', 11: 'DC', 12: 'FL', 13: 'GA', 15: 'HI', 16: 'ID',
        17: 'IL', 18: 'IN', 19: 'IA', 20: 'KS', 21: 'KY', 22: 'LA',
        23: 'ME', 24: 'MD', 25: 'MA', 26: 'MI', 27: 'MN', 28: 'MS',
        29: 'MO', 30: 'MT', 31: 'NE', 32: 'NV', 33: 'NH', 34: 'NJ',
        35: 'NM', 36: 'NY', 37: 'NC', 38: 'ND', 39: 'OH', 40: 'OK',
        41: 'OR', 42: 'PA', 44: 'RI', 45: 'SC', 46: 'SD', 47: 'TN',
        48: 'TX', 49: 'UT', 50: 'VT', 51: 'VA', 53: 'WA', 54: 'WV',
        55: 'WI', 56: 'WY'
    }
    
    return state_map, abbr_map


def print_scenario_details(results_df, scenario_idx):
    """
    특정 시나리오의 상세 정보 출력
    
    Parameters:
    -----------
    results_df : DataFrame
        MCLP 결과
    scenario_idx : int
        시나리오 인덱스
    """
    row = results_df.iloc[scenario_idx]
    _, abbr_map = create_state_mapping()
    
    print(f"\n{'=' * 60}")
    print(f"시나리오 #{scenario_idx + 1} 상세 정보")
    print(f"{'=' * 60}")
    print(f"임계거리:     {row['threshold_miles']} 마일")
    print(f"허브 개수:     {row['p']}개")
    print(f"커버된 주:     {row['covered_states']}개 ({row['cover_rate']:.1%})")
    print(f"목적함수 값:   {row['objective']:.6f}")
    
    print(f"\n선택된 허브:")
    for i, hub in enumerate(row['chosen_hubs'], 1):
        state_name = abbr_map.get(hub, f"State_{hub}")
        print(f"   {i}. {state_name} (#{hub})")
    print("=" * 60)


def compare_scenarios(results_df, idx1, idx2):
    """
    두 시나리오 비교
    """
    row1 = results_df.iloc[idx1]
    row2 = results_df.iloc[idx2]
    _, abbr_map = create_state_mapping()
    
    print(f"\n{'=' * 80}")
    print(f"시나리오 비교: #{idx1+1} vs #{idx2+1}")
    print(f"{'=' * 80}")
    
    print(f"\n{'항목':<20} {'시나리오 1':<25} {'시나리오 2':<25}")
    print("-" * 80)
    print(f"{'임계거리':<20} {row1['threshold_miles']}mi{'':<20} {row2['threshold_miles']}mi")
    print(f"{'허브 개수':<20} {row1['p']}개{'':<22} {row2['p']}개")
    print(f"{'커버율':<20} {row1['cover_rate']:.1%}{'':<19} {row2['cover_rate']:.1%}")
    print(f"{'목적함수':<20} {row1['objective']:.4f}{'':<16} {row2['objective']:.4f}")
    
    # 허브 비교
    hubs1 = set(row1['chosen_hubs'])
    hubs2 = set(row2['chosen_hubs'])
    
    common = hubs1 & hubs2
    only1 = hubs1 - hubs2
    only2 = hubs2 - hubs1
    
    print(f"\n공통 허브 ({len(common)}개):")
    for hub in sorted(common):
        print(f"   - {abbr_map.get(hub, hub)}", end="  ")
    
    if only1:
        print(f"\n\n시나리오 1에만 있는 허브 ({len(only1)}개):")
        for hub in sorted(only1):
            print(f"   - {abbr_map.get(hub, hub)}", end="  ")
    
    if only2:
        print(f"\n\n시나리오 2에만 있는 허브 ({len(only2)}개):")
        for hub in sorted(only2):
            print(f"   - {abbr_map.get(hub, hub)}", end="  ")
    
    print("\n" + "=" * 80)


def export_results_to_excel(results_df, rankings_df, filename='mclp_results.xlsx'):
    """
    결과를 Excel 파일로 저장 (여러 시트)
    """
    _, abbr_map = create_state_mapping()
    
    # 결과 데이터 가공
    results_export = results_df.copy()
    results_export['hub_names'] = results_export['chosen_hubs'].apply(
        lambda hubs: ', '.join([abbr_map.get(h, str(h)) for h in hubs])
    )
    
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        # Sheet 1: 시나리오 결과
        results_export.to_excel(writer, sheet_name='Scenarios', index=False)
        
        # Sheet 2: 주별 순위
        rankings_export = rankings_df.copy()
        rankings_export['state_name'] = rankings_export['state'].apply(
            lambda x: abbr_map.get(int(x), str(x))
        )
        rankings_export.to_excel(writer, sheet_name='State_Rankings', index=False)
        
        # Sheet 3: 허브 빈도 분석
        all_hubs = []
        for hubs in results_df['chosen_hubs']:
            all_hubs.extend(hubs)
        
        from collections import Counter
        hub_freq = Counter(all_hubs)
        freq_df = pd.DataFrame([
            {
                'state_code': state,
                'state_name': abbr_map.get(state, str(state)),
                'selection_count': count,
                'selection_rate': f"{count/len(results_df)*100:.1f}%"
            }
            for state, count in hub_freq.most_common()
        ])
        freq_df.to_excel(writer, sheet_name='Hub_Frequency', index=False)
    
    print(f"\n결과를 Excel 파일로 저장했습니다: {filename}")


def visualize_single_scenario(results_df, scenario_idx):
    """
    단일 시나리오 시각화
    """
    row = results_df.iloc[scenario_idx]
    _, abbr_map = create_state_mapping()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. 허브 위치 (막대 그래프)
    ax1 = axes[0]
    hubs = row['chosen_hubs']
    hub_names = [abbr_map.get(h, str(h)) for h in hubs]
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(hubs)))
    ax1.barh(hub_names, [1]*len(hubs), color=colors)
    ax1.set_xlabel('Selected as Hub', fontsize=12, fontweight='bold')
    ax1.set_title(f'Scenario: {row["threshold_miles"]}mi, p={row["p"]}', 
                  fontsize=13, fontweight='bold')
    ax1.set_xlim(0, 1.5)
    
    # 2. 시나리오 정보
    ax2 = axes[1]
    ax2.axis('off')
    
    info_text = f"""
    Scenario Information
    ━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    Threshold Distance:  {row['threshold_miles']} miles
    Number of Hubs:      {row['p']}
    
    Coverage:            {row['covered_states']}/51 states
    Coverage Rate:       {row['cover_rate']:.1%}
    
    Objective Value:     {row['objective']:.6f}
    
    Selected Hubs:
    {', '.join(hub_names)}
    """
    
    ax2.text(0.1, 0.5, info_text, fontsize=11, verticalalignment='center',
             family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(f'/home/claude/scenario_{scenario_idx+1}.png', dpi=150, bbox_inches='tight')
    print(f"\n시각화 저장: scenario_{scenario_idx+1}.png")
    plt.close()


def recommend_strategy(results_df, objective='balanced'):
    """
    목적에 따른 전략 추천
    
    Parameters:
    -----------
    results_df : DataFrame
        MCLP 결과
    objective : str
        'cost_efficient', 'balanced', 'high_service'
    """
    print(f"\n{'=' * 80}")
    print(f"전략 추천: {objective.upper()}")
    print(f"{'=' * 80}")
    
    if objective == 'cost_efficient':
        # 최소 비용: p=5, 가장 긴 거리
        recommended = results_df[(results_df['p'] == 5) & 
                                (results_df['threshold_miles'] >= 300)].iloc[0]
        print("\n목표: 최소 비용으로 운영")
        print("특징: 적은 수의 허브로 넓은 범위 커버")
        
    elif objective == 'balanced':
        # 균형: p=8, 중간 거리
        recommended = results_df[(results_df['p'] == 8) & 
                                (results_df['threshold_miles'] == 200)].iloc[0]
        print("\n목표: 비용과 서비스 품질의 균형")
        print("특징: 적절한 허브 수와 배송 거리")
        
    else:  # high_service
        # 고품질 서비스: p=10, 짧은 거리
        recommended = results_df[(results_df['p'] == 10) & 
                                (results_df['threshold_miles'] <= 100)].iloc[0]
        print("\n목표: 최고 수준의 배송 서비스")
        print("특징: 많은 허브로 빠른 배송 가능")
    
    print_scenario_details(results_df, recommended.name)
    
    return recommended


# ============================================================================
# 사용 예제
# ============================================================================

def example_usage():
    """
    전체 분석 파이프라인 사용 예제
    """
    print("\n" + "="*80)
    print("MCLP 분석 사용 예제")
    print("="*80)
    
    # 1. 데이터가 있는 경우
    print("\n[예제 1] 실제 데이터로 전체 분석 실행")
    print("-" * 80)
    print("""
    # CSV 파일에서 분석 실행
    results, rankings = run_full_analysis(
        csv_path='../data/logistics.csv',
        output_path='mclp_results.csv'
    )
    
    # 결과 분석
    hub_freq = analyze_results(results)
    """)
    
    # 2. 특정 시나리오 확인
    print("\n[예제 2] 특정 시나리오 상세 확인")
    print("-" * 80)
    print("""
    # 첫 번째 시나리오 상세 정보
    print_scenario_details(results, scenario_idx=0)
    
    # 시나리오 비교
    compare_scenarios(results, idx1=0, idx2=5)
    """)
    
    # 3. 시각화
    print("\n[예제 3] 결과 시각화")
    print("-" * 80)
    print("""
    # 단일 시나리오 시각화
    visualize_single_scenario(results, scenario_idx=0)
    """)
    
    # 4. Excel 저장
    print("\n[예제 4] Excel로 결과 저장")
    print("-" * 80)
    print("""
    # Excel 파일로 내보내기 (여러 시트)
    export_results_to_excel(results, rankings, 'mclp_results.xlsx')
    """)
    
    # 5. 전략 추천
    print("\n[예제 5] 목적에 맞는 전략 추천")
    print("-" * 80)
    print("""
    # 비용 효율적 전략
    strategy = recommend_strategy(results, objective='cost_efficient')
    
    # 균형잡힌 전략
    strategy = recommend_strategy(results, objective='balanced')
    
    # 고품질 서비스 전략
    strategy = recommend_strategy(results, objective='high_service')
    """)
    
    # 6. 커스텀 분석
    print("\n[예제 6] 커스텀 파라미터로 분석")
    print("-" * 80)
    print("""
    # 특정 거리와 허브 수만 테스트
    custom_results = run_mclp_scenarios(
        df=df,
        rank_df=rankings,
        thresholds=(100, 150, 250),  # 커스텀 거리
        ps=(6, 7, 9),                # 커스텀 허브 수
        alpha=0.7                    # 수요 가중치 조정
    )
    """)


# ============================================================================
# 메인 실행
# ============================================================================

if __name__ == "__main__":
    # 사용 예제 출력
    example_usage()
    
    print("\n" + "="*80)
    print("분석 코드 준비 완료!")
    print("="*80)
    print("\n실제 데이터로 분석을 실행하려면:")
    print("   python mclp_analysis_code.py")
    print("\n또는 Python 스크립트에서:")
    print("   from mclp_analysis_code import run_full_analysis")
    print("   results, rankings = run_full_analysis('your_data.csv')")
