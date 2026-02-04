"""
MCLP (Maximum Covering Location Problem) 물류 허브 최적화 분석
미국 주(state) 단위 물류 네트워크 허브 입지 선정
"""

import pandas as pd
import numpy as np
from collections import Counter
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# 1. 데이터 전처리 함수들
# ============================================================================

def add_post_tmiles(df):
    """
    COVID-19 이후(2022-2024) 평균 ton-miles 계산
    """
    df_copy = df.copy()
    post_cols = ['tmiles_2022', 'tmiles_2023', 'tmiles_2024']
    df_copy['post_tmiles'] = df_copy[post_cols].mean(axis=1)
    return df_copy


def build_state_year(df, mode="domestic"):
    """
    주-연도별 집계 데이터 생성
    
    Parameters:
    -----------
    df : DataFrame
        원본 물류 데이터
    mode : str
        "domestic" 또는 "international"
    
    Returns:
    --------
    DataFrame : 주-연도별 집계 데이터
    """
    # trade_type 필터링 (1=domestic, 2,3=international)
    if mode == "domestic":
        df_filtered = df[df['trade_type'] == 1].copy()
    else:
        df_filtered = df[df['trade_type'].isin([2, 3])].copy()
    
    # 연도별 집계
    years = range(2018, 2025)
    result_list = []
    
    for year in years:
        value_col = f'current_value_{year}'
        tmiles_col = f'tmiles_{year}'
        
        # dms_orig 기준 집계
        state_agg = df_filtered.groupby('dms_orig').agg({
            value_col: 'sum',
            tmiles_col: 'sum'
        }).reset_index()
        
        state_agg.columns = ['state', 'value', 'tmiles']
        state_agg['year'] = year
        result_list.append(state_agg)
    
    result_df = pd.concat(result_list, ignore_index=True)
    result_df = result_df[['state', 'year', 'value', 'tmiles']]
    
    return result_df


def build_pre_post_comp(state_year_df):
    """
    COVID-19 이전(2018-2019)과 이후(2022-2024) 비교
    
    Returns:
    --------
    DataFrame : 주별 pre/post 비교 데이터
    """
    # Pre: 2018-2019
    pre = state_year_df[state_year_df['year'].isin([2018, 2019])].copy()
    pre_agg = pre.groupby('state').agg({
        'value': 'mean',
        'tmiles': 'mean'
    }).reset_index()
    pre_agg.columns = ['state', 'pre_value', 'pre_tmiles']
    
    # Post: 2022-2024
    post = state_year_df[state_year_df['year'].isin([2022, 2023, 2024])].copy()
    post_agg = post.groupby('state').agg({
        'value': 'mean',
        'tmiles': 'mean'
    }).reset_index()
    post_agg.columns = ['state', 'post_value', 'post_tmiles']
    
    # 병합
    comp = pd.merge(pre_agg, post_agg, on='state', how='outer')
    
    # 변화율 계산
    comp['value_change_pct'] = ((comp['post_value'] - comp['pre_value']) / 
                                 comp['pre_value'] * 100)
    comp['tmiles_change_pct'] = ((comp['post_tmiles'] - comp['pre_tmiles']) / 
                                  comp['pre_tmiles'] * 100)
    
    return comp


def add_resilience_score(comp_df, alpha=0.5):
    """
    복원력(resilience) 점수 계산
    
    Parameters:
    -----------
    comp_df : DataFrame
        pre/post 비교 데이터
    alpha : float
        가치와 톤마일의 가중치 (0~1)
    
    Returns:
    --------
    DataFrame : 복원력 점수가 추가된 데이터
    """
    df = comp_df.copy()
    
    # 결측치 처리
    df['value_change_pct'].fillna(0, inplace=True)
    df['tmiles_change_pct'].fillna(0, inplace=True)
    
    # 정규화 (0-1 스케일)
    val_min = df['value_change_pct'].min()
    val_max = df['value_change_pct'].max()
    tm_min = df['tmiles_change_pct'].min()
    tm_max = df['tmiles_change_pct'].max()
    
    df['value_norm'] = (df['value_change_pct'] - val_min) / (val_max - val_min + 1e-10)
    df['tmiles_norm'] = (df['tmiles_change_pct'] - tm_min) / (tm_max - tm_min + 1e-10)
    
    # 복원력 점수 (가중 평균)
    df['resilience_score'] = alpha * df['value_norm'] + (1 - alpha) * df['tmiles_norm']
    
    # 순위 매기기 (높을수록 좋음)
    df = df.sort_values('resilience_score', ascending=False).reset_index(drop=True)
    df['rank'] = range(1, len(df) + 1)
    
    return df


# ============================================================================
# 2. OD 데이터 생성
# ============================================================================

def build_od_df(df, use_post_tmiles=True):
    """
    Origin-Destination 매트릭스 생성
    
    Parameters:
    -----------
    df : DataFrame
        원본 데이터 (post_tmiles 컬럼 필요)
    use_post_tmiles : bool
        post_tmiles 사용 여부
    
    Returns:
    --------
    DataFrame : OD 쌍별 집계 데이터
    """
    # domestic만 필터링
    df_dom = df[df['trade_type'] == 1].copy()
    
    # 집계
    if use_post_tmiles:
        od = df_dom.groupby(['dms_orig', 'dms_dest']).agg({
            'post_tmiles': 'sum'
        }).reset_index()
        od.columns = ['i_state', 'j_state', 'demand']
    else:
        od = df_dom.groupby(['dms_orig', 'dms_dest']).agg({
            'tmiles_2024': 'sum'
        }).reset_index()
        od.columns = ['i_state', 'j_state', 'demand']
    
    # 문자열로 변환 (매칭 용이)
    od['i_state'] = od['i_state'].astype(str)
    od['j_state'] = od['j_state'].astype(str)
    
    return od


def compute_distance_matrix(states, seed=42):
    """
    주 간 거리 매트릭스 생성 (시뮬레이션용)
    실제로는 실제 거리 데이터 사용 필요
    
    Parameters:
    -----------
    states : list
        주 리스트
    seed : int
        랜덤 시드
    
    Returns:
    --------
    dict : {(i, j): distance}
    """
    np.random.seed(seed)
    n = len(states)
    distances = {}
    
    for i in range(n):
        for j in range(n):
            if i == j:
                distances[(states[i], states[j])] = 0
            elif i < j:
                # 50~2000 마일 범위
                dist = np.random.uniform(50, 2000)
                distances[(states[i], states[j])] = dist
                distances[(states[j], states[i])] = dist
            # else: 이미 추가됨 (대칭)
    
    return distances


# ============================================================================
# 3. MCLP 최적화
# ============================================================================

def solve_mclp_greedy(od_df, rank_df, distance_dict, threshold, p, alpha=0.8):
    """
    Greedy 알고리즘으로 MCLP 문제 해결
    
    Parameters:
    -----------
    od_df : DataFrame
        OD 수요 데이터
    rank_df : DataFrame
        주별 순위 데이터 (resilience_score 포함)
    distance_dict : dict
        거리 매트릭스
    threshold : float
        커버리지 임계거리 (마일)
    p : int
        선택할 허브 수
    alpha : float
        수요와 복원력의 가중치
    
    Returns:
    --------
    tuple : (선택된 허브 리스트, 목적함수 값, 커버된 주 set)
    """
    # 후보 주 리스트
    candidates = sorted(rank_df['state'].astype(str).unique())
    
    # 초기화
    selected_hubs = []
    covered_states = set()
    
    # 주별 복원력 점수 매핑
    resilience_map = dict(zip(rank_df['state'].astype(str), 
                              rank_df['resilience_score']))
    
    # Greedy 선택
    for _ in range(p):
        best_hub = None
        best_value = -np.inf
        best_new_covered = set()
        
        for candidate in candidates:
            if candidate in selected_hubs:
                continue
            
            # 이 허브를 추가했을 때 새로 커버되는 주들
            new_covered = set()
            for state in candidates:
                # 이미 커버된 주는 제외
                if state in covered_states:
                    continue
                
                # 거리 체크
                key = (candidate, state)
                if key in distance_dict and distance_dict[key] <= threshold:
                    new_covered.add(state)
            
            # 목적함수 계산
            # = alpha * (새로 커버되는 수요) + (1-alpha) * (새로 커버되는 주의 복원력 합)
            demand_value = 0
            resilience_value = 0
            
            for state in new_covered:
                # 해당 주에서 발생하는 수요
                state_demand = od_df[od_df['i_state'] == state]['demand'].sum()
                demand_value += state_demand
                
                # 복원력 점수
                resilience_value += resilience_map.get(state, 0)
            
            # 정규화를 위해 개수로 나눔
            if len(new_covered) > 0:
                total_value = alpha * demand_value + (1 - alpha) * resilience_value
            else:
                total_value = 0
            
            # 최선의 선택 업데이트
            if total_value > best_value:
                best_value = total_value
                best_hub = candidate
                best_new_covered = new_covered
        
        # 선택
        if best_hub is not None:
            selected_hubs.append(best_hub)
            covered_states.update(best_new_covered)
    
    # 최종 목적함수 값
    total_demand = 0
    total_resilience = 0
    
    for state in covered_states:
        state_demand = od_df[od_df['i_state'] == state]['demand'].sum()
        total_demand += state_demand
        total_resilience += resilience_map.get(state, 0)
    
    objective = alpha * total_demand + (1 - alpha) * total_resilience
    
    # 정규화 (로그 스케일)
    objective = np.log10(objective + 1)
    
    return selected_hubs, objective, covered_states


def run_mclp_scenarios(df, rank_df, thresholds=(50, 100, 200, 300, 400), 
                       ps=(5, 8, 10), alpha=0.8):
    """
    여러 시나리오에 대해 MCLP 실행
    
    Parameters:
    -----------
    df : DataFrame
        원본 데이터
    rank_df : DataFrame
        순위 데이터
    thresholds : tuple
        임계거리 리스트
    ps : tuple
        허브 개수 리스트
    alpha : float
        가중치
    
    Returns:
    --------
    DataFrame : 시나리오별 결과
    """
    # 데이터 준비
    df_with_post = add_post_tmiles(df)
    od_df = build_od_df(df_with_post)
    
    # 거리 매트릭스 생성
    all_states = sorted(set(od_df['i_state'].unique()) | set(od_df['j_state'].unique()))
    distance_dict = compute_distance_matrix(all_states)
    
    # 시나리오 실행
    results = []
    
    for threshold in thresholds:
        for p in ps:
            print(f"Running: threshold={threshold}mi, p={p}")
            
            hubs, objective, covered = solve_mclp_greedy(
                od_df, rank_df, distance_dict, threshold, p, alpha
            )
            
            results.append({
                'threshold_miles': threshold,
                'p': p,
                'covered_states': len(covered),
                'cover_rate': len(covered) / len(all_states),
                'objective': objective,
                'chosen_hubs': [int(h) for h in hubs]
            })
    
    return pd.DataFrame(results)


# ============================================================================
# 4. 메인 실행 함수
# ============================================================================

def run_full_analysis(csv_path, output_path='mclp_results.csv'):
    """
    전체 분석 파이프라인 실행
    
    Parameters:
    -----------
    csv_path : str
        원본 CSV 파일 경로
    output_path : str
        결과 저장 경로
    
    Returns:
    --------
    tuple : (결과 DataFrame, 순위 DataFrame)
    """
    print("=" * 80)
    print("MCLP 물류 허브 최적화 분석 시작")
    print("=" * 80)
    
    # 1. 데이터 로딩
    print("\n[1/5] 데이터 로딩...")
    df = pd.read_csv(csv_path)
    print(f"   - 데이터 shape: {df.shape}")
    print(f"   - 컬럼: {list(df.columns)[:10]}...")
    
    # 2. 전처리
    print("\n[2/5] 데이터 전처리...")
    dom_sy = build_state_year(df, mode="domestic")
    print(f"   - 주-연도 데이터: {dom_sy.shape}")
    
    # 3. 비교 및 복원력 점수
    print("\n[3/5] Pre/Post 비교 및 복원력 점수 계산...")
    comp_dom = build_pre_post_comp(dom_sy)
    rank_dom = add_resilience_score(comp_dom, alpha=0.8)
    print(f"   - 주별 순위 데이터: {rank_dom.shape}")
    print(f"   - Top 5 states by resilience:")
    for idx, row in rank_dom.head().iterrows():
        print(f"      {int(row['rank'])}. State {int(row['state'])}: "
              f"score={row['resilience_score']:.4f}")
    
    # 4. MCLP 실행
    print("\n[4/5] MCLP 시나리오 실행...")
    res_dom = run_mclp_scenarios(
        df=df,
        rank_df=rank_dom,
        thresholds=(50, 100, 200, 300, 400),
        ps=(5, 8, 10),
        alpha=0.8
    )
    
    # 5. 결과 저장
    print("\n[5/5] 결과 저장...")
    res_dom.to_csv(output_path, index=False)
    print(f"   - 저장 완료: {output_path}")
    
    print("\n" + "=" * 80)
    print("분석 완료!")
    print("=" * 80)
    print(f"\n결과 요약:")
    print(f"   - 총 시나리오: {len(res_dom)}")
    print(f"   - 평균 커버율: {res_dom['cover_rate'].mean():.2%}")
    print(f"   - 평균 목적함수: {res_dom['objective'].mean():.4f}")
    
    return res_dom, rank_dom


# ============================================================================
# 5. 결과 분석 함수
# ============================================================================

def analyze_results(results_df):
    """
    결과 분석 및 인사이트 추출
    """
    print("\n" + "=" * 80)
    print("결과 분석")
    print("=" * 80)
    
    # 1. 허브 빈도 분석
    print("\n[1] 허브 선택 빈도 분석")
    all_hubs = []
    for hubs in results_df['chosen_hubs']:
        all_hubs.extend(hubs)
    
    hub_freq = Counter(all_hubs)
    print("\n가장 많이 선택된 허브 Top 10:")
    for i, (state, count) in enumerate(hub_freq.most_common(10), 1):
        pct = count / len(results_df) * 100
        print(f"   {i:2d}. State {state:3d}: {count:2d}/15 ({pct:5.1f}%)")
    
    # 2. 거리별 패턴
    print("\n[2] 거리 임계값별 패턴")
    for threshold in results_df['threshold_miles'].unique():
        subset = results_df[results_df['threshold_miles'] == threshold]
        unique_hubs = set()
        for hubs in subset['chosen_hubs']:
            unique_hubs.update(hubs)
        print(f"   {threshold:3d}mi: {len(unique_hubs):2d}개 고유 허브 사용")
    
    # 3. p값별 분석
    print("\n[3] 시설 수(p)별 분석")
    for p in results_df['p'].unique():
        subset = results_df[results_df['p'] == p]
        avg_coverage = subset['cover_rate'].mean()
        print(f"   p={p:2d}: 평균 커버율 {avg_coverage:.2%}")
    
    # 4. 핵심 허브 식별
    print("\n[4] 전략적 핵심 허브 (80% 이상 선택)")
    core_hubs = [state for state, count in hub_freq.items() 
                 if count >= len(results_df) * 0.8]
    print(f"   핵심 허브: {core_hubs}")
    
    return hub_freq


# ============================================================================
# 메인 실행
# ============================================================================

if __name__ == "__main__":
    # 예시 실행
    # CSV 파일 경로 지정 필요
    csv_path = '../data/logistics.csv'
    
    # 전체 분석 실행
    results, rankings = run_full_analysis(csv_path, output_path='mclp_results.csv')
    
    # 결과 분석
    hub_frequency = analyze_results(results)
    
    # 결과 표시
    print("\n최종 결과 (처음 5개 시나리오):")
    print(results.head().to_string())
