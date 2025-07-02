# Xtructure 벤치마크 시스템

이 디렉토리는 Xtructure의 JAX 최적화 데이터 구조와 표준 Python 구현체 간의 포괄적인 성능 벤치마크를 제공합니다.

## 📁 프로젝트 구조

```
benchmarks/
├── __init__.py                 # 메인 패키지
├── README.md                   # 이 파일
├── run_all.py                  # 모든 벤치마크 실행기
├── common/                     # 공통 유틸리티
│   ├── __init__.py
│   ├── base_benchmark.py       # 기본 벤치마크 클래스
│   ├── hardware_info.py        # 하드웨어 정보 수집
│   └── test_data.py           # 공통 테스트 데이터
├── stack_benchmark.py          # Stack vs list 벤치마크
├── queue_benchmark.py          # Queue vs deque 벤치마크
├── hashtable_benchmark.py      # HashTable vs dict 벤치마크
└── bgpq_benchmark.py          # BGPQ vs heapq 벤치마크
```

## 🚀 빠른 시작

### 모든 벤치마크 실행

```bash
# 프로젝트 루트에서
python3 -m benchmarks
```

또는

```bash
cd benchmarks
python3 run_all.py
```

### 개별 데이터 구조 벤치마크 실행

```bash
# Stack 벤치마크만 실행
python3 -m benchmarks.stack_benchmark

# Queue 벤치마크만 실행  
python3 -m benchmarks.queue_benchmark

# HashTable 벤치마크만 실행
python3 -m benchmarks.hashtable_benchmark

# BGPQ 벤치마크만 실행 (GPU 권장)
python3 -m benchmarks.bgpq_benchmark
```

## 🔧 설정 옵션

### JAX 백엔드 설정

```python
from benchmarks import run_all_benchmarks

# CPU 백엔드 사용 (기본값)
runner = run_all_benchmarks(backend="cpu")

# GPU 백엔드 사용 (BGPQ에 권장)
runner = run_all_benchmarks(backend="gpu")
```

### 테스트 크기 및 반복 횟수 설정

```python
# 사용자 정의 테스트 크기
custom_sizes = [1000, 5000, 10000, 50000]
runner = run_all_benchmarks(test_sizes=custom_sizes, num_iterations=5)
```

## 📊 벤치마크 대상

### 1. Stack vs Python list
- **Xtructure Stack**: JAX 최적화된 배치 연산
- **Python list**: 표준 append/pop 연산

### 2. Queue vs Python deque
- **Xtructure Queue**: JAX 최적화된 배치 연산
- **Python deque**: 표준 append/popleft 연산

### 3. HashTable vs Python dict
- **Xtructure HashTable**: Cuckoo 해싱 병렬 연산
- **Python dict**: 표준 update/get 연산

### 4. BGPQ vs Python heapq
- **Xtructure BGPQ**: 배치 GPU 우선순위 큐
- **Python heapq**: 표준 heappush/heappop 연산

## 📈 출력 정보

### 하드웨어 정보
```
================================================================================
SYSTEM INFORMATION
================================================================================
System: Linux
Platform: Linux-6.8.0-1024-aws-x86_64-with-glibc2.39
Architecture: 64bit
Processor: Intel(R) Xeon(R) CPU E5-2686 v4 @ 2.30GHz
Python Version: 3.13.0
CPU Cores: 2
CPU Threads: 4
Total Memory: 7.6 GB
Available Memory: 6.2 GB

JAX Version: 0.6.2
JAX Backend: cpu
JAX Devices: 1
  - CpuDevice(id=0)

GPU: Not detected
```

### 성능 결과
```
================================================================================
DETAILED BENCHMARK REPORT
================================================================================

Xtructure_Stack - push_batch:
Size       Time (ms)       Throughput (ops/sec)
--------------------------------------------------
100        0.03            3808557             
1000       0.03            32036907            
5000       0.02            223237539           
10000      0.04            269769617           

================================================================================
PERFORMANCE COMPARISONS (Xtructure vs Python)
================================================================================
Xtructure_Stack push_batch (size 5,000): 4.65x FASTER than Python_list
Xtructure_Queue enqueue_batch (size 10,000): 7.32x FASTER than Python_deque
```

### CSV 결과 파일
모든 결과는 타임스탬프가 포함된 CSV 파일로 자동 저장됩니다:
```
xtructure_benchmark_results_1751454996.csv
```

## 🎯 성능 해석

### 크로스오버 포인트
- **Stack Push**: 5,000+ 요소에서 Xtructure가 더 빠름
- **Queue Enqueue**: 5,000+ 요소에서 Xtructure가 더 빠름  
- **HashTable Insert**: 10,000+ 요소에서 경쟁력 있음
- **Pop/Dequeue**: 모든 크기에서 Python이 우세

### Xtructure 사용 권장 사례
✅ **대용량 데이터셋** (5,000+ 요소)  
✅ **배치 연산** 중심  
✅ **GPU 가속** 계획  
✅ **구조화된/중첩 데이터**  
✅ **ML/과학 컴퓨팅** 파이프라인  

### Python 표준 라이브러리 권장 사례
✅ **소규모 데이터셋** (<5,000 요소)  
✅ **개별 연산** 중심  
✅ **최대 단일 연산 성능**  
✅ **전통적 애플리케이션** 로직  
✅ **단순함 우세**  

## ⚠️ 주의사항

### BGPQ 제한사항
- **CPU 백엔드에서 제한적**: Pallas 연산으로 인한 한계
- **GPU 백엔드 권장**: 최적 성능을 위해 CUDA 사용
- 에러 발생 시 자동으로 스킵됨

### 메모리 사용량
- 대용량 테스트 시 메모리 사용량 주의
- 필요시 `test_sizes` 조정

## 🔧 문제 해결

### BGPQ 에러
```
ValueError: Only interpret mode is supported on CPU backend
```
**해결책**: GPU 백엔드 사용 또는 BGPQ 벤치마크 제외

### Import 에러
```bash
pip install jax chex numpy tabulate
```

### 성능 변동성
벤치마크 결과는 다음 요인에 따라 달라질 수 있습니다:
- 시스템 부하
- Python/JAX 버전
- 하드웨어 사양
- 백그라운드 프로세스

## 📧 API 사용 예제

```python
from benchmarks import BenchmarkRunner
from benchmarks.stack_benchmark import run_stack_benchmark

# 전체 벤치마크 실행
runner = BenchmarkRunner(
    test_sizes=[1000, 5000, 10000],
    num_iterations=3,
    backend="cpu"
)
results = runner.run_all_benchmarks()

# 상세 보고서 출력
runner.print_detailed_report()

# CSV 저장
runner.save_results_to_csv("my_results.csv")

# 특정 데이터 구조만 테스트
stack_results = run_stack_benchmark([1000, 5000], num_iterations=5)
```

## 📝 결과 분석

생성된 CSV 파일을 사용하여 추가 분석을 수행할 수 있습니다:

```python
import pandas as pd
import matplotlib.pyplot as plt

# 결과 로드
df = pd.read_csv("xtructure_benchmark_results_xxx.csv")

# 크기별 성능 비교 플롯
xtructure_data = df[df['data_structure'].str.contains('Xtructure')]
python_data = df[df['data_structure'].str.contains('Python')]

plt.figure(figsize=(12, 8))
# 여기에 시각화 코드 추가
```

더 자세한 분석은 생성된 `benchmark_results.md` 파일을 참조하세요.