# `_update_array_on_condition` Logic Review / 로직 검증 보고서

| Line | Code | English Analysis | 한국어 분석 |
| --- | --- | --- | --- |
| 8 | `def _update_array_on_condition(...):` | Establishes helper for conditional updates on JAX arrays; signature matches callers. | JAX 배열 조건부 업데이트용 헬퍼를 정의하며, 호출부와 동일한 시그니처를 유지합니다. |
| 9 | `original_array: jnp.ndarray,` | Accepts target array; annotated to clarify expected JAX array input. | 업데이트 대상 배열을 입력으로 받아 JAX 배열임을 명시합니다. |
| 10 | `indices: Union[jnp.ndarray, tuple[...]` | Allows either flat or advanced (tuple) indexing; supports existing API usage. | 단일 배열 및 튜플 기반 고급 인덱싱 모두 허용해 기존 API 사용 패턴을 지원합니다. |
| 11 | `condition: jnp.ndarray,` | Boolean mask per update slot; later normalized to ensure dtype safety. | 각 업데이트에 대응하는 불리언 마스크이며 이후 dtype 안전성을 위해 정규화합니다. |
| 12 | `values_to_set: Any,` | Flexible payload so callers can provide scalars or tensors. | 스칼라·텐서 모두 전달 가능하도록 유연성을 부여합니다. |
| 13 | `) -> jnp.ndarray:` | Returns new array maintaining functional style; no in-place mutation. | 함수형 스타일로 새로운 배열을 반환하며, 제자리 변경이 없습니다. |
| 14 | `""" ... """` | Docstring restates “first True wins” contract; aligns with tests. | 도큐스트링에서 “처음 True 우선” 계약을 재확인하며 테스트와 일치합니다. |
| 20 | `if isinstance(indices, tuple) ...` | Detects advanced indexing; ensures compatible path before flat logic. | 튜플 형태 인덱싱을 감지해 평탄화 로직 이전에 호환 경로를 사용하도록 합니다. |
| 21 | `batch_rank = len(indices)` | Derives number of batch axes to ravel; safe for tuple inputs. | 배치 축 개수를 계산해 ravel 대상 차원을 식별합니다. |
| 22 | `batch_shape = original_array.shape[:batch_rank]` | Captures leading dimensions that indexing spans; used for raveling. | 인덱싱이 가리키는 선행 차원 형태를 추출해 ravel용으로 사용합니다. |
| 23 | `item_shape = original_array.shape[batch_rank:]` | Stores trailing item shape for later reshape back. | 후속 복원 시 필요하므로 나머지 아이템 형태를 저장합니다. |
| 25 | `flat_batch_size = np.prod(batch_shape).item()` | Uses NumPy prod to compute flatten size; deterministic scalar. | 배치 차원 크기 곱을 NumPy로 계산해 확정적인 스칼라를 얻습니다. |
| 26 | `reshaped_array = original_array.reshape((flat_batch_size, *item_shape))` | Flattens batch axes to simplify 1D scatter; no data copy in JAX. | 배치 축을 평탄화해 1D 스캐터로 단순화하며 JAX에서 복사 없이 처리됩니다. |
| 28 | `raveled_indices = jnp.ravel_multi_index(..., mode="clip")` | Converts multi-d indices to flat positions while clipping out-of-range gracefully. | 다차원 인덱스를 평탄한 위치로 변환하며 mode="clip"으로 경계 밖 값을 안전하게 처리합니다. |
| 30 | `result = _update_array_on_condition(...` | Recursively reuse flattened logic; ensures shared duplicate handling path. | 평탄화된 로직을 재사용하기 위해 재귀 호출하여 중복 처리 방식을 일관되게 유지합니다. |
| 33 | `return result.reshape(original_array.shape)` | Restores original shape, undoing earlier flatten; pure transformation. | 앞서 평탄화한 형태를 되돌려 원래 형태로 복원합니다. |
| 35 | `condition = jnp.asarray(condition, dtype=bool)` | Normalizes mask to boolean dtype to prevent implicit casts later. | 마스크를 불리언 dtype으로 변환해 이후 암묵적 형변환을 방지합니다. |
| 36 | `num_updates = len(condition)` | Derives update count; relies on 1-D mask invariant enforced by callers/tests. | 업데이트 개수를 계산하며 호출부/테스트가 1차원 마스크를 보장합니다. |
| 37 | `if num_updates == 0:` | Early exit for empty work; avoids downstream shape issues. | 작업이 없을 경우 조기 종료하여 이후 형태 처리 문제를 예방합니다. |
| 38 | `return original_array` | Preserves original array when no updates; no scatter overhead. | 업데이트가 없을 때 원본 배열을 그대로 반환해 불필요한 연산을 생략합니다. |
| 40 | `indices_array = jnp.asarray(indices)` | Coerces indices into JAX array, matching scatter expectations. | 인덱스를 JAX 배열로 변환해 스캐터가 요구하는 입력 형태를 맞춥니다. |
| 41 | `indices_array = jnp.reshape(indices_array, (num_updates,))` | Guarantees 1-D index list aligned with condition length; fails fast if mismatch. | 인덱스를 조건 길이에 맞춘 1차원 배열로 강제해 불일치 시 즉시 오류를 노출합니다. |
| 42 | `invalid_index = jnp.array(original_array.shape[0], dtype=indices_array.dtype)` | Builds sentinel equal to size (exclusive upper bound) for drop-mode filtering. | drop 모드 필터링을 위해 배열 크기(상한)와 동일한 센티널 인덱스를 생성합니다. |
| 43 | `safe_indices = jnp.where(condition, indices_array, invalid_index)` | Maps False slots to sentinel so scatter skips them without extra masking. | False 위치를 센티널로 대체해 별도 마스크 없이 스캐터가 건너뛰도록 합니다. |
| 45 | `safe_indices = jnp.flip(safe_indices, axis=0)` | Reverses order ensuring earliest True (original front) applies last. | 순서를 뒤집어 최초 True 항목이 마지막에 적용되도록 해 “처음 True 우선” 규칙을 지킵니다. |
| 47 | `value_array = jnp.asarray(values_to_set, dtype=original_array.dtype)` | Aligns update dtype with target array to avoid implicit casts in scatter. | 스캐터 중 암묵적 형변환을 방지하도록 업데이트 값을 대상 배열 dtype에 맞춥니다. |
| 48 | `if value_array.ndim > 0 and value_array.shape[0] == num_updates:` | Detects per-update payloads; avoids flipping scalars or mismatched shapes. | 업데이트별 값 배열만 감지해 스칼라 또는 불일치 형태는 건드리지 않습니다. |
| 49 | `value_array = jnp.flip(value_array, axis=0)` | Mirrors indices reversal so index-value pairs stay aligned. | 인덱스 뒤집기에 맞춰 값도 뒤집어 대응이 유지되도록 합니다. |
| 51 | `return original_array.at[safe_indices].set(value_array, mode="drop")` | Applies scatter update: sentinel entries dropped, reversed order enforces first-hit wins, no large broadcasts. | 스캐터 업데이트를 수행하며 센티널은 무시되고, 역순 적용으로 최초 히트만 남기며 대규모 브로드캐스트를 사용하지 않습니다. |

## Verification Notes / 검증 메모

- **Semantics 유지 확인** / Verified that reversing scatter order preserves the “first True wins” contract by reasoning through duplicate scenarios (`[True, True]`, `[True, False]`, `[False, True]`).  
  **중복 시나리오** (`[True, True]`, `[True, False]`, `[False, True]`)를 분석해 역순 스캐터가 “처음 True 우선” 규칙을 유지함을 확인했습니다.
- **False 항목 처리** / Sentinel index equal to array length leverages `mode="drop"` so False-flagged updates are discarded without requiring explicit masking.  
  **False 인덱스**는 배열 길이와 같은 센티널로 치환되어 `mode="drop"`에 의해 추가 마스크 없이 무시됩니다.
- **Broadcast 회피** / No use of `broadcast_to` or `take`; scatter operates on 1-D vectors sized `num_updates`, meeting the request to avoid wide broadcasts.  
  `broadcast_to`나 `take`를 사용하지 않아 스캐터가 `num_updates` 길이의 1차원 벡터만 처리하며, 대규모 브로드캐스트를 회피하는 요구사항을 만족합니다.
- **Scalar & tensor 호환성** / Scalars remain untouched while tensor payloads are flipped only when their leading axis matches `num_updates`, mirroring previous functionality.  
  스칼라는 그대로 유지되고, `num_updates`와 선행 축이 일치하는 텐서만 뒤집어 기존 동작과 호환됩니다.
- **테스트 상황** / Attempted to execute `python -m pytest tests/dataclass_test.py -k set_as_condition`; execution failed because `pytest` is not installed in the current environment.  
  `python -m pytest tests/dataclass_test.py -k set_as_condition`을 시도했으나 현재 환경에 `pytest`가 설치되어 있지 않아 실행되지 않았습니다.
