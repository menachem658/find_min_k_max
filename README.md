# find_min_k_max



## Installation
```bash
pip install find_min_k_max
```

## Usage
```python
from find_min_k_max import lower_edge, upper_edge

# Example usage of functions
n = 100
alpha = 0.05
k_max_values = list(range(6, 13))
ab_size_vec = np.array([100]) 
differences_list = []

for k_max in k_max_values:
    CI = CI_for_non_frequent_symbols(n, k_max, alpha * (1 - n / (ab_size_vec * (k_max + 1))))
    differences = CI[:, 1] - CI[:, 0] 
    differences_list.append(differences)

log_values = []
min_log_value = float('inf')
best_k_max = None

for k_max, differences in zip(k_max_values, differences_list):
    result = expected_log_S_with_max_E_ri(n, alpha, k_max, differences, ab_size_vec)
    log_values.append(result)
    
    if result < min_log_value:
        min_log_value = result
        best_k_max = k_max

print(f'The k_max with the lowest expected log S(X^n) is {best_k_max} with a value of {min_log_value:.4f}')
```
