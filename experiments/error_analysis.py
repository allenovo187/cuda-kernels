import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def fp16(x):
    return np.float16(x).astype(np.float32)

def online_softmax_ref(x):
    x = x.astype(np.float32)
    m = -np.inf
    d = 0.0
    for xi in x:
        new_m = max(m, xi)
        scale = np.exp(m - new_m)
        d = d * scale + np.exp(xi - new_m)
        m = new_m
    
    out = np.exp(x - m) / d
    return out, d, m

def online_softmax_fp16_pure(x):
    x = x.astype(np.float32)
    m = fp16(-1e9)
    d = fp16(0.0)
    
    for xi in x:
        new_m = fp16(max(m, xi))
        scale = fp16(np.exp(fp16(m - new_m)))
        exp_x = fp16(np.exp(fp16(xi - new_m)))
        d_scaled = fp16(d * scale)
        d = fp16(d_scaled + exp_x)
        m = new_m
    
    out = fp16(np.exp(fp16(x - m)) / d)
    return out, d, m

def online_softmax_fp32_acc(x):
    x = x.astype(np.float32)
    m = -np.inf
    d = 0.0
    
    for xi in x:
        new_m = max(m, xi)
        scale = np.exp(m - new_m)
        d = d * scale + np.exp(xi - new_m)
        m = new_m
    
    out = np.exp(x - m) / d
    return out, d, m

def online_softmax_mixed(x):
    x = x.astype(np.float32)
    m = fp16(-1e9)
    d = 0.0          

    for xi in x:
        new_m = fp16(max(m, xi))
        scale = np.exp(float(m) - float(new_m))
        exp_x = np.exp(float(xi) - float(new_m))
        d = d * scale + exp_x
        m = new_m

    out = np.exp(x - float(m)) / d
    return out, d, m

def online_softmax_kahan(x):
    x = x.astype(np.float32)
    m = fp16(-1e9)
    d = fp16(0.0)
    c = fp16(0.0)

    for xi in x:
        new_m = fp16(max(m, xi))
        scale = fp16(np.exp(fp16(m - new_m)))
        exp_x = fp16(np.exp(fp16(xi - new_m)))

        d_scaled = fp16(d * scale)
        c_scaled = fp16(c * scale)

        y = fp16(exp_x - c_scaled)
        t = fp16(d_scaled + y)
        c = fp16(fp16(t - d_scaled) - y)
        d = t

        m = new_m

    out = fp16(np.exp(fp16(x - m)) / d)
    return out, d, m

def run_experiment(seq_lengths, distributions):
    results = {name: {"fp16_err": [], "fp32_err": [], "mixed_err": [],
                       "kahan_err": [], "bound": []} 
               for name in distributions}
    
    for n in seq_lengths:
        for name, gen in distributions.items():
            x = gen(n)
            
            out_ref, d_ref, _ = online_softmax_ref(x)
            out_fp16, d_fp16, _ = online_softmax_fp16_pure(x)
            out_fp32, d_fp32, _ = online_softmax_fp32_acc(x)
            out_mix, d_mix, _ = online_softmax_mixed(x)
            out_kahan, d_kahan, _ = online_softmax_kahan(x)
            
            err_fp16 = abs(d_fp16 - d_ref) / d_ref
            err_fp32 = abs(d_fp32 - d_ref) / d_ref
            err_mix = abs(d_mix - d_ref) / d_ref
            err_kahan = abs(d_kahan - d_ref) / d_ref
            
            u = 2**(-11)
            bound = 3 * n * u    # Theoretical worst-case: 3·n·u
            
            results[name]["fp16_err"].append(err_fp16)
            results[name]["fp32_err"].append(err_fp32)
            results[name]["mixed_err"].append(err_mix)
            results[name]["kahan_err"].append(err_kahan)
            results[name]["bound"].append(bound)
            
            print(f"n={n:6d} | {name:12s} | FP16={err_fp16:.3e} | "
                  f"FP32={err_fp32:.3e} | Mixed={err_mix:.3e} | "
                  f"Kahan={err_kahan:.3e}")
    
    return results

def plot_results(seq_lengths, results, save_path="results/error_growth.png"):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    dist_names = list(results.keys())
    
    for idx, name in enumerate(dist_names):
        ax = axes[idx]
        r = results[name]
        
        ax.loglog(seq_lengths, r["fp16_err"], 'o-', label='FP16 Pure', color='red')
        ax.loglog(seq_lengths, r["mixed_err"], 'D-', label='Mixed (FP16+FP32)', color='blue')
        ax.loglog(seq_lengths, r["kahan_err"], '^-', label='FP16+Kahan', color='orange')
        ax.loglog(seq_lengths, r["bound"], '--', label='Worst-case 3n·u', color='gray')
        
        ax.set_xlabel('Sequence Length')
        ax.set_ylabel('Relative Error of Denominator $d_n$')
        ax.set_title(f'Input: {name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=250)
    print(f"\nPlot saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    np.random.seed(42)
    
    seq_lengths = [128, 256, 512, 1024, 2048, 4096, 8192, 32768]
    
    distributions = {
        "Normal(0,1)": lambda n: np.random.randn(n).astype(np.float32),
        "Increasing": lambda n: (np.arange(n) / np.sqrt(n)).astype(np.float32),
        "Spiky": lambda n: np.array([10.0 if i == n//2 else 0.0 for i in range(n)], 
                                     dtype=np.float32),
    }
    
    results = run_experiment(seq_lengths, distributions)
    plot_results(seq_lengths, results)
