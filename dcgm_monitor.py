#!/usr/bin/env python3
"""
DCGM GPU Monitor - 監控 GPU 溫度、功耗、FLOPS 並生成報告
用法: python dcgm_monitor.py [監控秒數] [採樣間隔]
範例: python dcgm_monitor.py 600 1    # 監控 600 秒，每秒採樣
"""

import subprocess
import time
import csv
import sys
import os
from datetime import datetime
from collections import defaultdict

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
except ImportError:
    print("安裝 matplotlib...")
    subprocess.run([sys.executable, "-m", "pip", "install", "matplotlib", "-q"])
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

try:
    import numpy as np
except ImportError:
    print("安裝 numpy...")
    subprocess.run([sys.executable, "-m", "pip", "install", "numpy", "-q"])
    import numpy as np


def get_gpu_count():
    """取得 GPU 數量"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=count", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        )
        return int(result.stdout.strip().split('\n')[0])
    except:
        return 1


def get_power_limit():
    """取得 GPU Power Limit"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=power.limit", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        )
        limits = [float(x.strip()) for x in result.stdout.strip().split('\n') if x.strip()]
        return limits
    except:
        return [1000.0]


def get_temp_limit():
    """取得 GPU Temperature Limit"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu.tlimit", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        )
        limits = []
        for line in result.stdout.strip().split('\n'):
            line = line.strip()
            if line and line != '[Not Supported]':
                try:
                    limits.append(float(line))
                except:
                    limits.append(83.0)  # 預設值
            else:
                limits.append(83.0)
        return limits if limits else [83.0]
    except:
        return [83.0]


def collect_nvidia_smi_data():
    """使用 nvidia-smi 收集基本 GPU 數據"""
    try:
        result = subprocess.run(
            ["nvidia-smi", 
             "--query-gpu=index,temperature.gpu,power.draw,utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        )
        
        data = []
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                parts = [x.strip() for x in line.split(',')]
                if len(parts) >= 6:
                    try:
                        data.append({
                            'gpu_id': int(parts[0]),
                            'temp': float(parts[1]) if parts[1] != '[Not Supported]' else 0,
                            'power': float(parts[2]) if parts[2] != '[Not Supported]' else 0,
                            'util': float(parts[3]) if parts[3] != '[Not Supported]' else 0,
                            'mem_used': float(parts[4]) if parts[4] != '[Not Supported]' else 0,
                            'mem_total': float(parts[5]) if parts[5] != '[Not Supported]' else 0,
                        })
                    except ValueError:
                        continue
        return data
    except Exception as e:
        print(f"nvidia-smi 錯誤: {e}")
        return []


def collect_dcgm_data():
    """使用 dcgmi 收集 FLOPS 數據"""
    try:
        # 嘗試用 dcgmi dmon 取得效能數據
        result = subprocess.run(
            ["dcgmi", "dmon", "-e", "1001,1002,1003,1004,1005", "-c", "1"],
            capture_output=True, text=True, timeout=15
        )
        
        flops_data = {}
        for line in result.stdout.strip().split('\n'):
            if line.startswith('GPU') or line.startswith('#') or not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 6:
                try:
                    gpu_id = int(parts[0])
                    flops_data[gpu_id] = {
                        'fp64_flops': float(parts[1]) if parts[1] != 'N/A' else 0,
                        'fp32_flops': float(parts[2]) if parts[2] != 'N/A' else 0,
                        'fp16_flops': float(parts[3]) if parts[3] != 'N/A' else 0,
                        'tensor_flops': float(parts[4]) if parts[4] != 'N/A' else 0,
                        'mem_bw': float(parts[5]) if parts[5] != 'N/A' else 0,
                    }
                except (ValueError, IndexError):
                    continue
        return flops_data
    except Exception as e:
        return {}


def monitor_gpus(duration=60, interval=1, output_prefix="gpu_monitor"):
    """主監控函數"""
    
    print("=" * 60)
    print("DCGM GPU Monitor")
    print("=" * 60)
    
    gpu_count = get_gpu_count()
    power_limits = get_power_limit()
    temp_limits = get_temp_limit()
    
    print(f"偵測到 {gpu_count} 張 GPU")
    print(f"Power Limits: {power_limits}")
    print(f"Temp Limits: {temp_limits}")
    print(f"監控時間: {duration} 秒")
    print(f"採樣間隔: {interval} 秒")
    print("=" * 60)
    
    # 資料儲存
    all_data = []
    gpu_stats = defaultdict(lambda: {
        'temps': [], 'powers': [], 'utils': [],
        'fp32_flops': [], 'fp64_flops': [], 'tensor_flops': [],
        'mem_bw': []
    })
    
    start_time = time.time()
    sample_count = 0
    
    print("\n開始監控... (Ctrl+C 可提前結束)\n")
    print(f"{'Time':<12} {'GPU':<5} {'Temp(C)':<10} {'Power(W)':<12} {'Util(%)':<10} {'TensorFLOPS':<12}")
    print("-" * 70)
    
    try:
        while time.time() - start_time < duration:
            timestamp = datetime.now()
            elapsed = time.time() - start_time
            
            # 收集數據
            smi_data = collect_nvidia_smi_data()
            dcgm_data = collect_dcgm_data()
            
            for gpu in smi_data:
                gpu_id = gpu['gpu_id']
                
                # 合併 DCGM 數據
                dcgm = dcgm_data.get(gpu_id, {})
                
                record = {
                    'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                    'elapsed_sec': round(elapsed, 1),
                    'gpu_id': gpu_id,
                    'temp_c': gpu['temp'],
                    'temp_limit_c': temp_limits[gpu_id] if gpu_id < len(temp_limits) else temp_limits[0],
                    'power_w': gpu['power'],
                    'power_limit_w': power_limits[gpu_id] if gpu_id < len(power_limits) else power_limits[0],
                    'util_pct': gpu['util'],
                    'mem_used_mb': gpu['mem_used'],
                    'mem_total_mb': gpu['mem_total'],
                    'fp32_flops': dcgm.get('fp32_flops', 0),
                    'fp64_flops': dcgm.get('fp64_flops', 0),
                    'tensor_flops': dcgm.get('tensor_flops', 0),
                    'mem_bw_pct': dcgm.get('mem_bw', 0),
                }
                
                all_data.append(record)
                
                # 統計
                stats = gpu_stats[gpu_id]
                stats['temps'].append(gpu['temp'])
                stats['powers'].append(gpu['power'])
                stats['utils'].append(gpu['util'])
                stats['fp32_flops'].append(dcgm.get('fp32_flops', 0))
                stats['fp64_flops'].append(dcgm.get('fp64_flops', 0))
                stats['tensor_flops'].append(dcgm.get('tensor_flops', 0))
                stats['mem_bw'].append(dcgm.get('mem_bw', 0))
                
                # 即時顯示
                print(f"{elapsed:>8.1f}s   {gpu_id:<5} {gpu['temp']:<10.1f} {gpu['power']:<12.1f} {gpu['util']:<10.1f} {dcgm.get('tensor_flops', 0):<12.1f}")
            
            sample_count += 1
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n\n監控被中斷...")
    
    print("\n" + "=" * 60)
    print("監控結束，正在生成報告...")
    print("=" * 60)
    
    # 保存 CSV
    csv_file = f"{output_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with open(csv_file, 'w', newline='') as f:
        if all_data:
            writer = csv.DictWriter(f, fieldnames=all_data[0].keys())
            writer.writeheader()
            writer.writerows(all_data)
    print(f"\n✓ CSV 已保存: {csv_file}")
    
    # 生成統計摘要
    summary_file = f"{output_prefix}_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    summary_data = []
    
    print("\n" + "=" * 60)
    print("統計摘要")
    print("=" * 60)
    
    for gpu_id, stats in sorted(gpu_stats.items()):
        temp_limit = temp_limits[gpu_id] if gpu_id < len(temp_limits) else temp_limits[0]
        power_limit = power_limits[gpu_id] if gpu_id < len(power_limits) else power_limits[0]
        
        summary = {
            'gpu_id': gpu_id,
            'temp_avg': np.mean(stats['temps']),
            'temp_max': np.max(stats['temps']),
            'temp_min': np.min(stats['temps']),
            'temp_limit': temp_limit,
            'power_avg': np.mean(stats['powers']),
            'power_max': np.max(stats['powers']),
            'power_min': np.min(stats['powers']),
            'power_limit': power_limit,
            'util_avg': np.mean(stats['utils']),
            'util_max': np.max(stats['utils']),
            'fp32_flops_avg': np.mean(stats['fp32_flops']),
            'fp32_flops_max': np.max(stats['fp32_flops']),
            'fp64_flops_avg': np.mean(stats['fp64_flops']),
            'fp64_flops_max': np.max(stats['fp64_flops']),
            'tensor_flops_avg': np.mean(stats['tensor_flops']),
            'tensor_flops_max': np.max(stats['tensor_flops']),
            'mem_bw_avg': np.mean(stats['mem_bw']),
            'mem_bw_max': np.max(stats['mem_bw']),
            'samples': len(stats['temps']),
        }
        summary_data.append(summary)
        
        print(f"\n--- GPU {gpu_id} ---")
        print(f"溫度: 平均 {summary['temp_avg']:.1f}°C | 最高 {summary['temp_max']:.1f}°C | 上限 {temp_limit}°C")
        print(f"功耗: 平均 {summary['power_avg']:.1f}W | 最高 {summary['power_max']:.1f}W | 上限 {power_limit}W")
        print(f"使用率: 平均 {summary['util_avg']:.1f}% | 最高 {summary['util_max']:.1f}%")
        print(f"Tensor FLOPS: 平均 {summary['tensor_flops_avg']:.1f} | 最高 {summary['tensor_flops_max']:.1f}")
        print(f"FP32 FLOPS: 平均 {summary['fp32_flops_avg']:.1f} | 最高 {summary['fp32_flops_max']:.1f}")
        print(f"FP64 FLOPS: 平均 {summary['fp64_flops_avg']:.1f} | 最高 {summary['fp64_flops_max']:.1f}")
    
    # 保存摘要 CSV
    with open(summary_file, 'w', newline='') as f:
        if summary_data:
            writer = csv.DictWriter(f, fieldnames=summary_data[0].keys())
            writer.writeheader()
            writer.writerows(summary_data)
    print(f"\n✓ 摘要 CSV 已保存: {summary_file}")
    
    # 生成圖表
    if all_data:
        chart_file = generate_charts(all_data, gpu_stats, power_limits, temp_limits, output_prefix)
        print(f"✓ 圖表已保存: {chart_file}")
    
    return csv_file, summary_file


def generate_charts(all_data, gpu_stats, power_limits, temp_limits, output_prefix):
    """生成監控圖表"""
    
    chart_file = f"{output_prefix}_charts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    
    # 準備每個 GPU 的時序數據
    gpu_ids = sorted(gpu_stats.keys())
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle('GPU Monitoring Report', fontsize=14, fontweight='bold')
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(gpu_ids)))
    
    # 每個 GPU 的採樣數
    samples_per_gpu = len(gpu_stats[gpu_ids[0]]['temps'])
    x = np.arange(samples_per_gpu)
    
    # 1. 溫度曲線
    ax1 = axes[0, 0]
    for i, gpu_id in enumerate(gpu_ids):
        ax1.plot(x, gpu_stats[gpu_id]['temps'], label=f'GPU {gpu_id}', color=colors[i], linewidth=1.5)
    temp_limit = temp_limits[0] if temp_limits else 83
    ax1.axhline(y=temp_limit, color='red', linestyle='--', label=f'Limit ({temp_limit}°C)', alpha=0.7)
    ax1.set_xlabel('Sample')
    ax1.set_ylabel('Temperature (°C)')
    ax1.set_title('GPU Temperature')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # 2. 功耗曲線
    ax2 = axes[0, 1]
    for i, gpu_id in enumerate(gpu_ids):
        ax2.plot(x, gpu_stats[gpu_id]['powers'], label=f'GPU {gpu_id}', color=colors[i], linewidth=1.5)
    power_limit = power_limits[0] if power_limits else 1000
    ax2.axhline(y=power_limit, color='red', linestyle='--', label=f'Limit ({power_limit}W)', alpha=0.7)
    ax2.set_xlabel('Sample')
    ax2.set_ylabel('Power (W)')
    ax2.set_title('GPU Power Consumption')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # 3. GPU 使用率
    ax3 = axes[1, 0]
    for i, gpu_id in enumerate(gpu_ids):
        ax3.plot(x, gpu_stats[gpu_id]['utils'], label=f'GPU {gpu_id}', color=colors[i], linewidth=1.5)
    ax3.set_xlabel('Sample')
    ax3.set_ylabel('Utilization (%)')
    ax3.set_title('GPU Utilization')
    ax3.set_ylim(0, 105)
    ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # 4. Tensor FLOPS
    ax4 = axes[1, 1]
    for i, gpu_id in enumerate(gpu_ids):
        ax4.plot(x, gpu_stats[gpu_id]['tensor_flops'], label=f'GPU {gpu_id}', color=colors[i], linewidth=1.5)
    ax4.set_xlabel('Sample')
    ax4.set_ylabel('Tensor Core Activity (%)')
    ax4.set_title('Tensor Core FLOPS')
    ax4.legend(loc='upper right', fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    # 5. 統計長條圖 - 溫度 & 功耗
    ax5 = axes[2, 0]
    bar_width = 0.35
    x_bars = np.arange(len(gpu_ids))
    
    temp_avgs = [np.mean(gpu_stats[gid]['temps']) for gid in gpu_ids]
    temp_maxs = [np.max(gpu_stats[gid]['temps']) for gid in gpu_ids]
    
    ax5.bar(x_bars - bar_width/2, temp_avgs, bar_width, label='Average', color='steelblue')
    ax5.bar(x_bars + bar_width/2, temp_maxs, bar_width, label='Peak', color='coral')
    ax5.axhline(y=temp_limits[0], color='red', linestyle='--', label=f'Limit', alpha=0.7)
    ax5.set_xlabel('GPU ID')
    ax5.set_ylabel('Temperature (°C)')
    ax5.set_title('Temperature Statistics')
    ax5.set_xticks(x_bars)
    ax5.set_xticklabels([f'GPU {gid}' for gid in gpu_ids])
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. 統計長條圖 - 功耗
    ax6 = axes[2, 1]
    
    power_avgs = [np.mean(gpu_stats[gid]['powers']) for gid in gpu_ids]
    power_maxs = [np.max(gpu_stats[gid]['powers']) for gid in gpu_ids]
    
    ax6.bar(x_bars - bar_width/2, power_avgs, bar_width, label='Average', color='steelblue')
    ax6.bar(x_bars + bar_width/2, power_maxs, bar_width, label='Peak', color='coral')
    ax6.axhline(y=power_limits[0], color='red', linestyle='--', label=f'Limit', alpha=0.7)
    ax6.set_xlabel('GPU ID')
    ax6.set_ylabel('Power (W)')
    ax6.set_title('Power Statistics')
    ax6.set_xticks(x_bars)
    ax6.set_xticklabels([f'GPU {gid}' for gid in gpu_ids])
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(chart_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    return chart_file


if __name__ == "__main__":
    duration = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    interval = float(sys.argv[2]) if len(sys.argv) > 2 else 1
    output_prefix = sys.argv[3] if len(sys.argv) > 3 else "gpu_monitor"
    
    print(f"\n使用方式: python {sys.argv[0]} [監控秒數] [採樣間隔] [輸出前綴]")
    print(f"範例: python {sys.argv[0]} 600 1 b200_test\n")
    
    monitor_gpus(duration, interval, output_prefix)
