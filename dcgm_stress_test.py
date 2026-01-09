#!/usr/bin/env python3
"""
DCGM 壓力測試腳本 - 自動執行 Tensor/FP16/FP32 測試並監控
用法: sudo python dcgm_stress_test.py
"""

import subprocess
import time
import csv
import sys
import os
import signal
from datetime import datetime
from collections import defaultdict
from threading import Thread, Event

try:
    import matplotlib.pyplot as plt
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "matplotlib", "-q"])
    import matplotlib.pyplot as plt

try:
    import numpy as np
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "numpy", "-q"])
    import numpy as np

# ==================== 設定區 ====================
DCGM_PROFTESTER_PATH = "/snap/dcgm/62/usr/bin/dcgmproftester13"
TEST_DURATION = 300  # 每個測試 300 秒
SAMPLE_INTERVAL = 1  # 採樣間隔（秒）
OUTPUT_DIR = "dcgm_results"

# 測試項目
TESTS = [
    {"id": 1004, "name": "Tensor_Core"},
    {"id": 1003, "name": "FP16"},
    {"id": 1002, "name": "FP32"},
]
# ================================================

stop_monitor = Event()


def get_gpu_info():
    """取得 GPU 資訊"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,power.limit", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        )
        gpus = []
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                parts = [p.strip() for p in line.split(',')]
                gpus.append({
                    'id': int(parts[0]),
                    'name': parts[1],
                    'power_limit': float(parts[2]) if len(parts) > 2 else 1000
                })
        return gpus
    except Exception as e:
        print(f"錯誤: {e}")
        return []


def get_temp_limit():
    """取得 GPU 溫度上限"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu.tlimit", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        )
        limits = []
        for line in result.stdout.strip().split('\n'):
            line = line.strip()
            try:
                limits.append(float(line))
            except:
                limits.append(83.0)
        return limits if limits else [83.0]
    except:
        return [83.0]


def collect_gpu_data():
    """收集 GPU 數據"""
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
        return []


def collect_dcgm_flops():
    """使用 dcgmi 收集 FLOPS 數據 (實際 TFLOPS 數值)"""
    flops_data = {}
    
    try:
        # 使用 dcgmi dmon 取得效能數據
        # 1006=FP64 TFLOPS, 1007=FP32 TFLOPS, 1008=FP16 TFLOPS, 
        # 1009=Tensor TFLOPS, 1010=Mem BW GB/s
        result = subprocess.run(
            ["dcgmi", "dmon", "-e", "1006,1007,1008,1009,1010", "-c", "1"],
            capture_output=True, text=True, timeout=15
        )
        
        for line in result.stdout.strip().split('\n'):
            # 跳過標題行
            if line.startswith('GPU') or line.startswith('#') or line.startswith('Id') or not line.strip():
                continue
            
            parts = line.split()
            if len(parts) >= 6:
                try:
                    gpu_id = int(parts[0])
                    flops_data[gpu_id] = {
                        'fp64_tflops': float(parts[1]) if parts[1] not in ['N/A', 'NA', '-'] else 0,
                        'fp32_tflops': float(parts[2]) if parts[2] not in ['N/A', 'NA', '-'] else 0,
                        'fp16_tflops': float(parts[3]) if parts[3] not in ['N/A', 'NA', '-'] else 0,
                        'tensor_tflops': float(parts[4]) if parts[4] not in ['N/A', 'NA', '-'] else 0,
                        'mem_bw_gbps': float(parts[5]) if parts[5] not in ['N/A', 'NA', '-'] else 0,
                    }
                except (ValueError, IndexError):
                    continue
    except Exception as e:
        pass
    
    # 如果上面的 field ID 沒數據，嘗試其他方式
    if not flops_data:
        try:
            # 嘗試用 dcgmi 的 profiling metrics
            result = subprocess.run(
                ["dcgmi", "dmon", "-e", "203,204,1001,1002,1003,1004,1005", "-c", "1"],
                capture_output=True, text=True, timeout=15
            )
            
            for line in result.stdout.strip().split('\n'):
                if line.startswith('GPU') or line.startswith('#') or line.startswith('Id') or not line.strip():
                    continue
                
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        gpu_id = int(parts[0])
                        flops_data[gpu_id] = {
                            'fp64_tflops': float(parts[3]) if len(parts) > 3 and parts[3] not in ['N/A', 'NA', '-'] else 0,
                            'fp32_tflops': float(parts[4]) if len(parts) > 4 and parts[4] not in ['N/A', 'NA', '-'] else 0,
                            'fp16_tflops': float(parts[5]) if len(parts) > 5 and parts[5] not in ['N/A', 'NA', '-'] else 0,
                            'tensor_tflops': float(parts[6]) if len(parts) > 6 and parts[6] not in ['N/A', 'NA', '-'] else 0,
                            'mem_bw_gbps': float(parts[7]) if len(parts) > 7 and parts[7] not in ['N/A', 'NA', '-'] else 0,
                        }
                    except (ValueError, IndexError):
                        continue
        except:
            pass
    
    return flops_data


def get_gpu_theoretical_flops():
    """取得 GPU 理論 FLOPS (用於參考)"""
    # B200 理論值 (參考)
    return {
        'fp64_peak': 40.0,      # TFLOPS
        'fp32_peak': 80.0,      # TFLOPS
        'fp16_peak': 160.0,     # TFLOPS
        'tensor_peak': 2500.0,  # TFLOPS (FP8)
        'mem_bw_peak': 8000.0,  # GB/s (HBM3e)
    }


def get_system_sensors():
    """取得系統溫度和功耗"""
    system_data = {
        'inlet_temp': 0,
        'outlet_temp': 0,
        'cpu_temp': 0,
        'system_power': 0,
    }
    
    try:
        result = subprocess.run(
            ["ipmitool", "sensor", "list"],
            capture_output=True, text=True, timeout=10
        )
        
        for line in result.stdout.split('\n'):
            line_lower = line.lower()
            parts = [p.strip() for p in line.split('|')]
            if len(parts) >= 2:
                try:
                    value = float(parts[1].split()[0]) if parts[1].split() else 0
                except:
                    continue
                
                if any(x in line_lower for x in ['inlet', 'ambient']) and 'temp' in line_lower:
                    system_data['inlet_temp'] = value
                if any(x in line_lower for x in ['outlet', 'exhaust']) and 'temp' in line_lower:
                    system_data['outlet_temp'] = value
                if 'cpu' in line_lower and 'temp' in line_lower and system_data['cpu_temp'] == 0:
                    system_data['cpu_temp'] = value
    except:
        pass
    
    try:
        result = subprocess.run(
            ["ipmitool", "dcmi", "power", "reading"],
            capture_output=True, text=True, timeout=10
        )
        for line in result.stdout.split('\n'):
            if 'instantaneous' in line.lower() or 'current' in line.lower():
                parts = line.split(':')
                if len(parts) >= 2:
                    try:
                        system_data['system_power'] = float(parts[1].strip().split()[0])
                    except:
                        pass
    except:
        pass
    
    return system_data


def monitor_thread(test_name, duration, output_prefix, all_results):
    """監控執行緒"""
    gpu_stats = defaultdict(lambda: {
        'temps': [], 'powers': [], 'utils': [],
        'fp64_tflops': [], 'fp32_tflops': [], 'fp16_tflops': [],
        'tensor_tflops': [], 'mem_bw_gbps': []
    })
    system_stats = {
        'inlet_temps': [], 'outlet_temps': [], 'cpu_temps': [], 'system_powers': []
    }
    all_data = []
    
    temp_limits = get_temp_limit()
    gpus = get_gpu_info()
    power_limits = [g['power_limit'] for g in gpus]
    
    start_time = time.time()
    
    while not stop_monitor.is_set() and (time.time() - start_time) < duration + 30:
        timestamp = datetime.now()
        elapsed = time.time() - start_time
        
        gpu_data = collect_gpu_data()
        dcgm_flops = collect_dcgm_flops()
        sys_data = get_system_sensors()
        
        system_stats['inlet_temps'].append(sys_data['inlet_temp'])
        system_stats['outlet_temps'].append(sys_data['outlet_temp'])
        system_stats['cpu_temps'].append(sys_data['cpu_temp'])
        system_stats['system_powers'].append(sys_data['system_power'])
        
        for gpu in gpu_data:
            gpu_id = gpu['gpu_id']
            flops = dcgm_flops.get(gpu_id, {})
            
            record = {
                'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'elapsed_sec': round(elapsed, 1),
                'test_name': test_name,
                'gpu_id': gpu_id,
                'gpu_temp_c': gpu['temp'],
                'gpu_temp_limit_c': temp_limits[gpu_id] if gpu_id < len(temp_limits) else 83,
                'gpu_power_w': gpu['power'],
                'gpu_power_limit_w': power_limits[gpu_id] if gpu_id < len(power_limits) else 1000,
                'gpu_util_pct': gpu['util'],
                'gpu_mem_used_mb': gpu['mem_used'],
                'fp64_tflops': flops.get('fp64_tflops', 0),
                'fp32_tflops': flops.get('fp32_tflops', 0),
                'fp16_tflops': flops.get('fp16_tflops', 0),
                'tensor_tflops': flops.get('tensor_tflops', 0),
                'mem_bw_gbps': flops.get('mem_bw_gbps', 0),
                'sys_inlet_temp_c': sys_data['inlet_temp'],
                'sys_outlet_temp_c': sys_data['outlet_temp'],
                'sys_cpu_temp_c': sys_data['cpu_temp'],
                'sys_power_w': sys_data['system_power'],
            }
            all_data.append(record)
            
            gpu_stats[gpu_id]['temps'].append(gpu['temp'])
            gpu_stats[gpu_id]['powers'].append(gpu['power'])
            gpu_stats[gpu_id]['utils'].append(gpu['util'])
            gpu_stats[gpu_id]['fp64_tflops'].append(flops.get('fp64_tflops', 0))
            gpu_stats[gpu_id]['fp32_tflops'].append(flops.get('fp32_tflops', 0))
            gpu_stats[gpu_id]['fp16_tflops'].append(flops.get('fp16_tflops', 0))
            gpu_stats[gpu_id]['tensor_tflops'].append(flops.get('tensor_tflops', 0))
            gpu_stats[gpu_id]['mem_bw_gbps'].append(flops.get('mem_bw_gbps', 0))
        
        # 即時顯示
        if gpu_data:
            total_gpu_power = sum(g['power'] for g in gpu_data)
            avg_temp = np.mean([g['temp'] for g in gpu_data])
            avg_tensor = np.mean([dcgm_flops.get(g['gpu_id'], {}).get('tensor_tflops', 0) for g in gpu_data])
            avg_fp32 = np.mean([dcgm_flops.get(g['gpu_id'], {}).get('fp32_tflops', 0) for g in gpu_data])
            print(f"\r[{test_name}] {elapsed:>6.0f}s | Temp: {avg_temp:.1f}°C | GPU: {total_gpu_power:.0f}W | Sys: {sys_data['system_power']:.0f}W | Tensor: {avg_tensor:.1f} | FP32: {avg_fp32:.1f} TFLOPS", end='', flush=True)
        
        time.sleep(SAMPLE_INTERVAL)
    
    print()  # 換行
    
    # 儲存結果
    all_results[test_name] = {
        'data': all_data,
        'gpu_stats': dict(gpu_stats),
        'system_stats': system_stats,
        'power_limits': power_limits,
        'temp_limits': temp_limits,
    }
    
    # 保存 CSV
    csv_file = f"{output_prefix}_{test_name}.csv"
    if all_data:
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_data[0].keys())
            writer.writeheader()
            writer.writerows(all_data)
        print(f"  ✓ CSV 已保存: {csv_file}")


def run_stress_test(test_id, test_name, duration, gpu_list):
    """執行壓力測試"""
    gpu_str = ",".join(map(str, gpu_list))
    
    cmd = [
        DCGM_PROFTESTER_PATH,
        "--no-dcgm-validation",
        "-t", str(test_id),
        "-d", str(duration),
        "-i", gpu_str
    ]
    
    print(f"\n執行命令: {' '.join(cmd)}")
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid
        )
        return process
    except Exception as e:
        print(f"啟動測試失敗: {e}")
        return None


def generate_summary_and_charts(all_results, output_prefix):
    """生成摘要報告和圖表"""
    
    print("\n" + "=" * 70)
    print("測試摘要")
    print("=" * 70)
    
    summary_data = []
    
    for test_name, results in all_results.items():
        gpu_stats = results['gpu_stats']
        system_stats = results['system_stats']
        power_limits = results['power_limits']
        temp_limits = results['temp_limits']
        
        print(f"\n【{test_name}】")
        
        for gpu_id in sorted(gpu_stats.keys()):
            stats = gpu_stats[gpu_id]
            if not stats['temps']:
                continue
                
            temp_limit = temp_limits[gpu_id] if gpu_id < len(temp_limits) else 83
            power_limit = power_limits[gpu_id] if gpu_id < len(power_limits) else 1000
            
            row = {
                'test_name': test_name,
                'gpu_id': gpu_id,
                'temp_avg': np.mean(stats['temps']),
                'temp_max': np.max(stats['temps']),
                'temp_limit': temp_limit,
                'power_avg': np.mean(stats['powers']),
                'power_max': np.max(stats['powers']),
                'power_limit': power_limit,
                'util_avg': np.mean(stats['utils']),
                'util_max': np.max(stats['utils']),
                'fp64_tflops_avg': np.mean(stats['fp64_tflops']),
                'fp64_tflops_max': np.max(stats['fp64_tflops']),
                'fp32_tflops_avg': np.mean(stats['fp32_tflops']),
                'fp32_tflops_max': np.max(stats['fp32_tflops']),
                'fp16_tflops_avg': np.mean(stats['fp16_tflops']),
                'fp16_tflops_max': np.max(stats['fp16_tflops']),
                'tensor_tflops_avg': np.mean(stats['tensor_tflops']),
                'tensor_tflops_max': np.max(stats['tensor_tflops']),
                'mem_bw_gbps_avg': np.mean(stats['mem_bw_gbps']),
                'mem_bw_gbps_max': np.max(stats['mem_bw_gbps']),
                'sys_inlet_avg': np.mean([t for t in system_stats['inlet_temps'] if t > 0]) if any(t > 0 for t in system_stats['inlet_temps']) else 0,
                'sys_outlet_avg': np.mean([t for t in system_stats['outlet_temps'] if t > 0]) if any(t > 0 for t in system_stats['outlet_temps']) else 0,
                'sys_power_avg': np.mean([p for p in system_stats['system_powers'] if p > 0]) if any(p > 0 for p in system_stats['system_powers']) else 0,
                'sys_power_max': np.max([p for p in system_stats['system_powers'] if p > 0]) if any(p > 0 for p in system_stats['system_powers']) else 0,
            }
            summary_data.append(row)
            
            print(f"  GPU {gpu_id}: Temp {row['temp_avg']:.1f}°C (max {row['temp_max']:.1f}°C) | Power {row['power_avg']:.1f}W (max {row['power_max']:.1f}W)")
            print(f"          TFLOPS - Tensor: {row['tensor_tflops_avg']:.2f} (max {row['tensor_tflops_max']:.2f}) | FP32: {row['fp32_tflops_avg']:.2f} | FP16: {row['fp16_tflops_avg']:.2f} | FP64: {row['fp64_tflops_avg']:.2f}")
            print(f"          Mem BW: {row['mem_bw_gbps_avg']:.1f} GB/s (max {row['mem_bw_gbps_max']:.1f} GB/s)")
        
        if any(p > 0 for p in system_stats['system_powers']):
            print(f"  System: Power avg {row['sys_power_avg']:.0f}W (max {row['sys_power_max']:.0f}W)")
    
    # 保存摘要 CSV
    summary_file = f"{output_prefix}_summary.csv"
    if summary_data:
        with open(summary_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=summary_data[0].keys())
            writer.writeheader()
            writer.writerows(summary_data)
        print(f"\n✓ 摘要 CSV 已保存: {summary_file}")
    
    # 生成比較圖表
    generate_comparison_charts(all_results, output_prefix)


def generate_comparison_charts(all_results, output_prefix):
    """生成測試比較圖表"""
    
    chart_file = f"{output_prefix}_comparison.png"
    
    test_names = list(all_results.keys())
    if not test_names:
        return
    
    # 取得 GPU 數量
    first_test = all_results[test_names[0]]
    gpu_ids = sorted(first_test['gpu_stats'].keys())
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 14))
    fig.suptitle('DCGM Stress Test Comparison', fontsize=14, fontweight='bold')
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(test_names)))
    bar_width = 0.25
    x = np.arange(len(gpu_ids))
    
    # 1. 平均溫度比較
    ax1 = axes[0, 0]
    for i, test_name in enumerate(test_names):
        temps = [np.mean(all_results[test_name]['gpu_stats'].get(gid, {}).get('temps', [0])) for gid in gpu_ids]
        ax1.bar(x + i * bar_width, temps, bar_width, label=test_name, color=colors[i])
    temp_limit = first_test['temp_limits'][0] if first_test['temp_limits'] else 83
    ax1.axhline(y=temp_limit, color='red', linestyle='--', label='Limit', alpha=0.7)
    ax1.set_xlabel('GPU ID')
    ax1.set_ylabel('Temperature (°C)')
    ax1.set_title('Average GPU Temperature by Test')
    ax1.set_xticks(x + bar_width)
    ax1.set_xticklabels([f'GPU {gid}' for gid in gpu_ids])
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. 平均功耗比較
    ax2 = axes[0, 1]
    for i, test_name in enumerate(test_names):
        powers = [np.mean(all_results[test_name]['gpu_stats'].get(gid, {}).get('powers', [0])) for gid in gpu_ids]
        ax2.bar(x + i * bar_width, powers, bar_width, label=test_name, color=colors[i])
    power_limit = first_test['power_limits'][0] if first_test['power_limits'] else 1000
    ax2.axhline(y=power_limit, color='red', linestyle='--', label='Limit', alpha=0.7)
    ax2.set_xlabel('GPU ID')
    ax2.set_ylabel('Power (W)')
    ax2.set_title('Average GPU Power by Test')
    ax2.set_xticks(x + bar_width)
    ax2.set_xticklabels([f'GPU {gid}' for gid in gpu_ids])
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. 最大溫度比較
    ax3 = axes[1, 0]
    for i, test_name in enumerate(test_names):
        temps = [np.max(all_results[test_name]['gpu_stats'].get(gid, {}).get('temps', [0])) for gid in gpu_ids]
        ax3.bar(x + i * bar_width, temps, bar_width, label=test_name, color=colors[i])
    ax3.axhline(y=temp_limit, color='red', linestyle='--', label='Limit', alpha=0.7)
    ax3.set_xlabel('GPU ID')
    ax3.set_ylabel('Temperature (°C)')
    ax3.set_title('Peak GPU Temperature by Test')
    ax3.set_xticks(x + bar_width)
    ax3.set_xticklabels([f'GPU {gid}' for gid in gpu_ids])
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. 最大功耗比較
    ax4 = axes[1, 1]
    for i, test_name in enumerate(test_names):
        powers = [np.max(all_results[test_name]['gpu_stats'].get(gid, {}).get('powers', [0])) for gid in gpu_ids]
        ax4.bar(x + i * bar_width, powers, bar_width, label=test_name, color=colors[i])
    ax4.axhline(y=power_limit, color='red', linestyle='--', label='Limit', alpha=0.7)
    ax4.set_xlabel('GPU ID')
    ax4.set_ylabel('Power (W)')
    ax4.set_title('Peak GPU Power by Test')
    ax4.set_xticks(x + bar_width)
    ax4.set_xticklabels([f'GPU {gid}' for gid in gpu_ids])
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. TFLOPS 比較 (平均)
    ax5 = axes[2, 0]
    flops_types = ['tensor_tflops', 'fp32_tflops', 'fp16_tflops', 'fp64_tflops']
    flops_labels = ['Tensor', 'FP32', 'FP16', 'FP64']
    x_flops = np.arange(len(test_names))
    bar_w = 0.2
    
    for j, (ft, fl) in enumerate(zip(flops_types, flops_labels)):
        avgs = []
        for test_name in test_names:
            all_gpu_avg = np.mean([np.mean(all_results[test_name]['gpu_stats'].get(gid, {}).get(ft, [0])) for gid in gpu_ids])
            avgs.append(all_gpu_avg)
        ax5.bar(x_flops + j * bar_w, avgs, bar_w, label=fl)
    
    ax5.set_xlabel('Test')
    ax5.set_ylabel('TFLOPS')
    ax5.set_title('Average TFLOPS by Test (All GPU)')
    ax5.set_xticks(x_flops + 1.5 * bar_w)
    ax5.set_xticklabels(test_names)
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. 系統功耗比較
    ax6 = axes[2, 1]
    sys_power_avg = []
    sys_power_max = []
    for test_name in test_names:
        sp = all_results[test_name]['system_stats']['system_powers']
        valid_sp = [p for p in sp if p > 0]
        sys_power_avg.append(np.mean(valid_sp) if valid_sp else 0)
        sys_power_max.append(np.max(valid_sp) if valid_sp else 0)
    
    x_sys = np.arange(len(test_names))
    ax6.bar(x_sys - 0.15, sys_power_avg, 0.3, label='Average', color='steelblue')
    ax6.bar(x_sys + 0.15, sys_power_max, 0.3, label='Peak', color='coral')
    ax6.set_xlabel('Test')
    ax6.set_ylabel('Power (W)')
    ax6.set_title('System Power by Test')
    ax6.set_xticks(x_sys)
    ax6.set_xticklabels(test_names)
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(chart_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ 比較圖表已保存: {chart_file}")
    
    # 生成每個測試的時序圖
    for test_name, results in all_results.items():
        generate_test_timeline(test_name, results, output_prefix)


def generate_test_timeline(test_name, results, output_prefix):
    """生成單一測試的時序圖"""
    
    chart_file = f"{output_prefix}_{test_name}_timeline.png"
    
    gpu_stats = results['gpu_stats']
    system_stats = results['system_stats']
    gpu_ids = sorted(gpu_stats.keys())
    
    if not gpu_ids or not gpu_stats[gpu_ids[0]]['temps']:
        return
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 14))
    fig.suptitle(f'{test_name} Test Timeline', fontsize=14, fontweight='bold')
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(gpu_ids)))
    samples = len(gpu_stats[gpu_ids[0]]['temps'])
    x = np.arange(samples)
    
    # 1. GPU 溫度
    ax1 = axes[0, 0]
    for i, gpu_id in enumerate(gpu_ids):
        ax1.plot(x, gpu_stats[gpu_id]['temps'], label=f'GPU {gpu_id}', color=colors[i], linewidth=1)
    temp_limit = results['temp_limits'][0] if results['temp_limits'] else 83
    ax1.axhline(y=temp_limit, color='red', linestyle='--', label='Limit', alpha=0.7)
    ax1.set_xlabel('Sample')
    ax1.set_ylabel('Temperature (°C)')
    ax1.set_title('GPU Temperature')
    ax1.legend(loc='upper right', fontsize=7, ncol=2)
    ax1.grid(True, alpha=0.3)
    
    # 2. GPU 功耗
    ax2 = axes[0, 1]
    for i, gpu_id in enumerate(gpu_ids):
        ax2.plot(x, gpu_stats[gpu_id]['powers'], label=f'GPU {gpu_id}', color=colors[i], linewidth=1)
    power_limit = results['power_limits'][0] if results['power_limits'] else 1000
    ax2.axhline(y=power_limit, color='red', linestyle='--', label='Limit', alpha=0.7)
    ax2.set_xlabel('Sample')
    ax2.set_ylabel('Power (W)')
    ax2.set_title('GPU Power')
    ax2.legend(loc='upper right', fontsize=7, ncol=2)
    ax2.grid(True, alpha=0.3)
    
    # 3. Tensor Core TFLOPS
    ax3 = axes[1, 0]
    for i, gpu_id in enumerate(gpu_ids):
        ax3.plot(x, gpu_stats[gpu_id]['tensor_tflops'], label=f'GPU {gpu_id}', color=colors[i], linewidth=1)
    ax3.set_xlabel('Sample')
    ax3.set_ylabel('TFLOPS')
    ax3.set_title('Tensor Core Performance')
    ax3.legend(loc='upper right', fontsize=7, ncol=2)
    ax3.grid(True, alpha=0.3)
    
    # 4. FP32 / FP16 / FP64 TFLOPS
    ax4 = axes[1, 1]
    # 計算所有 GPU 的平均值
    fp32_avg = [np.mean([gpu_stats[gid]['fp32_tflops'][i] for gid in gpu_ids if i < len(gpu_stats[gid]['fp32_tflops'])]) for i in range(samples)]
    fp16_avg = [np.mean([gpu_stats[gid]['fp16_tflops'][i] for gid in gpu_ids if i < len(gpu_stats[gid]['fp16_tflops'])]) for i in range(samples)]
    fp64_avg = [np.mean([gpu_stats[gid]['fp64_tflops'][i] for gid in gpu_ids if i < len(gpu_stats[gid]['fp64_tflops'])]) for i in range(samples)]
    ax4.plot(x, fp32_avg, label='FP32 Avg', color='blue', linewidth=1.5)
    ax4.plot(x, fp16_avg, label='FP16 Avg', color='green', linewidth=1.5)
    ax4.plot(x, fp64_avg, label='FP64 Avg', color='orange', linewidth=1.5)
    ax4.set_xlabel('Sample')
    ax4.set_ylabel('TFLOPS')
    ax4.set_title('FP32 / FP16 / FP64 Performance (All GPU Avg)')
    ax4.legend(loc='upper right', fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    # 5. 系統溫度
    ax5 = axes[2, 0]
    x_sys = np.arange(len(system_stats['inlet_temps']))
    if any(t > 0 for t in system_stats['inlet_temps']):
        ax5.plot(x_sys, system_stats['inlet_temps'], label='Inlet', color='blue', linewidth=1.5)
    if any(t > 0 for t in system_stats['outlet_temps']):
        ax5.plot(x_sys, system_stats['outlet_temps'], label='Outlet', color='red', linewidth=1.5)
    if any(t > 0 for t in system_stats['cpu_temps']):
        ax5.plot(x_sys, system_stats['cpu_temps'], label='CPU', color='green', linewidth=1.5)
    ax5.set_xlabel('Sample')
    ax5.set_ylabel('Temperature (°C)')
    ax5.set_title('System Temperature')
    ax5.legend(loc='upper right', fontsize=8)
    ax5.grid(True, alpha=0.3)
    
    # 6. 系統功耗 vs GPU 總功耗
    ax6 = axes[2, 1]
    if any(p > 0 for p in system_stats['system_powers']):
        ax6.plot(x_sys, system_stats['system_powers'], label='System Power', color='purple', linewidth=1.5)
    total_gpu = [sum(gpu_stats[gid]['powers'][i] for gid in gpu_ids if i < len(gpu_stats[gid]['powers'])) for i in range(samples)]
    ax6.plot(x[:len(total_gpu)], total_gpu, label='Total GPU Power', color='orange', linewidth=1.5, linestyle='--')
    ax6.set_xlabel('Sample')
    ax6.set_ylabel('Power (W)')
    ax6.set_title('System vs GPU Power')
    ax6.legend(loc='upper right', fontsize=8)
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(chart_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ 時序圖已保存: {chart_file}")


def main():
    print("=" * 70)
    print("DCGM 壓力測試工具")
    print("=" * 70)
    
    # 檢查 dcgmproftester 路徑
    if not os.path.exists(DCGM_PROFTESTER_PATH):
        print(f"錯誤: 找不到 {DCGM_PROFTESTER_PATH}")
        sys.exit(1)
    
    # 取得 GPU 資訊
    gpus = get_gpu_info()
    if not gpus:
        print("錯誤: 找不到 GPU")
        sys.exit(1)
    
    gpu_list = [g['id'] for g in gpus]
    
    print(f"\n偵測到 {len(gpus)} 張 GPU:")
    for g in gpus:
        print(f"  GPU {g['id']}: {g['name']} (Power Limit: {g['power_limit']}W)")
    
    print(f"\n測試設定:")
    print(f"  測試時間: {TEST_DURATION} 秒/每項測試")
    print(f"  測試項目: {[t['name'] for t in TESTS]}")
    print(f"  總預估時間: {len(TESTS) * TEST_DURATION / 60:.1f} 分鐘")
    
    # 建立輸出目錄
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_prefix = f"{OUTPUT_DIR}/dcgm_test_{timestamp}"
    
    all_results = {}
    
    print("\n" + "=" * 70)
    print("開始測試")
    print("=" * 70)
    
    for test in TESTS:
        test_id = test['id']
        test_name = test['name']
        
        print(f"\n>>> 測試 {test_name} (ID: {test_id}) - {TEST_DURATION} 秒")
        print("-" * 50)
        
        # 重置停止信號
        stop_monitor.clear()
        
        # 啟動監控執行緒
        monitor = Thread(target=monitor_thread, args=(test_name, TEST_DURATION, output_prefix, all_results))
        monitor.start()
        
        # 等待監控開始
        time.sleep(2)
        
        # 啟動壓力測試
        process = run_stress_test(test_id, test_name, TEST_DURATION, gpu_list)
        
        if process:
            try:
                # 等待測試完成或超時
                process.wait(timeout=TEST_DURATION + 60)
            except subprocess.TimeoutExpired:
                print(f"\n警告: 測試超時，強制終止...")
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                time.sleep(5)
                if process.poll() is None:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except Exception as e:
                print(f"\n測試過程錯誤: {e}")
        
        # 停止監控
        stop_monitor.set()
        monitor.join(timeout=10)
        
        print(f"\n✓ {test_name} 測試完成")
        
        # 測試間休息
        if test != TESTS[-1]:
            print("\n休息 10 秒後進行下一項測試...")
            time.sleep(10)
    
    # 生成摘要報告和圖表
    generate_summary_and_charts(all_results, output_prefix)
    
    print("\n" + "=" * 70)
    print("所有測試完成!")
    print(f"結果保存在: {OUTPUT_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
