#!/usr/bin/env python3
"""
DCGM Tensor Core 壓力測試腳本
- 執行 dcgmproftester 1004 (Tensor Core) 測試
- 即時解析 TFLOPS 數據
- 同時收集 GPU 溫度、功耗、系統數據
用法: sudo python dcgm_tensor_test.py [測試秒數]
範例: sudo python dcgm_tensor_test.py 300
"""

import subprocess
import time
import csv
import sys
import os
import re
import signal
from datetime import datetime
from collections import defaultdict
from threading import Thread, Event, Lock

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
TEST_DURATION = 300  # 預設測試時間（秒）
SAMPLE_INTERVAL = 1  # 採樣間隔（秒）
OUTPUT_DIR = "dcgm_results"
# ================================================

# 全域變數
stop_event = Event()
flops_lock = Lock()
gpu_flops_data = defaultdict(list)  # {gpu_id: [(timestamp, gflops), ...]}


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
    except:
        return []


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


def parse_flops_from_line(line):
    """從 dcgmproftester 輸出解析 GFLOPS 數據
    格式: Worker X:0 [1004]: TensorEngineActive: ... (X.XXe+XX gflops)
    """
    # 匹配 Worker ID 和 GFLOPS
    pattern = r'Worker (\d+):\d+ \[1004\]:.*\((\d+\.?\d*e[+\-]?\d+) gflops\)'
    match = re.search(pattern, line)
    
    if match:
        gpu_id = int(match.group(1))
        gflops = float(match.group(2))
        return gpu_id, gflops
    
    return None, None


def dcgm_output_reader(process, log_handle=None):
    """讀取 dcgmproftester 的輸出並解析 FLOPS，同時寫入 log"""
    global gpu_flops_data
    
    while not stop_event.is_set() and process.poll() is None:
        try:
            line = process.stdout.readline()
            if line:
                line_str = line.decode('utf-8', errors='ignore').strip()
                
                # 寫入 log 檔案
                if log_handle:
                    try:
                        log_handle.write(line_str + '\n')
                        log_handle.flush()
                    except:
                        pass
                
                # 解析 FLOPS 數據
                gpu_id, gflops = parse_flops_from_line(line_str)
                if gpu_id is not None:
                    timestamp = datetime.now()
                    with flops_lock:
                        gpu_flops_data[gpu_id].append((timestamp, gflops))
        except:
            pass


def get_latest_flops():
    """取得每個 GPU 最新的 FLOPS 數據"""
    latest = {}
    with flops_lock:
        for gpu_id, data_list in gpu_flops_data.items():
            if data_list:
                # 取最近 5 秒內的數據平均
                now = datetime.now()
                recent = [g for t, g in data_list if (now - t).total_seconds() < 5]
                if recent:
                    latest[gpu_id] = np.mean(recent)
    return latest


def monitor_and_collect(duration, output_prefix):
    """主監控函數"""
    
    gpus = get_gpu_info()
    gpu_list = [g['id'] for g in gpus]
    gpu_str = ",".join(map(str, gpu_list))
    temp_limits = get_temp_limit()
    power_limits = [g['power_limit'] for g in gpus]
    
    print("=" * 70)
    print("DCGM Tensor Core 壓力測試")
    print("=" * 70)
    print(f"\n偵測到 {len(gpus)} 張 GPU:")
    for g in gpus:
        print(f"  GPU {g['id']}: {g['name']} (Power Limit: {g['power_limit']}W)")
    print(f"\n測試時間: {duration} 秒")
    print("=" * 70)
    
    # 啟動 dcgmproftester
    cmd = [
        DCGM_PROFTESTER_PATH,
        "--no-dcgm-validation",
        "-t", "1004",  # Tensor Core 測試
        "-d", str(duration),
        "-i", gpu_str
    ]
    
    print(f"\n執行命令: {' '.join(cmd)}")
    print("\n開始測試... (Ctrl+C 可提前結束)\n")
    
    # Log 檔案路徑
    log_file = f"{output_prefix}_dcgmproftester.log"
    log_handle = open(log_file, 'w', encoding='utf-8')
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # 合併 stderr 到 stdout
        preexec_fn=os.setsid
    )
    
    # 啟動輸出讀取執行緒
    reader_thread = Thread(target=dcgm_output_reader, args=(process, log_handle))
    reader_thread.daemon = True
    reader_thread.start()
    
    # 資料儲存
    all_data = []
    gpu_stats = defaultdict(lambda: {
        'temps': [], 'powers': [], 'utils': [], 'tflops': []
    })
    system_stats = {
        'inlet_temps': [], 'outlet_temps': [], 'cpu_temps': [], 'system_powers': []
    }
    
    start_time = time.time()
    
    print(f"{'Time':<8} {'GPU':<4} {'Temp':<8} {'Power':<10} {'TFLOPS':<12} | {'Sys Pwr':<10} {'Inlet':<8}")
    print("-" * 80)
    
    try:
        while process.poll() is None and (time.time() - start_time) < duration + 30:
            timestamp = datetime.now()
            elapsed = time.time() - start_time
            
            # 收集數據
            gpu_data = collect_gpu_data()
            sys_data = get_system_sensors()
            latest_flops = get_latest_flops()
            
            # 記錄系統數據
            system_stats['inlet_temps'].append(sys_data['inlet_temp'])
            system_stats['outlet_temps'].append(sys_data['outlet_temp'])
            system_stats['cpu_temps'].append(sys_data['cpu_temp'])
            system_stats['system_powers'].append(sys_data['system_power'])
            
            for gpu in gpu_data:
                gpu_id = gpu['gpu_id']
                tflops = latest_flops.get(gpu_id, 0) / 1000  # GFLOPS -> TFLOPS
                
                record = {
                    'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                    'elapsed_sec': round(elapsed, 1),
                    'gpu_id': gpu_id,
                    'gpu_temp_c': gpu['temp'],
                    'gpu_temp_limit_c': temp_limits[gpu_id] if gpu_id < len(temp_limits) else 83,
                    'gpu_power_w': gpu['power'],
                    'gpu_power_limit_w': power_limits[gpu_id] if gpu_id < len(power_limits) else 1000,
                    'gpu_util_pct': gpu['util'],
                    'gpu_mem_used_mb': gpu['mem_used'],
                    'tensor_tflops': round(tflops, 2),
                    'sys_inlet_temp_c': sys_data['inlet_temp'],
                    'sys_outlet_temp_c': sys_data['outlet_temp'],
                    'sys_cpu_temp_c': sys_data['cpu_temp'],
                    'sys_power_w': sys_data['system_power'],
                }
                all_data.append(record)
                
                # 統計 (只計算有在運行的 GPU，power > 500W)
                if gpu['power'] > 500:
                    gpu_stats[gpu_id]['temps'].append(gpu['temp'])
                    gpu_stats[gpu_id]['powers'].append(gpu['power'])
                    gpu_stats[gpu_id]['utils'].append(gpu['util'])
                    if tflops > 0:
                        gpu_stats[gpu_id]['tflops'].append(tflops)
            
            # 即時顯示
            if gpu_data:
                # 只顯示活躍的 GPU
                active_gpus = [g for g in gpu_data if g['power'] > 500]
                if active_gpus:
                    for g in active_gpus[:2]:  # 只顯示前兩張
                        tflops = latest_flops.get(g['gpu_id'], 0) / 1000
                        print(f"\r{elapsed:>6.0f}s  {g['gpu_id']:<4} {g['temp']:<8.1f} {g['power']:<10.1f} {tflops:<12.2f} | {sys_data['system_power']:<10.0f} {sys_data['inlet_temp']:<8.1f}", end='')
            
            time.sleep(SAMPLE_INTERVAL)
            
    except KeyboardInterrupt:
        print("\n\n測試被中斷...")
    finally:
        stop_event.set()
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            time.sleep(2)
            if process.poll() is None:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except:
            pass
        
        # 關閉 log 檔案
        try:
            log_handle.close()
            print(f"\n✓ dcgmproftester log 已保存: {log_file}")
        except:
            pass
    
    print("\n\n" + "=" * 70)
    print("測試結束，正在生成報告...")
    print("=" * 70)
    
    # 保存 CSV
    csv_file = f"{output_prefix}_data.csv"
    if all_data:
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_data[0].keys())
            writer.writeheader()
            writer.writerows(all_data)
        print(f"\n✓ CSV 已保存: {csv_file}")
    
    # 生成摘要
    generate_summary(gpu_stats, system_stats, temp_limits, power_limits, output_prefix)
    
    # 生成圖表
    generate_charts(gpu_stats, system_stats, temp_limits, power_limits, output_prefix)
    
    return all_data, gpu_stats, system_stats


def generate_summary(gpu_stats, system_stats, temp_limits, power_limits, output_prefix):
    """生成統計摘要"""
    
    summary_file = f"{output_prefix}_summary.csv"
    summary_data = []
    
    print("\n" + "=" * 70)
    print("測試摘要 (只計算活躍 GPU)")
    print("=" * 70)
    
    active_gpus = [gid for gid, stats in gpu_stats.items() if stats['temps']]
    
    for gpu_id in sorted(active_gpus):
        stats = gpu_stats[gpu_id]
        
        temp_limit = temp_limits[gpu_id] if gpu_id < len(temp_limits) else 83
        power_limit = power_limits[gpu_id] if gpu_id < len(power_limits) else 1000
        
        row = {
            'gpu_id': gpu_id,
            'temp_avg': np.mean(stats['temps']),
            'temp_max': np.max(stats['temps']),
            'temp_limit': temp_limit,
            'power_avg': np.mean(stats['powers']),
            'power_max': np.max(stats['powers']),
            'power_limit': power_limit,
            'util_avg': np.mean(stats['utils']),
            'tflops_avg': np.mean(stats['tflops']) if stats['tflops'] else 0,
            'tflops_max': np.max(stats['tflops']) if stats['tflops'] else 0,
        }
        summary_data.append(row)
        
        print(f"\nGPU {gpu_id}:")
        print(f"  溫度: 平均 {row['temp_avg']:.1f}°C | 最高 {row['temp_max']:.1f}°C | 上限 {temp_limit}°C")
        print(f"  功耗: 平均 {row['power_avg']:.1f}W | 最高 {row['power_max']:.1f}W | 上限 {power_limit}W")
        print(f"  TFLOPS: 平均 {row['tflops_avg']:.2f} | 最高 {row['tflops_max']:.2f}")
    
    # 整體統計
    if active_gpus:
        print("\n" + "-" * 50)
        print("整體統計 (活躍 GPU):")
        all_temps = [t for gid in active_gpus for t in gpu_stats[gid]['temps']]
        all_powers = [p for gid in active_gpus for p in gpu_stats[gid]['powers']]
        all_tflops = [t for gid in active_gpus for t in gpu_stats[gid]['tflops']]
        
        print(f"  平均溫度: {np.mean(all_temps):.1f}°C | 最高: {np.max(all_temps):.1f}°C")
        print(f"  平均功耗: {np.mean(all_powers):.1f}W | 最高: {np.max(all_powers):.1f}W")
        print(f"  平均 TFLOPS: {np.mean(all_tflops):.2f} | 最高: {np.max(all_tflops):.2f}")
        print(f"  總 GPU 功耗 (估算): {np.mean(all_powers) * len(active_gpus):.1f}W")
    
    # 系統統計
    print("\n" + "-" * 50)
    print("系統統計:")
    valid_inlet = [t for t in system_stats['inlet_temps'] if t > 0]
    valid_power = [p for p in system_stats['system_powers'] if p > 0]
    
    if valid_inlet:
        print(f"  進氣溫度: 平均 {np.mean(valid_inlet):.1f}°C | 最高 {np.max(valid_inlet):.1f}°C")
    if valid_power:
        print(f"  系統功耗: 平均 {np.mean(valid_power):.0f}W | 最高 {np.max(valid_power):.0f}W")
    
    # 保存摘要 CSV
    if summary_data:
        with open(summary_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=summary_data[0].keys())
            writer.writeheader()
            writer.writerows(summary_data)
        print(f"\n✓ 摘要 CSV 已保存: {summary_file}")


def generate_charts(gpu_stats, system_stats, temp_limits, power_limits, output_prefix):
    """生成圖表"""
    
    chart_file = f"{output_prefix}_charts.png"
    
    active_gpus = [gid for gid, stats in gpu_stats.items() if stats['temps']]
    if not active_gpus:
        print("沒有活躍 GPU 數據，跳過圖表生成")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('DCGM Tensor Core Test Results', fontsize=14, fontweight='bold')
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(active_gpus)))
    
    # 1. GPU 溫度時序圖
    ax1 = axes[0, 0]
    for i, gpu_id in enumerate(sorted(active_gpus)):
        temps = gpu_stats[gpu_id]['temps']
        ax1.plot(range(len(temps)), temps, label=f'GPU {gpu_id}', color=colors[i], linewidth=1)
    if temp_limits:
        ax1.axhline(y=temp_limits[0], color='red', linestyle='--', label='Limit', alpha=0.7)
    ax1.set_xlabel('Sample')
    ax1.set_ylabel('Temperature (°C)')
    ax1.set_title('GPU Temperature')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # 2. GPU 功耗時序圖
    ax2 = axes[0, 1]
    for i, gpu_id in enumerate(sorted(active_gpus)):
        powers = gpu_stats[gpu_id]['powers']
        ax2.plot(range(len(powers)), powers, label=f'GPU {gpu_id}', color=colors[i], linewidth=1)
    if power_limits:
        ax2.axhline(y=power_limits[0], color='red', linestyle='--', label='Limit', alpha=0.7)
    ax2.set_xlabel('Sample')
    ax2.set_ylabel('Power (W)')
    ax2.set_title('GPU Power')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # 3. TFLOPS 時序圖
    ax3 = axes[1, 0]
    for i, gpu_id in enumerate(sorted(active_gpus)):
        tflops = gpu_stats[gpu_id]['tflops']
        if tflops:
            ax3.plot(range(len(tflops)), tflops, label=f'GPU {gpu_id}', color=colors[i], linewidth=1)
    ax3.set_xlabel('Sample')
    ax3.set_ylabel('TFLOPS')
    ax3.set_title('Tensor Core Performance')
    ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # 4. 統計長條圖
    ax4 = axes[1, 1]
    x = np.arange(len(active_gpus))
    bar_width = 0.35
    
    temp_avgs = [np.mean(gpu_stats[gid]['temps']) for gid in sorted(active_gpus)]
    temp_maxs = [np.max(gpu_stats[gid]['temps']) for gid in sorted(active_gpus)]
    
    ax4_twin = ax4.twinx()
    
    bars1 = ax4.bar(x - bar_width/2, temp_avgs, bar_width, label='Temp Avg', color='steelblue')
    bars2 = ax4.bar(x + bar_width/2, temp_maxs, bar_width, label='Temp Max', color='coral')
    
    tflops_avgs = [np.mean(gpu_stats[gid]['tflops']) if gpu_stats[gid]['tflops'] else 0 for gid in sorted(active_gpus)]
    ax4_twin.plot(x, tflops_avgs, 'go-', label='TFLOPS Avg', linewidth=2, markersize=8)
    
    ax4.set_xlabel('GPU')
    ax4.set_ylabel('Temperature (°C)')
    ax4_twin.set_ylabel('TFLOPS')
    ax4.set_title('GPU Statistics')
    ax4.set_xticks(x)
    ax4.set_xticklabels([f'GPU {gid}' for gid in sorted(active_gpus)])
    ax4.legend(loc='upper left', fontsize=8)
    ax4_twin.legend(loc='upper right', fontsize=8)
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(chart_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ 圖表已保存: {chart_file}")


def main():
    global TEST_DURATION
    
    # 解析命令列參數
    if len(sys.argv) > 1:
        try:
            TEST_DURATION = int(sys.argv[1])
        except:
            pass
    
    # 檢查 dcgmproftester 路徑
    if not os.path.exists(DCGM_PROFTESTER_PATH):
        print(f"錯誤: 找不到 {DCGM_PROFTESTER_PATH}")
        print("請確認 DCGM 已安裝，或修改腳本中的 DCGM_PROFTESTER_PATH")
        sys.exit(1)
    
    # 建立輸出目錄
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_prefix = f"{OUTPUT_DIR}/tensor_test_{timestamp}"
    
    # 執行測試
    monitor_and_collect(TEST_DURATION, output_prefix)
    
    print("\n" + "=" * 70)
    print("測試完成!")
    print(f"結果保存在: {OUTPUT_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
