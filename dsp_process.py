import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import re

# 配置参数
DATA_DIR = '槽楔模型测试数据'
SAMPLE_RATE = 100000  # 假设采样率，如果未知，结果将是归一化频率或需要用户提供
# 注意：如果不知道采样率，算出来的频率只能用于相对比较，绝对值没有物理意义（Hz）。
# 通常这种实验采样率可能在 10kHz - 100kHz 之间。这里先假设 100kHz 方便计算，
# 实际上我们主要关注不同压力下频率的相对变化。

WINDOW_SIZE = 4096     # FFT 窗口大小 (2^12)
STEP_SIZE = 2000       # 峰值检测最小间隔

def parse_pressure(filename):
    """从文件名提取压力值"""
    match = re.search(r'acquisitionData-(\d+)', filename)
    if match:
        return int(match.group(1))
    return None

def segment_signal(data, window_size=WINDOW_SIZE):
    """
    自动切分信号，提取每次敲击的数据段
    """
    # 1. 移除直流分量
    data = data - np.mean(data)
    
    # 2. 计算能量或包络来检测敲击
    # 简单起见，使用绝对值阈值
    abs_data = np.abs(data)
    threshold = np.max(abs_data) * 0.15 # 动态阈值
    
    peaks = []
    last_peak = -STEP_SIZE
    
    segments = []
    
    for i in range(len(data)):
        if abs_data[i] > threshold:
            if i - last_peak > STEP_SIZE:
                # 找到一个新的敲击事件
                # 我们希望截取峰值附近的信号。
                # 简单的做法：以触发点为起点，或者往回找一点
                start_idx = i
                
                # 检查是否有足够的数据
                if start_idx + window_size <= len(data):
                    segment = data[start_idx : start_idx + window_size]
                    segments.append(segment)
                    peaks.append(start_idx)
                    last_peak = start_idx
                
    return segments, peaks

def compute_fft(segment, fs=SAMPLE_RATE):
    """
    计算信号的FFT
    """
    N = len(segment)
    # 加窗 (Hanning window) 减少频谱泄漏
    windowed_segment = segment * np.hanning(N)
    
    fft_val = np.fft.fft(windowed_segment)
    fft_freq = np.fft.fftfreq(N, d=1/fs)
    
    # 只取正半轴
    positive_freqs = fft_freq[:N//2]
    magnitude = np.abs(fft_val)[:N//2]
    
    return positive_freqs, magnitude

def extract_features(freqs, magnitude):
    """
    提取特征：主频 (最大幅值对应的频率)
    """
    # 忽略低频噪音 (比如 50Hz 工频干扰或直流偏移)
    # 假设有效信号在 500Hz 以上 (根据实际情况调整)
    valid_mask = freqs > 100 
    
    if np.sum(valid_mask) == 0:
        return 0, 0
        
    valid_freqs = freqs[valid_mask]
    valid_mag = magnitude[valid_mask]
    
    peak_idx = np.argmax(valid_mag)
    peak_freq = valid_freqs[peak_idx]
    peak_mag = valid_mag[peak_idx]
    
    # 计算质心频率 (Spectral Centroid) 作为辅助特征
    centroid_freq = np.sum(valid_freqs * valid_mag) / np.sum(valid_mag)
    
    return peak_freq, centroid_freq

def main():
    # 查找所有数据文件
    file_pattern = os.path.join(DATA_DIR, 'acquisitionData-*.txt')
    files = glob.glob(file_pattern)
    
    results = []
    
    print(f"找到 {len(files)} 个数据文件。开始处理...")
    
    # 为了绘图对比，创建一个图
    plt.figure(figsize=(15, 10))
    
    for file_path in files:
        filename = os.path.basename(file_path)
        pressure = parse_pressure(filename)
        if pressure is None:
            continue
            
        print(f"正在处理: {filename} (压力: {pressure})")
        
        try:
            raw_data = np.loadtxt(file_path)
            segments, _ = segment_signal(raw_data)
            
            if not segments:
                print(f"  警告: 未检测到有效敲击信号")
                continue
            
            print(f"  检测到 {len(segments)} 次敲击")
            
            # 对每个片段做FFT并取平均
            avg_magnitude = np.zeros(WINDOW_SIZE//2)
            freqs = None
            
            peak_freqs = []
            
            for seg in segments:
                f, mag = compute_fft(seg)
                avg_magnitude += mag
                if freqs is None:
                    freqs = f
                
                # 也可以对每个单独片段提取特征然后取平均
                pf, _ = extract_features(f, mag)
                peak_freqs.append(pf)
            
            avg_magnitude /= len(segments)
            
            # 提取平均谱的特征
            final_peak_freq, final_centroid = extract_features(freqs, avg_magnitude)
            
            # 或者使用所有片段特征的平均值 (通常更稳健)
            mean_peak_freq = np.mean(peak_freqs)
            
            results.append({
                'pressure': pressure,
                'peak_freq': mean_peak_freq,
                'centroid_freq': final_centroid,
                'spectrum': avg_magnitude,
                'freqs': freqs
            })
            
            # 绘制归一化的频谱图 (选几个典型的画)
            if pressure in [400, 2000, 4000]:
                plt.plot(freqs, avg_magnitude / np.max(avg_magnitude), label=f'Pressure {pressure}')
                
        except Exception as e:
            print(f"  处理出错: {e}")

    # 排序结果
    results.sort(key=lambda x: x['pressure'])
    
    # 绘制频谱对比图
    plt.title("Frequency Spectrum Comparison (Normalized)")
    plt.xlabel("Frequency (Hz) [Assumed Fs=100kHz]")
    plt.ylabel("Normalized Amplitude")
    plt.xlim(0, 20000) # 只看前20k
    plt.legend()
    plt.grid(True)
    plt.savefig('spectrum_comparison.png')
    
    # 绘制 压力 vs 频率 关系图
    pressures = [r['pressure'] for r in results]
    peak_freqs = [r['peak_freq'] for r in results]
    centroid_freqs = [r['centroid_freq'] for r in results]
    
    plt.figure(figsize=(10, 6))
    plt.plot(pressures, peak_freqs, 'o-', label='Peak Frequency')
    plt.plot(pressures, centroid_freqs, 's--', label='Centroid Frequency')
    plt.title("Pressure vs Frequency Feature")
    plt.xlabel("Pressure (Tightness)")
    plt.ylabel("Frequency Feature (Hz)")
    plt.grid(True)
    plt.legend()
    plt.savefig('pressure_vs_frequency.png')
    
    print("\n处理完成！")
    print("生成了以下图表:")
    print("1. spectrum_comparison.png (不同压力下的频谱对比)")
    print("2. pressure_vs_frequency.png (松紧度与频率的关系)")
    
    print("\n数据摘要:")
    print(f"{'Pressure':<10} | {'Peak Freq':<15} | {'Centroid Freq':<15}")
    print("-" * 45)
    for r in results:
        print(f"{r['pressure']:<10} | {r['peak_freq']:<15.2f} | {r['centroid_freq']:<15.2f}")

if __name__ == "__main__":
    main()
