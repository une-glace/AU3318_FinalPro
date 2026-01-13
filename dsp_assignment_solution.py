import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import re
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
from scipy.stats import skew, kurtosis

# ================= 配置参数 =================
DATA_DIR = '槽楔模型测试数据'
SAMPLE_RATE = 100000   # 假设采样率 100kHz
WINDOW_SIZE = 4096     # 4096点 FFT
STEP_SIZE = 2000       # 最小峰值间隔
# ===========================================

def parse_pressure(filename):
    match = re.search(r'acquisitionData-(\d+)', filename)
    if match:
        return int(match.group(1))
    return None

def segment_signal(data, window_size=WINDOW_SIZE):
    # 去直流
    data = data - np.mean(data)
    # 取绝对值
    abs_data = np.abs(data)
    # 动态阈值
    threshold = np.max(abs_data) * 0.15
    
    segments = []
    last_peak = -STEP_SIZE
    
    for i in range(len(data)):
        if abs_data[i] > threshold:
            if i - last_peak > STEP_SIZE:
                start_idx = i
                if start_idx + window_size <= len(data):
                    segment = data[start_idx : start_idx + window_size]
                    segments.append(segment)
                    last_peak = start_idx
    return segments

def compute_fft(segment, fs=SAMPLE_RATE):
    N = len(segment)
    window = np.hanning(N)
    segment_w = segment * window
    
    fft_val = np.abs(np.fft.fft(segment_w))[:N//2]
    fft_freq = np.fft.fftfreq(N, d=1/fs)[:N//2]
    
    return fft_freq, fft_val

def extract_features(freqs, magnitude, segment_time_domain):
    """
    提取多个特征用于分类 (增强版)
    输入:
      freqs: 频率轴
      magnitude: 幅值谱
      segment_time_domain: 时域信号片段
    """
    # ================= 频域特征 =================
    # 1. 忽略低频干扰 (例如 <1500Hz 的结构模态/台架共振)
    # 观察发现高压下 1000Hz 左右有强干扰峰，掩盖了真实的高频特征
    valid_mask = freqs > 1500
    if np.sum(valid_mask) == 0:
        return [0] * 8 # 返回8个0
        
    f = freqs[valid_mask]
    m = magnitude[valid_mask]
    
    # 特征1: 最大峰值频率 (Dominant Frequency)
    idx_max = np.argmax(m)
    peak_freq = f[idx_max]
    
    # 特征2: 频谱质心 (Spectral Centroid)
    centroid = np.sum(f * m) / (np.sum(m) + 1e-6)
    
    # 特征3: 带宽 (Spectral Bandwidth)
    bandwidth = np.sqrt(np.sum(((f - centroid)**2) * m) / (np.sum(m) + 1e-6))
    
    # 特征4: 能量 (Energy)
    energy = np.sum(m**2)

    # 特征5: 偏度 (Spectral Skewness) - 描述频谱分布的对称性
    # 简单的加权计算
    spec_skewness = np.sum(((f - centroid)**3) * m) / (np.sum(m) * bandwidth**3 + 1e-6)
    
    # ================= 时域特征 =================
    # 特征6: 峰度 (Kurtosis) - 描述时域波形的尖锐程度
    # 敲击声通常非常尖锐，峰度很高
    time_kurtosis = kurtosis(segment_time_domain)
    
    # 特征7: 偏度 (Skewness) - 描述时域波形的对称性
    time_skewness = skew(segment_time_domain)
    
    # 特征8: 均方根值 (RMS) - 描述时域信号的有效强度
    rms = np.sqrt(np.mean(segment_time_domain**2))
    
    return [peak_freq, centroid, bandwidth, energy, spec_skewness, time_kurtosis, time_skewness, rms]

def main():
    print("=== 开始处理 DSP 大作业任务 (增强特征版) ===")
    
    files = glob.glob(os.path.join(DATA_DIR, 'acquisitionData-*.txt'))
    
    X = [] # 特征矩阵
    y = [] # 标签 (压力值)
    
    # 用于统计每个压力下的平均特征
    pressure_stats = {}
    
    # 用于绘制频谱对比图的存储 (Pressure -> (freqs, avg_magnitude))
    spectra_for_plot = {}
    target_pressures_for_plot = [400, 2000, 4000]

    for file_path in files:
        filename = os.path.basename(file_path)
        pressure = parse_pressure(filename)
        if pressure is None: continue
        
        print(f"处理文件: {filename} (Label: {pressure})")
        
        try:
            raw_data = np.loadtxt(file_path)
            segments = segment_signal(raw_data)
            print(f"  -> 提取到 {len(segments)} 个样本")
            
            # 如果是目标压力，初始化平均谱计算
            if pressure in target_pressures_for_plot:
                accum_mag = np.zeros(WINDOW_SIZE//2)
                count_seg = 0
                freqs_ref = None

            for seg in segments:
                freqs, mag = compute_fft(seg)
                
                # 累加频谱用于绘图
                if pressure in target_pressures_for_plot:
                    accum_mag += mag
                    count_seg += 1
                    if freqs_ref is None: freqs_ref = freqs

                # 传入时域信号 segment 用于计算时域特征
                feats = extract_features(freqs, mag, seg)
                
                X.append(feats)
                y.append(pressure)
                
                # 累加用于统计
                if pressure not in pressure_stats:
                    pressure_stats[pressure] = []
                pressure_stats[pressure].append(feats)
            
            # 存储该压力的平均谱
            if pressure in target_pressures_for_plot and count_seg > 0:
                avg_mag = accum_mag / count_seg
                spectra_for_plot[pressure] = (freqs_ref, avg_mag)
                
        except Exception as e:
            print(f"  -> 错误: {e}")
            
    X = np.array(X)
    y = np.array(y)
    
    print(f"\n数据集构建完成: 总样本数 {len(X)}, 特征数 {X.shape[1]}")
    print("特征列表: [Peak Freq, Centroid, Bandwidth, Energy, Spec Skew, Time Kurt, Time Skew, RMS]")
    
    # ================= 6. 分类与精度计算 =================
    print("\n--- 开始分类训练 (使用 KNN 和 随机森林) ---")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 1. KNN
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    acc_knn = accuracy_score(y_test, y_pred_knn)
    print(f"KNN 分类准确率: {acc_knn:.4f}")
    
    # 2. Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    print(f"随机森林 分类准确率: {acc_rf:.4f}")
    
    print("\n随机森林分类报告:")
    print(classification_report(y_test, y_pred_rf))
    
    # ================= 8. 绘制 松紧度 vs 频率 关系 =================
    print("\n--- 绘制分析图表 ---")
    
    # 计算每个压力的平均特征
    sorted_pressures = sorted(pressure_stats.keys())
    avg_peak_freqs = []
    avg_centroids = []
    
    print(f"\n{'压力(Pressure)':<15} | {'平均主频(Hz)':<15} | {'平均质心(Hz)':<15}")
    print("-" * 50)
    
    for p in sorted_pressures:
        feats_matrix = np.array(pressure_stats[p])
        avg_feats = np.mean(feats_matrix, axis=0)
        
        avg_peak_freqs.append(avg_feats[0])
        avg_centroids.append(avg_feats[1])
        
        print(f"{p:<15} | {avg_feats[0]:<15.2f} | {avg_feats[1]:<15.2f}")
        
    plt.figure(figsize=(10, 6))
    
    # 绘制主频 (Peak Frequency) - 稍微淡一点，因为波动大
    plt.plot(sorted_pressures, avg_peak_freqs, 'o--', color='tab:blue', alpha=0.5, label='Peak Freq (Unstable)')
    
    # 绘制质心 (Spectral Centroid) - 加粗，作为主要结论
    plt.plot(sorted_pressures, avg_centroids, 's-', color='tab:orange', linewidth=2, label='Spectral Centroid (Stable)')
    
    # 添加线性拟合线 (针对质心)
    z = np.polyfit(sorted_pressures, avg_centroids, 1)
    p = np.poly1d(z)
    plt.plot(sorted_pressures, p(sorted_pressures), 'r:', linewidth=1.5, label=f'Trend Line (Centroid)')

    plt.title('Relationship between Tightness (Pressure) and Frequency Features')
    plt.xlabel('Pressure (Tightness)')
    plt.ylabel('Frequency (Hz)')
    plt.legend()
    plt.grid(True)
    plt.savefig('final_result_trend.png')
    print("\n趋势图已保存至: final_result_trend.png")
    
    # 绘制频谱对比图
    if spectra_for_plot:
        plt.figure(figsize=(12, 6))
        for p in sorted(spectra_for_plot.keys()):
            f, mag = spectra_for_plot[p]
            # Normalize by max value for comparison
            plt.plot(f, mag / np.max(mag), label=f'Pressure {p}')
        
        plt.title("Frequency Spectrum Comparison (Normalized)")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Normalized Amplitude")
        plt.xlim(0, 20000) # 只显示前20kHz
        plt.legend()
        plt.grid(True)
        plt.savefig('spectrum_comparison.png')
        print("频谱对比图已保存至: spectrum_comparison.png")

    # 混淆矩阵
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred_rf)
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=sorted(list(set(y))), yticklabels=sorted(list(set(y))))
    plt.title('Confusion Matrix (Random Forest)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('confusion_matrix.png')
    print("混淆矩阵已保存至: confusion_matrix.png")

if __name__ == "__main__":
    main()
