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

def extract_features(freqs, magnitude):
    """
    提取多个特征用于分类
    """
    # 1. 忽略低频 (0-100Hz)
    valid_mask = freqs > 100
    if np.sum(valid_mask) == 0:
        return [0, 0, 0, 0]
        
    f = freqs[valid_mask]
    m = magnitude[valid_mask]
    
    # 特征1: 最大峰值频率 (Dominant Frequency)
    idx_max = np.argmax(m)
    peak_freq = f[idx_max]
    
    # 特征2: 频谱质心 (Spectral Centroid)
    centroid = np.sum(f * m) / (np.sum(m) + 1e-6)
    
    # 特征3: 带宽 (Spectral Bandwidth)
    bandwidth = np.sqrt(np.sum(((f - centroid)**2) * m) / (np.sum(m) + 1e-6))
    
    # 特征4: 能量 (Energy) - 简单的幅度平方和
    energy = np.sum(m**2)
    
    return [peak_freq, centroid, bandwidth, energy]

def main():
    print("=== 开始处理 DSP 大作业任务 ===")
    
    files = glob.glob(os.path.join(DATA_DIR, 'acquisitionData-*.txt'))
    
    X = [] # 特征矩阵
    y = [] # 标签 (压力值)
    
    # 用于统计每个压力下的平均特征
    pressure_stats = {}
    
    for file_path in files:
        filename = os.path.basename(file_path)
        pressure = parse_pressure(filename)
        if pressure is None: continue
        
        print(f"处理文件: {filename} (Label: {pressure})")
        
        try:
            raw_data = np.loadtxt(file_path)
            segments = segment_signal(raw_data)
            print(f"  -> 提取到 {len(segments)} 个样本")
            
            for seg in segments:
                freqs, mag = compute_fft(seg)
                feats = extract_features(freqs, mag)
                
                X.append(feats)
                y.append(pressure)
                
                # 累加用于统计
                if pressure not in pressure_stats:
                    pressure_stats[pressure] = []
                pressure_stats[pressure].append(feats)
                
        except Exception as e:
            print(f"  -> 错误: {e}")
            
    X = np.array(X)
    y = np.array(y)
    
    print(f"\n数据集构建完成: 总样本数 {len(X)}, 特征数 {X.shape[1]}")
    
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
    plt.plot(sorted_pressures, avg_peak_freqs, 'b-o', label='Peak Frequency')
    plt.plot(sorted_pressures, avg_centroids, 'r-s', label='Spectral Centroid')
    plt.title('Relationship between Tightness (Pressure) and Frequency Features')
    plt.xlabel('Pressure (Tightness)')
    plt.ylabel('Frequency (Hz)')
    plt.legend()
    plt.grid(True)
    plt.savefig('final_result_trend.png')
    print("\n趋势图已保存至: final_result_trend.png")
    
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
