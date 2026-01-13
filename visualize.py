import numpy as np
import matplotlib.pyplot as plt
import os

# 配置
data_dir = '槽楔模型测试数据'
filename = 'acquisitionData-400.txt'
file_path = os.path.join(data_dir, filename)

def load_and_plot():
    print(f"正在读取文件: {file_path}")
    
    try:
        # 读取数据
        data = np.loadtxt(file_path)
        print(f"数据读取成功，总点数: {len(data)}")
        print(f"最大值: {np.max(data):.4f}, 最小值: {np.min(data):.4f}")
        
        # 简单可视化
        plt.figure(figsize=(12, 6))
        plt.plot(data, label='Raw Signal')
        plt.title(f'Waveform: {filename}')
        plt.xlabel('Sample Index')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.legend()
        
        # 保存图片
        output_img = 'waveform_preview.png'
        plt.savefig(output_img)
        print(f"波形图已保存至: {output_img}")
        
        # 简单的阈值检测来估算敲击次数
        # 假设信号绝对值超过某个阈值算作一次敲击的开始
        # 这里动态设定阈值为最大幅度的 20%
        threshold = np.max(np.abs(data)) * 0.2
        above_threshold = np.abs(data) > threshold
        
        # 简单的去抖动逻辑计算峰值数量
        # 只要间隔超过一定点数（比如2000点）就算新的敲击
        peaks = []
        last_peak_idx = -10000
        min_distance = 2000 # 假设两次敲击间隔至少2000个采样点
        
        for i, is_above in enumerate(above_threshold):
            if is_above:
                if i - last_peak_idx > min_distance:
                    peaks.append(i)
                    last_peak_idx = i
        
        print(f"初步检测到的敲击次数: {len(peaks)}")
        print(f"敲击大约发生的位置 (索引): {peaks}")
        
        # 在图上标记
        plt.plot(peaks, data[peaks], 'rx', label='Detected Taps')
        plt.legend()
        plt.savefig('waveform_with_peaks.png')
        
    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    load_and_plot()
