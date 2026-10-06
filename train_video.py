import os
import subprocess
import time
import threading
import psutil

# ===视频路径 ===
video_path = r'E:\ultralytics-main\outputs_seg_cls\predict32\VID_20240410_145322.avi'

fps = 2
current_path = os.getcwd()
folder_path = os.path.dirname(video_path)
images_path = os.path.join(folder_path, 'input')
ffmpeg_path = os.path.join(current_path, 'external', r'ffmpeg/bin/ffmpeg.exe')  # FFmpeg下载网站：      https://ffmpeg.org/
os.makedirs(images_path, exist_ok=True)

log_file = os.path.join(folder_path, 'resource_log.txt')

# === GPU/CPU 监测线程 ===
monitoring = True
stats = []

def monitor_resources(interval=5):
    global stats
    print("开始资源监控...")
    while monitoring:
        # GPU
        try:
            gpu_info = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu', '--format=csv,nounits,noheader'],
                encoding='utf-8'
            ).strip().split('\n')[0].split(',')
            gpu_mem_used = int(gpu_info[0].strip())
            gpu_mem_total = int(gpu_info[1].strip())
            gpu_util = int(gpu_info[2].strip())
        except Exception:
            gpu_mem_used = gpu_mem_total = gpu_util = -1  

        # CPU
        cpu_usage = psutil.cpu_percent(interval=None)
        ram = psutil.virtual_memory()
        ram_used = ram.used // (1024 * 1024)
        ram_total = ram.total // (1024 * 1024)

        stats.append({
            'time': time.strftime('%H:%M:%S'),
            'gpu_mem': f"{gpu_mem_used}/{gpu_mem_total} MB",
            'gpu_util': f"{gpu_util}%",
            'cpu': f"{cpu_usage}%",
            'ram': f"{ram_used}/{ram_total} MB"
        })

        time.sleep(interval)

# === 时间记录函数 ===
def print_time_cost(name, start_time, end_time):
    elapsed = end_time - start_time
    print(f"[时间统计] {name} 耗时：{elapsed:.2f} 秒 ≈ {elapsed / 60:.2f} 分钟")

def save_stats_to_file(path):
    with open(path, 'w') as f:
        f.write("Time,GPU_Mem,GPU_Util,CPU_Usage,RAM_Usage\n")
        for stat in stats:
            f.write(f"{stat['time']},{stat['gpu_mem']},{stat['gpu_util']},{stat['cpu']},{stat['ram']}\n")

all_start_time=time.time()
# === 步骤1：视频帧提取 ===
start = time.time()
print("正在进行视频帧提取...")
ffmpeg_cmd = f'{ffmpeg_path} -i "{video_path}" -qscale:v 1 -qmin 1 -vf fps={fps} "{images_path}\\%04d.jpg"'
subprocess.run(ffmpeg_cmd, shell=True)
end = time.time()
print_time_cost("视频帧提取", start, end)

# === 步骤2：COLMAP 位姿估计 ===
start = time.time()
print("正在进行 COLMAP 相机估计...")
colmap_cmd = f'python convert.py -s "{folder_path}"'
subprocess.run(colmap_cmd, shell=True)
end = time.time()
print_time_cost("COLMAP 位姿估计", start, end)

# === 步骤3：3DGS 模型训练 ===
start = time.time()
print("正在进行 3DGS 模型训练 + 资源监控...")

# 启动资源监控线程
monitor_thread = threading.Thread(target=monitor_resources)
monitor_thread.start()

train_cmd = f'python train.py -s "{folder_path}" --eval'
subprocess.run(train_cmd, shell=True)

# 停止资源监控
monitoring = False
monitor_thread.join()

end = time.time()
print_time_cost("3DGS 模型训练", start, end)

# === 输出资源占用记录 ===
print(f"保存资源使用日志到: {log_file}")
save_stats_to_file(log_file)

# === 总流程耗时 ===
print("=" * 40)
print_time_cost("总流程", all_start_time, end)
print("=" * 40)
