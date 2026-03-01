import os
import sys
import argparse
import time
import glob
import numpy as np
from tqdm import tqdm

# Ensure local imports work when run from anywhere
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 上级目录作为项目根目录
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils import image_read_cv2
from Evaluator import Evaluator


def build_basename_to_path_map(directory: str) -> dict:
    basename_to_path = {}
    if not os.path.isdir(directory):
        return basename_to_path
    for path in glob.glob(os.path.join(directory, '*')):
        if os.path.isdir(path):
            continue
        filename = os.path.basename(path)
        basename, _ = os.path.splitext(filename)
        basename_to_path[basename] = path
    return basename_to_path


def ensure_float32_0_255(image_array: np.ndarray) -> np.ndarray:
    if image_array.dtype != np.float32:
        image_array = image_array.astype(np.float32)
    # Heuristics: if max <= 1.0, assume 0..1 -> scale to 0..255
    max_val = float(np.max(image_array)) if image_array.size > 0 else 1.0
    min_val = float(np.min(image_array)) if image_array.size > 0 else 0.0
    if max_val <= 1.0 and min_val >= 0.0:
        image_array = image_array * 255.0
    return image_array


def compute_metrics_for_triplet(fused: np.ndarray, ir: np.ndarray, vi: np.ndarray) -> dict:
    fused = ensure_float32_0_255(fused)
    ir = ensure_float32_0_255(ir)
    vi = ensure_float32_0_255(vi)

    # Safety: shapes must match
    if fused.shape != ir.shape or fused.shape != vi.shape:
        raise ValueError(f"Shape mismatch: fused {fused.shape}, ir {ir.shape}, vi {vi.shape}")

    metrics = {}
    metrics['EN'] = float(Evaluator.EN(fused))
    metrics['PSNR'] = float(Evaluator.PSNR(fused, ir, vi))
    metrics['SCD'] = float(Evaluator.SCD(fused, ir, vi))
    metrics['VIFF'] = float(Evaluator.VIFF(fused, ir, vi))
    metrics['Qabf'] = float(Evaluator.Qabf(fused, ir, vi))
    metrics['SSIM'] = float(Evaluator.SSIM(fused, ir, vi))
    metrics['MS_SSIM'] = float(Evaluator.MS_SSIM(fused, ir, vi))
    return metrics


def main():
    parser = argparse.ArgumentParser('evaluate_saved_fusion_results')
    parser.add_argument('--ir_dir', type=str, default=os.path.join(PROJECT_ROOT, 'data/IVT_test/IVT_test_LLVIP', 'ir'), help='Path to IR images directory')
    parser.add_argument('--vi_dir', type=str, default=os.path.join(PROJECT_ROOT, 'data/IVT_test/IVT_test_LLVIP', 'vis'), help='Path to visible images directory')
    parser.add_argument('--fused_dir', type=str, default=os.path.join(PROJECT_ROOT, 'result/testImage/LLVIP'), help='Path to fused images directory produced by inference')
    # parser.add_argument('--save_dir', type=str, default=os.path.join(PROJECT_ROOT, 'metrics_result'), help='Where to save metrics outputs')
    # parser.add_argument('--per_image_csv', action='store_true', help='Also save per-image metrics as CSV')
    args = parser.parse_args()

    os.makedirs(args.fused_dir, exist_ok=True)

    ir_map = build_basename_to_path_map(args.ir_dir)
    vi_map = build_basename_to_path_map(args.vi_dir)
    fused_map = build_basename_to_path_map(args.fused_dir)

    common_basenames = sorted(set(fused_map.keys()) & set(ir_map.keys()) & set(vi_map.keys()))
    missing_in_ir = sorted(set(fused_map.keys()) - set(ir_map.keys()))
    missing_in_vi = sorted(set(fused_map.keys()) - set(vi_map.keys()))

    if len(common_basenames) == 0:
        print('未找到可匹配的文件，请检查文件名是否一致（不含扩展名）。')
        print(f"fused_dir: {args.fused_dir}\nir_dir: {args.ir_dir}\nvi_dir: {args.vi_dir}")
        sys.exit(1)

    if missing_in_ir:
        print(f"警告：以下融合结果在 ir_dir 中找不到同名源图：{missing_in_ir[:10]}{' ...' if len(missing_in_ir) > 10 else ''}")
    if missing_in_vi:
        print(f"警告：以下融合结果在 vi_dir 中找不到同名源图：{missing_in_vi[:10]}{' ...' if len(missing_in_vi) > 10 else ''}")

    metrics_names = ['EN', 'PSNR', 'SCD', 'VIFF', 'Qabf', 'SSIM', 'MS_SSIM']
    per_image_rows = []
    metrics_accumulator = {name: [] for name in metrics_names}

    for basename in tqdm(common_basenames, desc='Evaluating', ncols=80):
        fused_path = fused_map[basename]
        ir_path = ir_map[basename]
        vi_path = vi_map[basename]

        try:
            fused_img = image_read_cv2(fused_path, mode='GRAY')
            ir_img = image_read_cv2(ir_path, mode='GRAY')
            vi_img = image_read_cv2(vi_path, mode='GRAY')

            metrics = compute_metrics_for_triplet(fused_img, ir_img, vi_img)
            for k in metrics_names:
                metrics_accumulator[k].append(metrics[k])
            per_image_rows.append((basename, metrics))
        except Exception as exc:
            print(f"计算失败 {basename}: {exc}")
            continue

    # Averages
    avg_metrics = {k: (float(np.mean(v)) if len(v) > 0 else float('nan')) for k, v in metrics_accumulator.items()}

    # Human-readable summary
    checkpoint_hint = os.path.basename(os.path.abspath(args.fused_dir))
    metrics_filename = f"!metrics_{checkpoint_hint}.txt"
    metrics_file_path = os.path.join(args.fused_dir, metrics_filename)
    with open(metrics_file_path, 'w', encoding='utf-8') as f:
        f.write(f"Fused Dir: {args.fused_dir}\n")
        f.write(f"IR Dir: {args.ir_dir}\n")
        f.write(f"VI Dir: {args.vi_dir}\n")
        f.write(f"共评估图像数: {len(per_image_rows)}\n")
        f.write("\n--- 指标平均值（测试集） ---\n")
        explanations = {
            'EN': '(↑ 越大越好，信息量更丰富)√',
            'PSNR': '(↑ 越大越好，图像质量更好)',
            'SCD': '(↑ 越大越好，相关性更好)√',
            'VIFF': '(↑ 越大越好，视觉信息保真度更高)',
            'Qabf': '(↑ 越大越好，边缘信息保留更好)√',
            'SSIM': '(↑ 越大越好，结构相似性更高)',
            'MS_SSIM': '(↑ 越接近1越好，多尺度结构相似性更高)√',
        }
        for k in metrics_names:
            f.write(f"{k}: {avg_metrics[k]:.6f} {explanations.get(k, '')}\n")
        save_timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        f.write(f"保存时间: {save_timestamp}\n")

    print(f"平均指标已保存: {metrics_file_path}")

if __name__ == '__main__':
    main()


