import os.path
import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms
import cv2
import torch.nn as nn
from PIL import Image
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import pickle
from tqdm import tqdm
import threading
import random
from torchvision.transforms import functional as F
import tempfile
import shutil



class Data(Dataset):
    def __init__(self, mode, use_mask_num=20, cache_mask_num=50, crop_size=(600, 800), cache_dir=None, root_dir=None):
        self.root_dir = root_dir
        self.crop_size = crop_size
        
        # 获取文件列表并保存扩展名信息
        self.img_list = []
        self.extensions = {}
        self.text_files_map = {} # 新增：用于存储图像名到文本文件路径列表的映射
        
        for filename in os.listdir(os.path.join(self.root_dir, 'vis')):
            name, ext = os.path.splitext(filename)
            self.img_list.append(name)
            self.extensions[name] = ext
            
        self.img_dir = root_dir

        # 确认红外图像数量与可见光图像数量一致
        assert len(os.listdir(os.path.join(self.img_dir, 'ir'))) == len(self.img_list)

        assert mode == 'train' or mode == 'test', "dataset mode not specified"
        self.mode = mode
        if mode=='train':
            # 不使用RandomResizedCrop，我们将自定义裁剪逻辑
            self.transform = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5)])
        elif mode=='test':
            self.transform = transforms.Compose([])

        
        self.cache_mask_num = cache_mask_num  # 缓存中每张图片生成的掩码数量
        self.use_mask_num = min(use_mask_num, cache_mask_num)  # 实际使用的掩码数量，不能超过缓存的数量
        self.totensor = transforms.ToTensor()
        
        # 设置缓存目录
        # self.cache_dir = cache_dir if cache_dir else os.path.join(self.root_dir, 'Mask_cache')
        self.cache_dir = cache_dir if cache_dir else os.path.join(self.root_dir, 'refined_SAM_dual_modal')
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # 初始化掩码缓存
        self.mask_cache = {}
        
        # 检查是否有缓存文件 - 注意这里使用cache_mask_num作为缓存文件名的一部分
        # cache_file = os.path.join(self.cache_dir, f'mask_cache_{mode}_{cache_mask_num}.pkl')
        cache_file = os.path.join(self.cache_dir, 'merged_cache.pkl')

        if os.path.exists(cache_file):
            print(f"Loading mask cache from {cache_file}")
            
            with open(cache_file, 'rb') as f:
                self.mask_cache = pickle.load(f)
            # print(f"Loaded masks for {len(self.mask_cache)} images (cached: {cache_mask_num}, using: {use_mask_num})")
            # --从中断点开始
            print(f"Loaded masks for {len(self.mask_cache)} images.")
            if len(self.mask_cache) < len(self.img_list):
                print(f"Cache is incomplete ({len(self.mask_cache)} vs {len(self.img_list)} total). Resuming generation...")
                self._initialize_sam_and_generate_masks(cache_file) # 传入已部分加载的 self.mask_cache
                
        else:
            # 初始化SAM模型并生成所有掩码
            print(f"Initializing SAM model and generating {cache_mask_num} masks per image...")
            self._initialize_sam_and_generate_masks(cache_file) # self.mask_cache 为空
        
        # 用于跟踪是否已经打印过全零掩码警告
        self.zero_mask_warning_printed = False

        # !!!
        # 扫描 text 文件夹，构建一个从图像名到其对应文本文件路径列表的映射
        vis_dir = os.path.join(self.root_dir, 'vis')
        text_dir = os.path.join(self.root_dir, 'text') # 假设你的 text 文件夹与 Vis, Ir 同级

        for filename in os.listdir(vis_dir):
            name, ext = os.path.splitext(filename)
            if os.path.isfile(os.path.join(vis_dir, filename)): # 确保是文件
                
                self.extensions[name] = ext

                # 查找对应的文本文件
                # 假设文本文件名格式为 name_X.txt, 其中X是数字
                current_text_files = []
                if os.path.exists(text_dir): # 检查text目录是否存在
                    for text_filename in os.listdir(text_dir):
                        if text_filename.startswith(name + "_") and text_filename.endswith(".txt"):
                            current_text_files.append(os.path.join(text_dir, text_filename))
                self.text_files_map[name] = sorted(current_text_files) # 排序以保证一致性
        # !!!
        
    def _safe_save_cache(self, cache_file, cache_data):
        """
        安全保存缓存文件，使用临时文件 + 原子替换机制
        避免写入过程中中断导致原文件损坏
        """
        # 创建临时文件（在同一目录下，确保在同一文件系统上以支持原子移动）
        cache_dir = os.path.dirname(cache_file)
        temp_file = None
        
        try:
            # 使用临时文件，在同一目录下创建以确保原子操作
            with tempfile.NamedTemporaryFile(
                mode='wb', 
                dir=cache_dir, 
                prefix='mask_cache_tmp_', 
                suffix='.pkl', 
                delete=False
            ) as temp_f:
                temp_file = temp_f.name
                pickle.dump(cache_data, temp_f)
                temp_f.flush()  # 确保数据写入磁盘
                os.fsync(temp_f.fileno())  # 强制同步到磁盘
            
            # 原子替换：将临时文件重命名为目标文件
            shutil.move(temp_file, cache_file)
            print(f"Successfully saved cache to {cache_file}")
            
        except Exception as e:
            # 如果出错，清理临时文件
            if temp_file and os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass
            raise e  # 重新抛出异常

    def _initialize_sam_and_generate_masks(self, cache_file):
        # 初始化SAM模型
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        sam = sam_model_registry["vit_b"](checkpoint='/data2/lms/Model/SAM/sam_vit_b_01ec64.pth').to(device)
        mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=128,  # 降低点数
            pred_iou_thresh=0.86,  # 提高IoU阈值
            stability_score_thresh=0.92,  # 提高稳定性分数阈值
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100,  # 保持最小区域面积
            output_mode='binary_mask',
        )
        
        processed_keys = set(self.mask_cache.keys()) # 获取已在缓存中的图像名--从中断点开始
        
        # 生成所有掩码并缓存
        for idx in tqdm(range(len(self.img_list)), desc="Generating masks"):
            name_0 = self.img_list[idx]
            
            # --从中断点开始
            if name_0 in processed_keys: # 如果该图像的掩码已在缓存中，则跳过
                # print(f"Mask for {name_0} already in cache. Skipping.")
                continue

            ext = self.extensions.get(name_0, '.png')  # 获取扩展名，默认为.png
            
            ir_path_0 = os.path.join(self.img_dir, 'ir', name_0 + ext)
            vis_path_0 = os.path.join(self.img_dir, 'vis', name_0 + ext)
            
            # 读取图像
            ir_img = cv2.imread(ir_path_0)
            vis_img = cv2.imread(vis_path_0)
            
            # 生成掩码
            ir_patches = mask_generator.generate(ir_img)
            ir_patches.sort(key=lambda x: x['area'], reverse=True)
            
            vis_patches = mask_generator.generate(vis_img)
            vis_patches.sort(key=lambda x: x['area'], reverse=True)
            
            # --- 新增：定义并创建保存SAM分割结果的图像目录 ---
            # sam_image_save_dir_base = os.path.join(self.cache_dir, 'sam_images')
            # os.makedirs(sam_image_save_dir_base, exist_ok=True)
            # current_sam_image_save_dir = os.path.join(sam_image_save_dir_base, name_0)
            # os.makedirs(current_sam_image_save_dir, exist_ok=True)
            # --- 结束新增 ---
            
            # 存储掩码 - 使用cache_mask_num
            ir_masks = []
            vis_masks = []
            
            # --- 新增：初始化用于合并掩码的空白图像 ---
            # 需要获取图像尺寸来创建空白画布
            h, w = ir_img.shape[:2]
            combined_ir_mask_img = np.zeros((h, w), dtype=np.uint8)
            combined_vis_mask_img = np.zeros((h, w), dtype=np.uint8)
            # --- 结束新增 ---

            # --- 修改：在循环中合并掩码，而不是单独保存 ---
            for i in range(min(self.cache_mask_num, len(ir_patches), len(vis_patches))):
                ir_mask_bool = ir_patches[i]['segmentation']
                vis_mask_bool = vis_patches[i]['segmentation']

                ir_masks.append(ir_mask_bool)
                vis_masks.append(vis_mask_bool)

                # --- 修改：合并掩码到累积图像 --- 
                ir_mask_uint8 = ir_mask_bool.astype(np.uint8)
                vis_mask_uint8 = vis_mask_bool.astype(np.uint8)
                combined_ir_mask_img = cv2.bitwise_or(combined_ir_mask_img, ir_mask_uint8)
                combined_vis_mask_img = cv2.bitwise_or(combined_vis_mask_img, vis_mask_uint8)
                # --- 结束修改 --- 
                
            # --- 新增：在循环结束后保存合并后的掩码图像 ---
            # cv2.imwrite(os.path.join(current_sam_image_save_dir, 'combined_ir_masks.png'), combined_ir_mask_img * 255)
            # cv2.imwrite(os.path.join(current_sam_image_save_dir, 'combined_vis_masks.png'), combined_vis_mask_img * 255)
            # --- 结束新增 ---

            self.mask_cache[name_0] = {
                'ir_masks': ir_masks,
                'vis_masks': vis_masks
            }

            '''
            # 每100个样本保存一次缓存,防止中断丢失这意味着中间保存会覆盖上一次的 cache_file。
            # 例如,处理到第100个样本时,它会保存这100个样本的掩码到 cache_file。
            # 处理到第200个样本时,它会保存这200个样本的掩码到同一个 cache_file,
            # 覆盖掉之前只包含100个样本的版本
            
            # 每100个样本保存一次缓存，防止中断丢失
            if (idx + 1) % 100 == 0:
                with open(cache_file, 'wb') as f:
                    pickle.dump(self.mask_cache, f)
            '''

            # --从中断点开始
            # 每100个新处理的样本（或者按总数）保存一次缓存
            if (len(self.mask_cache) % 100 == 0 and len(self.mask_cache) > len(processed_keys)) or (idx + 1) == len(self.img_list) :
                 print(f"Saving intermediate cache with {len(self.mask_cache)} entries...")
                 self._safe_save_cache(cache_file, self.mask_cache)
        
        # 保存最终缓存
        self._safe_save_cache(cache_file, self.mask_cache)
        
        print(f"Mask generation complete. Saved {self.cache_mask_num} masks per image to {cache_file}")

    def random_crop(self, image, seed, target_size):
        """
        简单的随机裁剪函数，不依赖掩码
        """
        # 设置随机种子确保一致性
        torch.manual_seed(seed)
        random.seed(seed)
        
        c, h, w = image.shape
        target_h, target_w = target_size
        
        # 随机裁剪整个图像
        if h <= target_h:
            i = 0
            crop_h = h
        else:
            i = torch.randint(0, h - target_h + 1, (1,)).item()
            crop_h = target_h
            
        if w <= target_w:
            j = 0
            crop_w = w
        else:
            j = torch.randint(0, w - target_w + 1, (1,)).item()
            crop_w = target_w
            
        cropped = image[:, i:i+crop_h, j:j+crop_w]
        
        # 如果裁剪后的大小不符合目标大小，则fv调整大小
        if cropped.shape[1] != target_h or cropped.shape[2] != target_w:
            cropped = F.resize(cropped, target_size)
        
        return cropped

    def segmentation_aware_random_crop(self, image, mask, seed, target_size):
        """
        在包含分割区域的边界框内进行随机裁剪，然后调整大小----要限定边界框，不然会容易有全0掩码
        处理掩码全零的情况
        
        Args:
            image: 输入图像张量 [C, H, W]
            mask: 分割掩码张量 [H, W] 或 [1, H, W]
            seed: 随机种子
            target_size: 目标大小 (h, w)
            
        Returns:
            裁剪并调整大小后的图像
        """
        # 设置随机种子确保一致性
        torch.manual_seed(seed)
        random.seed(seed)
        
        # 确保mask是2D的
        if mask.dim() == 3 and mask.size(0) == 1:
            mask = mask.squeeze(0)
        
        # 获取图像和掩码的尺寸
        c, h, w = image.shape
        target_h, target_w = target_size
        
        # 找到掩码中非零区域的坐标
        non_zero_indices = torch.nonzero(mask > 0.5, as_tuple=False)
        
        # 如果掩码为空（全零），则进行普通的随机裁剪
        if len(non_zero_indices) == 0:
            # 只打印一次警告
            if not self.zero_mask_warning_printed:
                print("")
                self.zero_mask_warning_printed = True
                
            # 使用简单的随机裁剪
            return self.random_crop(image, seed, target_size)
        else:
            # 获取掩码的边界框
            min_y, min_x = non_zero_indices.min(0)[0]
            max_y, max_x = non_zero_indices.max(0)[0]
            
            # 计算边界框的尺寸
            box_h = max_y - min_y + 1
            box_w = max_x - min_x + 1
            
            # 确保边界框至少与目标大小一样大
            # 如果边界框小于目标大小，则扩展边界框
            if box_h < target_h:
                padding = target_h - box_h
                min_y = max(0, min_y - padding // 2)
                max_y = min(h - 1, max_y + padding // 2 + padding % 2)
                box_h = max_y - min_y + 1
            
            if box_w < target_w:
                padding = target_w - box_w
                min_x = max(0, min_x - padding // 2)
                max_x = min(w - 1, max_x + padding // 2 + padding % 2)
                box_w = max_x - min_x + 1
            
            # 在边界框内随机选择裁剪起点
            if box_h > target_h:
                i = min_y + torch.randint(0, box_h - target_h + 1, (1,)).item()
            else:
                i = min_y
            
            if box_w > target_w:
                j = min_x + torch.randint(0, box_w - target_w + 1, (1,)).item()
            else:
                j = min_x
            
            # 确保裁剪区域不超出图像边界
            i = min(max(0, i), h - target_h)
            j = min(max(0, j), w - target_w)
            
            # 执行裁剪
            crop_h = min(h - i, target_h)
            crop_w = min(w - j, target_w)
            cropped = image[:, i:i+crop_h, j:j+crop_w]
        
        # 如果裁剪后的大小不符合目标大小，则调整大小
        if cropped.shape[1] != target_h or cropped.shape[2] != target_w:
            cropped = F.resize(cropped, target_size)
        
        return cropped

    def smart_crop_union(self, image, ir_mask, vis_mask, seed, target_size):
        """
        使用 ir_mask 与 vis_mask 的并集来指导随机裁剪；若并集为空或尺寸不足则回退到 random_crop。

        Args:
            image (Tensor): 输入图像张量 [C, H, W]
            ir_mask (Tensor): 红外掩码张量 [H, W] 或 [1, H, W]
            vis_mask (Tensor): 可见光掩码张量 [H, W] 或 [1, H, W]
            seed (int): 随机种子，保证与其他模态同步
            target_size (tuple[int,int]): (target_h, target_w)

        Returns:
            Tensor: 裁剪并调整到 target_size 的图像张量
        """
        # 创建并集掩码
        # 先保证掩码维度为 [H, W]
        if ir_mask.dim() == 3 and ir_mask.size(0) == 1:
            ir_mask = ir_mask.squeeze(0)
        if vis_mask.dim() == 3 and vis_mask.size(0) == 1:
            vis_mask = vis_mask.squeeze(0)

        union_mask = (ir_mask > 0.5) | (vis_mask > 0.5)

        # 若并集为空，直接 fallback
        if torch.count_nonzero(union_mask) == 0:
            return self.random_crop(image, seed, target_size)

        # 设置随机种子，保证与其他裁剪同步
        torch.manual_seed(seed)
        random.seed(seed)

        c, h, w = image.shape
        target_h, target_w = target_size

        # 计算包围框
        non_zero = torch.nonzero(union_mask, as_tuple=False)
        min_y, min_x = non_zero.min(0)[0]
        max_y, max_x = non_zero.max(0)[0]

        box_h = max_y - min_y + 1
        box_w = max_x - min_x + 1

        # 如果包围框尺寸小于目标尺寸则尝试向外扩张
        if box_h < target_h:
            pad = target_h - box_h
            min_y = max(0, min_y - pad // 2)
            max_y = min(h - 1, max_y + pad - pad // 2)
            box_h = max_y - min_y + 1
        if box_w < target_w:
            pad = target_w - box_w
            min_x = max(0, min_x - pad // 2)
            max_x = min(w - 1, max_x + pad - pad // 2)
            box_w = max_x - min_x + 1

        # 再次检查尺寸，若仍不足则 fallback
        if box_h < target_h or box_w < target_w:
            return self.random_crop(image, seed, target_size)

        # 在边界框内随机选择裁剪起点
        if box_h > target_h:
            i = min_y + torch.randint(0, box_h - target_h + 1, (1,)).item()
        else:
            i = min_y
        
        if box_w > target_w:
            j = min_x + torch.randint(0, box_w - target_w + 1, (1,)).item()
        else:
            j = min_x
        
        # 确保裁剪区域不超出图像边界
        i = min(max(0, i), h - target_h)
        j = min(max(0, j), w - target_w)
        
        # 执行裁剪
        crop_h = min(h - i, target_h)
        crop_w = min(w - j, target_w)
        cropped = image[:, i:i+crop_h, j:j+crop_w]

        # 如果裁剪结果尺寸与目标尺寸不同（边界情况），进行 resize
        if cropped.shape[1] != target_h or cropped.shape[2] != target_w:
            cropped = F.resize(cropped, target_size)

        return cropped

    # ---------------------------  新增函数  --------------------------- #
    '''
    def smart_crop_separate1(self, image, mask, seed, target_size):
        """
        使用 *单一模态* 掩码来指导随机裁剪。

        与 smart_crop_union 不同，本方法**完全依赖传入的 mask**；
        如果该掩码为空，才退回到 random_crop。这样可避免 IR / VIS 并集带来的
        "互相污染" 问题，适合想单独保留各模态兴趣区域的实验。

        Args:
            image (Tensor): 输入图像张量 [C, H, W]
            mask  (Tensor): 对应模态的掩码张量 [H, W] 或 [1, H, W]
            seed  (int)   : 随机种子，保证同一图像多次调用可复现
            target_size (tuple[int,int]): (target_h, target_w)

        Returns:
            Tensor: 裁剪并调整到 target_size 的图像张量
        """
        # 直接调用已有的单掩码感知裁剪；内部已包含对全零掩码的处理与回退
        return self.segmentation_aware_random_crop(image, mask, seed, target_size)
        '''

    def __getitem__(self, idx):
        seed = torch.random.seed()

        name_0 = self.img_list[idx]  # 获取当前索引的图像名
        ext = self.extensions.get(name_0, '.png')  # 获取扩展名，默认为.png

        # label = []

        # label_item_path = os.path.join(self.img_dir, 'Label', name_0 + ext)
        # label_mask = cv2.imread(label_item_path)
        # label_mask_tensor = self.totensor(cv2.cvtColor(label_mask, cv2.COLOR_BGR2GRAY))

        ir_path_0 = os.path.join(self.img_dir, 'ir', name_0 + ext)
        vis_path_0 = os.path.join(self.img_dir, 'vis', name_0 + ext)
        ir_0 = cv2.imread(ir_path_0)
        vi_0 = cv2.imread(vis_path_0)
        ir_0_tensor = self.totensor(cv2.cvtColor(ir_0, cv2.COLOR_BGR2GRAY))
        vi_0_tensor = self.totensor(cv2.cvtColor(vi_0, cv2.COLOR_BGR2YCrCb)) # CHW
        
         # 在训练模式下，使用分割感知随机裁剪
        # 先尝试获取掩码（若存在，则用掩码驱动裁剪确保几何对齐）
        cached_masks = self.mask_cache.get(name_0)

        if self.mode == 'train':
            '''
            # 检查标签掩码是否全零
            if torch.sum(label_mask_tensor) > 0:
                # 先对标签掩码进行裁剪
                label_mask_tensor = self.segmentation_aware_random_crop(label_mask_tensor, label_mask_tensor, seed, self.crop_size)
                
                # 对其他图像使用相同的裁剪参数
                ir_0_tensor = self.segmentation_aware_random_crop(ir_0_tensor, label_mask_tensor, seed, self.crop_size)
                vi_0_tensor = self.segmentation_aware_random_crop(vi_0_tensor, label_mask_tensor, seed, self.crop_size)
            else:
                # 如果标签掩码全零，使用简单的随机裁剪
                if not self.zero_mask_warning_printed:
                    print("Warning: Label mask is zero, performing standard random crop")
                    self.zero_mask_warning_printed = True
                
                # 使用相同的种子进行简单的随机裁剪
                label_mask_tensor = self.random_crop(label_mask_tensor, seed, self.crop_size)
                ir_0_tensor = self.random_crop(ir_0_tensor, seed, self.crop_size)
                vi_0_tensor = self.random_crop(vi_0_tensor, seed, self.crop_size)
            '''


            # ================================================
            if cached_masks is not None and len(cached_masks['ir_masks']) > 0 and len(cached_masks['vis_masks']) > 0:
                # 策略：把N个mask做并集，再取包围框
                # 1. 计算IR模态内部所有mask的并集
                ir_union_mask = None
                for ir_mask in cached_masks['ir_masks'][:self.use_mask_num]:  # 只取实际使用的mask数量
                    if ir_union_mask is None:
                        ir_union_mask = ir_mask.astype(bool)
                    else:
                        ir_union_mask = ir_union_mask | ir_mask.astype(bool)
                
                # 2. 计算VIS模态内部所有mask的并集
                vis_union_mask = None
                for vis_mask in cached_masks['vis_masks'][:self.use_mask_num]:  # 只取实际使用的mask数量
                    if vis_union_mask is None:
                        vis_union_mask = vis_mask.astype(bool)
                    else:
                        vis_union_mask = vis_union_mask | vis_mask.astype(bool)
                
                # 3. 转换为tensor
                mask_ir_union_tensor = torch.from_numpy(ir_union_mask.astype(np.float32))
                mask_vis_union_tensor = torch.from_numpy(vis_union_mask.astype(np.float32))

                # 4. 分别用各自的并集mask裁剪主干输入
                ir_0_tensor = self.segmentation_aware_random_crop(ir_0_tensor,
                                                       mask_ir_union_tensor,
                                                       seed, self.crop_size)
                vi_0_tensor = self.segmentation_aware_random_crop(vi_0_tensor,
                                                       mask_vis_union_tensor,
                                                       seed, self.crop_size)
            else:
                # 没有掩码可用，回退到随机裁剪
                ir_0_tensor = self.random_crop(ir_0_tensor, seed, self.crop_size)
                vi_0_tensor = self.random_crop(vi_0_tensor, seed, self.crop_size)

            # ================================================

            # 使用随机裁剪---这是原图
            # ir_0_tensor=self.random_crop(ir_0_tensor,seed,self.crop_size)
            # vi_0_tensor=self.random_crop(vi_0_tensor,seed,self.crop_size)
            # 应用其他变换（如水平翻转）
            '''
            torch.manual_seed(seed)
            label_mask_tensor=self.transform(label_mask_tensor)
            '''

            # 统一做随机翻转等数据增强
            torch.manual_seed(seed)
            ir_0_tensor = self.transform(ir_0_tensor)

            torch.manual_seed(seed)
            vi_0_tensor = self.transform(vi_0_tensor)
        
        y_0 = vi_0_tensor[0, :, :].unsqueeze(dim=0).clone()
        cb = vi_0_tensor[1, :, :].unsqueeze(dim=0)
        cr = vi_0_tensor[2, :, :].unsqueeze(dim=0)

        irs = []
        ys = []
        
        # 从缓存中获取掩码
        # cached_masks = self.mask_cache.get(name_0)
        if cached_masks:
            ir_img = cv2.imread(ir_path_0)
            vis_img = cv2.imread(vis_path_0)
            
            # 计数有效掩码
            valid_mask_count = 0
            
            # 注意这里使用use_mask_num而不是cache_mask_num
            for i in range(min(self.cache_mask_num, len(cached_masks['ir_masks']))):
                # 检查掩码是否全零
                ir_mask = cached_masks['ir_masks'][i]
                vis_mask = cached_masks['vis_masks'][i]
                
                if not np.any(ir_mask) or not np.any(vis_mask):
                    # 跳过全零掩码，但不打印警告
                    # 尝试在这里打印东西，但是并没有，因为这里SAM生成的并没有全零掩码
                    continue
                
                # 应用红外掩码
                ir_position = ~ir_mask
                ir_masked = ir_img.copy()
                ir_masked[ir_position] = 0
                
                # 应用可见光掩码
                vis_position = ~vis_mask
                vis_masked = vis_img.copy()
                vis_masked[vis_position] = 0
                
                try:
                    ir_2_tensor = self.totensor(cv2.cvtColor(ir_masked, cv2.COLOR_BGR2GRAY))
                    vi_2_tensor = self.totensor(cv2.cvtColor(vis_masked, cv2.COLOR_BGR2YCrCb))
                    
                    # 在训练模式下，使用分割感知随机裁剪
                    if self.mode == 'train':
                        # 使用与原始图像相同的裁剪和变换
                        '''
                        if torch.sum(label_mask_tensor) > 0:
                            ir_2_tensor = self.segmentation_aware_random_crop(ir_2_tensor, label_mask_tensor, seed, self.crop_size)
                            vi_2_tensor = self.segmentation_aware_random_crop(vi_2_tensor, label_mask_tensor, seed, self.crop_size)
                        else:
                            ir_2_tensor = self.random_crop(ir_2_tensor, seed, self.crop_size)
                            vi_2_tensor = self.random_crop(vi_2_tensor, seed, self.crop_size)
                        '''
                        mask_ir_tensor  = torch.from_numpy(ir_mask.astype(np.float32))
                        mask_vis_tensor = torch.from_numpy(vis_mask.astype(np.float32))

                        # 使用并集掩码
                        # ir_2_tensor = self.smart_crop_union(ir_2_tensor,
                        #             mask_ir_tensor,
                        #             mask_vis_tensor,
                        #             seed, self.crop_size)

                        # vi_2_tensor = self.smart_crop_union(vi_2_tensor,
                        #             mask_ir_tensor,
                        #             mask_vis_tensor,
                        #             seed, self.crop_size)

                        # 使用单掩码
                        ir_2_tensor = self.segmentation_aware_random_crop(ir_2_tensor,
                                    mask_ir_tensor,
                                    seed, self.crop_size)
                        vi_2_tensor = self.segmentation_aware_random_crop(vi_2_tensor,
                                    mask_vis_tensor,
                                    seed, self.crop_size)

                        
                        torch.manual_seed(seed)
                        ir_2_tensor = self.transform(ir_2_tensor)
                        
                        torch.manual_seed(seed)
                        vi_2_tensor = self.transform(vi_2_tensor)
                    
                    y = vi_2_tensor[0, :, :].unsqueeze(dim=0)
                    
                    irs.append(ir_2_tensor)
                    ys.append(y)
                    
                    # 增加有效掩码计数
                    valid_mask_count += 1
                    
                    # 如果已经收集了足够的有效掩码，就退出循环
                    if valid_mask_count >= self.use_mask_num:
                        break
                        
                except Exception as e:
                    # 出错时继续，但不打印详细错误信息
                    continue
        
        # 如果掩码数量不足，用原图填充
        while len(irs) < self.use_mask_num:
            irs.append(ir_0_tensor.clone())
            ys.append(y_0.clone())

        ys_0 = torch.cat(ys, dim=0)
        irs_0 = torch.cat(irs, dim=0)
        
        # !!!
        # --- 新增：加载参考 captions ---
        reference_captions_list = []
        if name_0 in self.text_files_map:
            for text_file_path in self.text_files_map[name_0]:
                try:
                    with open(text_file_path, 'r', encoding='utf-8') as f:
                        caption = f.read().strip()
                        if caption: # 确保caption不是空的
                            reference_captions_list.append(caption)
                except Exception as e:
                    print(f"Warning: Could not read or process text file {text_file_path}: {e}")
        # 如果没有找到文本文件或所有文本文件都为空，reference_captions_list 将为空
        # 你可能需要根据具体情况处理这种情况，例如提供一个默认的通用caption或跳过这个样本的caption损失
        # ---!!!! ---

        result = {'name':name_0, 
                  'irs':irs_0, 
                  'ys':ys_0, 
                  # 'label':label, 
                  'ir':ir_0_tensor, 
                  'y':y_0, 
                  'cb':cb, 
                  'cr':cr, 
                  # 'label_mask': label_mask_tensor,   # gt
                  'reference_captions': reference_captions_list # 新增的参考captions
            }

        return result

    def trans(self, x, seed):
        torch.manual_seed(seed)
        x = self.transform(x)
        return x

    def __len__(self):
        return len(self.img_list)
