#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# ============================================
# @Time     : 2020/02/24 15:47
# @Author   : WanDaoYi
# @FileName : selective_search.py
# ============================================

from datetime import datetime
import numpy as np
import skimage.segmentation as seg
import skimage.util as util
import skimage.color as color
import skimage.feature as feature
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image


class SelectiveSearch(object):

    def __init__(self):

        self.colour_bins = 25
        self.texture_bins = 10
        pass

    def selective_search(self, image_orig, scale=1.0, sigma=0.8, min_size=50):
        # selective search 处理的是灰度图，单通道。如果不是 3 通道则拦截。
        # 如果是 PIL 的 Image.open() 读取的图像，则要用下面的方法判断 3 通道
        assert len(image_orig.split()) == 3, "3ch image is expected"
        # 如果是 cv2 的格式图像的话，直接 shape 就好。后面的处理 cv2 图像需要转成 PIL 图像
        # assert image_orig.shape[-1] == 3, "3ch image is expected"

        # 获取分割后的 4 通道 图像
        image_info = self.generate_segment(image_orig, scale=scale, sigma=sigma,
                                           min_size=min_size)

        if image_info is None:
            return None, {}

        # 获取 regions
        reg = self.extract_regions(image_info)

        # 提取相邻区域信息
        neighbours = self.extract_neighbours(reg)

        # 计算图像的大小
        h, w = image_info.shape[: 2]
        image_size = h * w

        # 相似度计算初始化
        similar = {}
        for (ai, ar), (bi, br) in neighbours:
            similar[(ai, bi)] = self.calc_similar(ar, br, image_size)

        # 层级查找
        while similar != {}:

            # 获取最高的相似度
            i, j = sorted(similar.items(), key=lambda i: i[1])[-1][0]

            # 合并相应的区域
            top_key = max(reg.keys()) + 1.0
            # 区域合并
            reg[top_key] = self.merge_regions(reg[i], reg[j])

            # 对需要移除的相似性区域进行标记
            key_to_delete = []
            for k, v in list(similar.items()):

                if (i in k) or (j in k):
                    key_to_delete.append(k)
                pass

            # 移除旧相似度的相关区域
            for k in key_to_delete:
                del similar[k]

            # 计算新区域的相似度
            for k in [a for a in key_to_delete if a != (i, j)]:
                n = k[1] if k[0] in (i, j) else k[0]
                similar[(top_key, n)] = self.calc_similar(reg[top_key], reg[n], image_size)
                pass
            pass

        reg_list = []
        for k, r in list(reg.items()):
            reg_list.append({"rect": (r["min_x"],
                                      r["min_y"],
                                      r["max_x"] - r["min_x"],
                                      r["max_y"] - r["min_y"]),
                             "size": r["size"],
                             "labels": r["labels"]
                             })
            pass

        return image_info, reg_list
        pass

    # 区域合并
    def merge_regions(self, r1, r2):
        new_size = r1["size"] + r2["size"]

        merge_reg = {"min_x": min(r1["min_x"], r2["min_x"]),
                     "min_y": min(r1["min_y"], r2["min_y"]),
                     "max_x": max(r1["max_x"], r2["max_x"]),
                     "max_y": max(r1["max_y"], r2["max_y"]),
                     "size": new_size,
                     "hist_c": (r1["hist_c"] * r1["size"] + r2["hist_c"] * r2["size"]) / new_size,
                     "hist_t": (r1["hist_t"] * r1["size"] + r2["hist_t"] * r2["size"]) / new_size,
                     "labels": r1["labels"] + r2["labels"]
                     }
        return merge_reg

    # 相似度计算
    def calc_similar(self, r1, r2, image_size):

        return (self.similar_colour(r1, r2) + self.similar_texture(r1, r2)
                + self.similar_size(r1, r2, image_size)
                + self.similar_fill(r1, r2, image_size))
        pass

    # 计算颜色直方图相交的和
    def similar_colour(self, r1, r2):
        return sum([min(a, b) for a, b in zip(r1["hist_c"], r2["hist_c"])])
        pass

    # 计算纹理直方图相交的和
    def similar_texture(self, r1, r2):
        return sum([min(a, b) for a, b in zip(r1["hist_t"], r2["hist_t"])])
        pass

    # 计算尺寸相似的图像
    def similar_size(self, r1, r2, image_size):
        return 1.0 - (r1["size"] + r2["size"]) / image_size
        pass

    # 计算交叠相似的图像
    def similar_fill(self, r1, r2, image_size):
        width = max(r1["max_x"], r2["max_x"]) - min(r1["min_x"], r2["min_x"])
        high = max(r1["max_y"], r2["max_y"]) - min(r1["min_y"], r2["min_y"])
        bbox_size = width * high
        return 1.0 - (bbox_size - r1["size"] - r2["size"]) / image_size
        pass

    # 图像分割
    def generate_segment(self, image_orig, scale, sigma, min_size):
        # 计算 Felsenszwalb 的基于有效图的图像分割。
        # 使用基于图像网格的快速，最小生成树聚类生成多通道（即RGB）图像的过度分割。
        # 该参数scale设置观察级别。规模越大意味着越来越小的部分。
        # sigma是高斯核的直径，用于在分割之前平滑图像。
        # 生产环节的数量及其规模只能通过scale间接控制。
        # 根据局部对比度不同，图像中的分段大小可能会有很大差异。
        # 图像:(宽度，高度，3）或（宽度，高度）ndarray输入图像。scale：float可用参数。
        # 较高意味着较大的群集 sigma：float预处理中使用的高斯内核的宽度。
        # min_size：int最小组件大小。使用后处理强制执行。
        # 多通道：bool，可选（默认值：True）图像的最后一个轴是否被解释为多个通道。
        # 对于3D图像，False的值目前不受支持。
        # segment_mask :(宽度，高度）ndarray整数掩码，指示段标签
        segment_mask = seg.felzenszwalb(util.img_as_float(image_orig),
                                        scale=scale, sigma=sigma,
                                        min_size=min_size
                                        )

        # 根据通道数 concatenation, 增加 1 个 0 值 的通道
        # image_mask.shape[:2] 获取 high 和 width 大小
        image_info = np.append(image_orig,
                               np.zeros(segment_mask.shape[:2])[:, :, np.newaxis],
                               axis=2)

        # 将上面 felzenszwalb 分割得到的图像 放入到 新增的 0 值 通道当中
        image_info[:, :, 3] = segment_mask

        # 返回 4 通道的图像
        return image_info
        pass

    # 提取 regions
    def extract_regions(self, image_info):

        # regions dic
        reg = {}
        # 第一步，计算像素的位置
        for number, pixel in enumerate(image_info):

            for i, (r, g, b, l) in enumerate(pixel):
                # 初始化一个新的 region
                if l not in reg:
                    reg[l] = {"min_x": 0xffff, "min_y": 0xffff,
                              "max_x": 0, "max_y": 0,
                              "labels": [l]}

                # bounding box
                if reg[l]["min_x"] > i:
                    reg[l]["min_x"] = i

                if reg[l]["min_y"] > number:
                    reg[l]["min_y"] = number

                if reg[l]["max_x"] < i:
                    reg[l]["max_x"] = i

                if reg[l]["max_y"] < number:
                    reg[l]["max_y"] = number

            pass

        # 第二步，计算 texture gradient
        tex_grad = np.zeros((image_info.shape[0], image_info.shape[1], image_info.shape[2]))
        # 基于高斯导数的选择性搜索算法对 8 个方向进行计算，这里使用 LBP 替代。
        for colour_channel in (0, 1, 2):
            tex_grad[:, :, colour_channel] = feature.local_binary_pattern(image_info[:, :, colour_channel],
                                                                          8, 1.0)

        # 第三步，计算每个 region 的 colour histogram
        # 将 rgb 转为 hsv
        image_hsv = color.rgb2hsv(image_info[:, :, :3])
        for k, v in list(reg.items()):
            masked_pixels = image_hsv[:, :, :][image_info[:, :, 3] == k]
            reg[k]["size"] = len(masked_pixels / 4)
            reg[k]["hist_c"] = self.calc_colour_hist(masked_pixels)

            # texture histogram
            reg[k]["hist_t"] = self.calc_texture_hist(tex_grad[:, :][image_info[:, :, 3] == k])
            pass

        return reg
        pass

    # 计算颜色直方图
    def calc_colour_hist(self, image_info):
        hist = np.array([])
        # 处理 (0, 1, 2) 通道的数据
        for colour_channel in range(3):
            # 提取 一个颜色通道
            c = image_info[:, colour_channel]
            # 计算每种颜色的直方图并加入结果中
            hist = np.concatenate([hist] + [np.histogram(c, self.colour_bins, (0.0, 255.0))[0]])

        # L1 norm
        hist = hist / len(image_info)

        return hist
        pass

    # 计算每个 region 的纹理直方图
    def calc_texture_hist(self, image_info):
        hist = np.array([])
        # 处理 (0, 1, 2) 通道的数据
        for colour_channel in range(3):
            # 彩色通道掩膜
            c = image_info[:, colour_channel]
            # 计算直方图的每个方向，并将结果 拼接起来。
            hist = np.concatenate([hist] + [np.histogram(c, self.texture_bins, (0.0, 1.0))[0]])
            pass

        # L1 norm
        hist = hist / len(image_info)
        return hist
        pass

    # 提取 相邻区域信息
    def extract_neighbours(self, regions):
        reg = list(regions.items())
        neighbours = []
        for cur, dic_a in enumerate(reg[: -1]):
            for dic_b in reg[cur + 1:]:
                if self.neighbour_flag(dic_a[1], dic_b[1]):
                    neighbours.append((dic_a, dic_b))

        return neighbours
        pass

    def neighbour_flag(self, dic_a, dic_b):
        flag_1 = dic_a["min_x"] < dic_b["min_x"] < dic_a["max_x"]
        flag_2 = dic_a["min_y"] < dic_b["min_y"] < dic_a["max_y"]

        flag_3 = dic_a["min_x"] < dic_b["max_x"] < dic_a["max_x"]
        flag_4 = dic_a["min_y"] < dic_b["max_y"] < dic_a["max_y"]

        if (flag_1 and flag_2) or (flag_3 and flag_4) or (flag_1 and flag_4) or (flag_3 and flag_2):
            return True

        return False
        pass


if __name__ == "__main__":
    # 代码开始时间
    start_time = datetime.utcnow()
    print("开始时间: ", start_time)

    image_input_path = "./dataset/images/Monica.png"
    image_output_path = "./output/images/newMonica.png"
    # 读取图像
    image_orig = Image.open(image_input_path)

    demo = SelectiveSearch()

    image_info, reg_list = demo.selective_search(image_orig, scale=400.0, sigma=0.9, min_size=10)
    print("ok")
    print(reg_list[2])

    bounding_box = set()
    for reg in reg_list:
        # 如果是相同的矩形框，则跳过
        if reg["rect"] in bounding_box:
            continue

        # 不包含小于 2k pixels 的区域
        if reg["size"] < 2000:
            continue

        # 筛除 长宽比过大的矩形框
        x, y, w, h = reg["rect"]
        if w / h > 1.2 or h / w > 1.2:
            continue

        bounding_box.add(reg["rect"])
        pass

    # 在原始图像上绘制矩形框
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    ax.imshow(image_orig)
    # # 或者 使用下面图像的 前面 3 个通道
    # # 如果展示的时候全是白光，那就是数值太大，除以 255 标准化后，可以将图像清晰展示
    # ax.imshow(image_info[:, :, :3] / 255)

    for x, y, w, h in bounding_box:
        print(x, y, w, h)
        rect = mpatches.Rectangle((x, y), w, h, fill=False,
                                  edgecolor="red", linewidth=1)
        ax.add_patch(rect)

    # 不展示坐标轴
    plt.axis("off")
    # 保存图像
    plt.savefig(image_output_path)
    plt.show()

    # 代码结束时间
    end_time = datetime.utcnow()
    print("结束时间: ", end_time, ", 训练模型耗时: ", end_time - start_time)
    pass
