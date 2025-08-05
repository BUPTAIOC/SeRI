# coding:utf-8
import os
import subprocess
import torch
import random
from PIL import Image
from torchvision import utils
from scipy.integrate import quad
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from sklearn.cluster import KMeans

def save_and_resize_image(image_tensor, file_path):
    utils.save_image(image_tensor, file_path)
    img = Image.open(file_path)
    img = img.resize((224, 224), Image.NEAREST)
    img.save(file_path, format='BMP')


def concatenate_images(image_paths):
    images = [[Image.open(img) for img in imgrol] for imgrol in image_paths]
    new_img = Image.new('RGB', (224*len(images[0]), 224*len(images)))
    for y, imgrol in enumerate(images):
        for x, img in enumerate(imgrol):
            new_img.paste(img, (224*x, 224*y))
    return new_img

""" """
def save_images(images, folder_name, plustring):
    img_floder_name = os.path.join(folder_name, "adversarial_samples")
    if not os.path.exists(img_floder_name):
        os.makedirs(img_floder_name)

    paths = []
    for i, imgrol in enumerate(images):
        paths.append([])
        for j, img in enumerate(imgrol):
            paths[i].append(os.path.join(img_floder_name, f"{i}-{j}OF_{plustring}.bmp"))
            save_and_resize_image(img, paths[i][j])

    # Concatenate and save combined image in BMP format
    combined_image = concatenate_images(paths)
    combined_image.save(os.path.join(img_floder_name, f"COMB_{plustring}.bmp"), format='BMP')

    for i, pthrol in enumerate(paths):
        for j, pth in enumerate(pthrol):
            os.remove(pth)
""""""

def open_image(image_path):
    if os.name == 'nt':  # Windows
        os.startfile(image_path)
    elif os.name == 'posix':  # macOS and Linux
        subprocess.run(['open', image_path], check=True)


def progress_bar(imgi, query, iter, total, ADB, l2, label_origin=0, label_after=0, bar_length=10):
    # percent = 100 * (progress / float(total))
    #bar_fill = int(bar_length * query / total)
    #bar = '█' * bar_fill + '-' * (bar_length - bar_fill)
    # sys.stdout.write(f'\r[{bar}] Q{percent:.1f}% R{Rnow:.3f}')
    sys.stdout.write(f'\rImg{imgi} Query{query :.0f} \tIter{iter :.0f} \tADB={ADB:.6f}({l2:.6f}) \tLAB={label_origin:.0f}->{label_after:.0f}')
    sys.stdout.flush()

def RlineQ(Rline, radius_line, budget):
    start = 0
    for t in range(len(Rline) - 1):
        for q in range(start, min(Rline[t + 1][0], budget)):
            radius_line[q] = radius_line[q] + Rline[t][1]
            start = Rline[t + 1][0]
    return

def draw_distribution(x, name="pic"):

    plt.figure(figsize=(10, 8))
    counts, bins, patches = plt.hist(x, bins=50, color='skyblue', edgecolor='black', weights=np.ones(len(x)) / len(x))

    #for i in range(len(patches)):
    #    plt.text(bins[i] + (bins[1] - bins[0]) / 2, counts[i] + 0.001, f"{counts[i] * 100:.1f}%", ha='center')
    plt.xlim(0, 1)
    plt.xticks(np.arange(0.1, 1.1, 0.1), fontsize=30)
    plt.tick_params(axis='both', which='major', labelsize=30)

    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y * 100:.0f}%"))

    #plt.title("Histogram with Percentage", fontsize=14)
    #plt.xlabel("Value", fontsize=12)
    #plt.ylabel("Frequency (Percentage)", fontsize=12)

    plt.tight_layout()
    plt.show()


##################################################################################


a_hat = 0.03133292769944518
b_hat = 3.0659694403842903
c_hat = 0.16755646970211466
d_hat = 0.13403850261898806



def func_y(r, a, b, c, d):
    return a / ((r + d) ** b) + c



def find_midK_of_k1k2(k1, k2):
    Sk1k2, error = quad(func_y, k1, k2, args=(a_hat, b_hat, c_hat, d_hat))
    low, high = k1, k2
    mid = (low + high) / 2
    Sk1mid = None
    while high - low > 1.0 / 10000:
        mid = (low + high) / 2
        Sk1mid, error = quad(func_y, k1, mid, args=(a_hat, b_hat, c_hat, d_hat))
        if Sk1mid < Sk1k2 / 2:
            low = mid
        else:
            high = mid
    return mid


def next_binary_rref(r1, r2, max_r, mod):
    if mod == 0:
        return (r1 + r2) / 2.0
    if mod == 1:
        k1, k2 = r1 / max_r, r2 / max_r
        kmid = find_midK_of_k1k2(1-k2, 1-k1)
        median = (1-kmid) * max_r
        return median

def cosine_similarity(tensor1, tensor2):
    dot_product = torch.dot(tensor1.flatten(), tensor2.flatten())
    norm1, norm2 = torch.norm(tensor1), torch.norm(tensor2)
    similarity = dot_product / (norm1 * norm2)
    return similarity