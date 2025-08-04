import math

import numpy as np
import torch
import DataTools
import os
import sys
import copy
import time

class V(object):
    def __init__(self):
        self.ADBmax = 0
        self.RealL2 = 0
        self.RealLinf = 0

class Attacker(object):
    def __init__(self, model, args):
        self.model = model
        self.epsilon = args.epsilon
        self.old_ADBbest = None
        self.old_ADBmax = 1.0
        self.x_final = None

        self.args = args
        self.Img_i = 0
        self.old_best_adv = V()
        self.File_string = "RayS"
        self.Img_result = []
        self.heatmaps = self.Img_result
        self.Img_result = []
        self.heatmaps = self.Img_result
        self.L2_line = [[0, 0]]
        self.Linf_line = [[0, 1.0]]
        self.success = -1
        self.queries = 0
        self.ACCquery = 0
        self.ACCiter_n = 1

    def get_xadv(self, x, v, d):
        out = x.cuda() + d * v.cuda()
        out = torch.clamp(out, 0., 1.)
        return out

    def attack(self, x, y, Img_i, target=None, query_limit=10000):
        self.Img_i = Img_i
        self.L2_line = [[0, x.numel() ** 0.5]]
        self.Linf_line = [[0, 1.0]]
        self.success = -1
        self.queries = 0
        self.ACCquery = 0
        self.ACCiter_n = 1

        shape = list(x.shape)
        dim = np.prod(shape[1:])

        self.old_ADBbest = torch.sign(torch.ones(shape)).cuda()
        self.old_ADBmax = 1.0

        real_linf = self.old_ADBmax
        self.x_final = self.get_xadv(x, self.old_ADBbest, self.old_ADBmax).cuda()
 
        block_level = 0
        block_ind = 0
        for i in range(query_limit):
            block_num = 2 ** block_level
            block_size = int(np.ceil(dim / block_num))
            start, end = block_ind * block_size, min(dim, (block_ind + 1) * block_size)
            mid = int(np.ceil((start+end)/2))

            attempt1 = self.old_ADBbest.clone().view(dim)
            attempt1[start:mid] *= -1.
            attempt1 = attempt1.view(shape)
            attempt2 = self.old_ADBbest.clone().view(dim)
            attempt2[mid:end] *= -1.
            attempt2 = attempt2.view(shape)
            attempts = [attempt1, attempt2]
            ADBmaxs = self.compare_directions_fast(x, y, attempts, 1e-5)
            if self.chosen_v >= 0:
                self.old_ADBbest = attempts[self.chosen_v]
                self.old_ADBmax = ADBmaxs[self.chosen_v]
                self.x_final = self.get_xadv(x, self.old_ADBbest, self.old_ADBmax).cuda()

            """
            attempt = self.old_ADBbest.clone().view(dim)
            attempt[start:end] *= -1.
            attempt = attempt.view(shape)
            #self.binary_search(x, y, target, attempt, 1e-3/math.sqrt(dim))
            """
            block_ind += 1
            if block_ind == 2 ** block_level or end == dim:
                block_level += 1
                block_ind = 0

            current_best_adv_v = self.x_final - x
            real_linf = torch.max(torch.abs(current_best_adv_v)).item()
            real_l2 = torch.norm(current_best_adv_v).item()

            sys.stdout.write(f'\rImg{self.Img_i} Query{self.queries:.0f} \tIter{i + 1:.0f} \tADB={self.old_ADBmax:.6f} \treal_l2|linf={real_l2:.5f}|{real_linf:.5f}')
            sys.stdout.flush()

            self.old_best_adv.RealL2 = min(torch.norm(x).item(), real_l2)
            self.old_best_adv.RealLinf = min(1, real_linf)
            self.old_best_adv.ADBmax = min(torch.norm(x).item(), real_linf)
            self.L2_line.append([self.queries, self.old_best_adv.RealL2])
            self.Linf_line.append([self.queries, self.old_best_adv.RealLinf])
            if self.queries >= self.args.budget:
                break
            if self.success == -1 and self.old_best_adv.RealLinf <= self.args.epsilon:
                self.success = 1
                self.ACCquery, self.ACCiter_n = self.queries, i
            if self.args.early == 1 and self.success == 1:
                break

        advimg = 0.5 * (1.0 + current_best_adv_v / self.old_best_adv.RealLinf)
        self.Img_result = [advimg, x, x + current_best_adv_v]
        self.heatmaps = self.Img_result
        return self.x_final, self.queries, real_linf, (real_linf <= self.epsilon)

    # check whether solution is found
    def ATK_succ(self, f_x, y, target=None):
        self.queries += 1
        if target:
            if f_x == target:
                return 1
            else:
                return 0
        else:
            if f_x != y:
                return 1
            else:
                return 0

    # binary search for decision boundary along attempts direction
    def binary_search(self, x, y, target, attempt, tol):
        d_min = 0
        d_max = copy.deepcopy(self.old_ADBmax)
        initial_succ = self.ATK_succ(self.model.predict_label(self.get_xadv(x, attempt, self.old_ADBmax)), y, target)
        while initial_succ == 1 and d_max - d_min > tol:
            d_mid = (d_min + d_max) / 2.0
            ATK_succ = self.ATK_succ(self.model.predict_label(self.get_xadv(x, attempt, d_mid)), y, target)
            if ATK_succ == 1:
                d_max = d_mid
            else:
                d_min = d_mid
        if d_max < self.old_ADBmax:
            self.old_ADBmax = d_max
            self.x_final = self.get_xadv(x, attempt, d_max)
            self.old_ADBbest = attempt
        return

    def compare_directions_fast(self, Img_cuda, label, NewDirs, tol):
        perturbed_images = []
        label_out = []
        succV = []
        ADBmaxs = [self.old_ADBmax, self.old_ADBmax]
        ADBmins = [0, 0]
        self.chosen_v = -1
        for i in range(len(NewDirs)):
            perturbed_images.append(self.get_xadv(Img_cuda, NewDirs[i], self.old_ADBmax))
            label_out.append(self.model.predict_label(perturbed_images[i]))
            if self.ATK_succ(label_out[i], label) == 1:
                succV.append(i)
                ADBmaxs[i] = copy.deepcopy(self.old_ADBmax)
                self.chosen_v = i
            else:
                ADBmins[i] = copy.deepcopy(self.old_ADBmax)

        if len(succV) == 0:
            self.chosen_v = -1
            return ADBmaxs
        elif len(succV) == 1:
            self.chosen_v = succV[0]
            return ADBmaxs

        low, high = 0, copy.deepcopy(self.old_ADBmax)
        while high - low > tol:
            mid = DataTools.next_binary_rref(low, high, self.old_ADBmax, self.args.binaryM)
            succVtemp = copy.deepcopy(succV)
            vi = 0
            while vi < len(succVtemp):
                DirI = succVtemp[vi]
                perturbed_images[DirI] = self.get_xadv(Img_cuda, NewDirs[DirI], mid)
                label_out[DirI] = self.model.predict_label(perturbed_images[DirI])
                if self.ATK_succ(label_out[DirI], label) == 1:
                    ADBmaxs[DirI] = mid
                    self.chosen_v = succVtemp[vi]
                    vi = vi + 1
                else:
                    ADBmins[DirI] = mid
                    succVtemp.pop(vi)

            if len(succVtemp) == 0:
                low = mid
            elif len(succVtemp) == 1:
                self.chosen_v = succVtemp[0]
                return ADBmaxs
            elif len(succVtemp) >= 2:
                high = mid
                succV = copy.deepcopy(succVtemp)

        self.chosen_v = succV[0]
        return ADBmaxs