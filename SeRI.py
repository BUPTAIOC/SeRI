# coding:utf-8
import DataTools
import torch
import math
import numpy as np
import copy
import random
import matplotlib.cm as cm
from datetime import datetime
import sys
import csv

class Block2D:
    def __init__(self, channel, x1, x2, y1, y2, advv, k=1.0, direct=1):
        self.channel = channel
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.width = max(1, x2 - x1 + 1)
        self.hight = max(1, y2 - y1 + 1)
        self.S = self.width * self.hight
        self.direct = direct
        self.block_advv_l2 = 0
        self.score = 0
        self.if_chosen = 0

    def Scorer(self, advv, RGB_mod):
        if RGB_mod == 1:
            self.block_advv_l2 = torch.norm(advv[self.channel,self.y1:self.y2 + 1,self.x1:self.x2 + 1], p=2).item()
        elif RGB_mod == 0:
            self.block_advv_l2 = torch.norm(advv[:,self.y1:self.y2 + 1,self.x1:self.x2 + 1], p=2).item()
        self.score = self.block_advv_l2# self.S * self.k * self.k
        if self.S <= 1:
            self.score = 0

    def cut_block(self, advv, RGB_mod=1, if_autoXY=1, X=2, Y=2):
        if if_autoXY == 0:
            if self.hight < self.width:
                X, Y = 2, 1
            else:
                X, Y = 1, 2
        if self.S <= 1:
                X, Y = 1, 1
        bs = []
        for i in range(X):
            bs.append([])
            for j in range(Y):
                bs[i].append(copy.deepcopy(self))

        for i in range(X):
            Xline1 = self.x1 + (i * (self.x2 - self.x1)) // X
            Xline2 = self.x1 + ((i + 1) * (self.x2 - self.x1)) // X
            for j in range(Y):
                Yline1 = self.y1 + (j * (self.y2 - self.y1)) // Y
                Yline2 = self.y1 + ((j + 1) * (self.y2 - self.y1)) // Y
                bs[i][j].x1, bs[i][j].x2, bs[i][j].y1, bs[i][j].y2 = Xline1, Xline2, Yline1, Yline2
                if i > 0:
                    bs[i][j].x1 = Xline1 + 1
                if j > 0:
                    bs[i][j].y1 = Yline1 + 1
                bs[i][j].width = max(1, bs[i][j].x2 - bs[i][j].x1 + 1)
                bs[i][j].hight = max(1, bs[i][j].y2 - bs[i][j].y1 + 1)
                bs[i][j].S = max(1, bs[i][j].hight * bs[i][j].width)
                bs[i][j].if_chosen = 0
        return sum(bs, [])

class V2D:
    def __init__(self, size_channel, size_x, size_y, initSign, label):
        self.size_channel = size_channel
        self.size_x = size_x
        self.size_y = size_y
        self.pixnum = size_x * size_y * size_channel
        self.initV = torch.ones(self.size_channel, self.size_y, self.size_x, dtype=torch.float32).cuda()
        self.sens = torch.ones(self.size_channel, self.size_y, self.size_x, dtype=torch.float32).cuda() # 此矩阵二范数为‘1’
        self.ADBmax = 1.0  #
        self.ADBmin = 0.0  # 扰动下界，必定扰动失败
        self.adv_v = self.initV.clone()
        self.theta = 0
        self.adversarial_image = None
        self.label_attack = label
        self.RealL2 = torch.norm(self.adv_v, p=2)
        self.RealLinf = torch.max(abs(self.adv_v))
        if initSign == 0:
            for c in range(self.size_channel):
                for y in range(self.size_y):
                    for x in range(self.size_x):
                        self.adv_v[c][y][x] = random.choice([-1, 1])

    def real_R(self, imageXcuda):
        self.adversarial_image = torch.clamp(imageXcuda + self.adv_v * self.ADBmax, 0.0, 1.0)
        adv_v_real = self.adversarial_image - imageXcuda
        self.RealL2 = torch.norm(adv_v_real, p=2).item()
        self.RealLinf = torch.max(abs(adv_v_real)).item()

class Attacker:
    def __init__(self, args, model, original_image, imgi, label, ATK_target_xi, ATK_target_yi, InitATKer=None, iter_n=0):
        self.chosen_v = -1
        self.args = args
        self.model = model
        self.Img_cpu = original_image
        self.Img_cuda = original_image.cuda()
        self.Img_i = imgi
        self.label_origin = label
        self.ATK_target_xi = ATK_target_xi
        self.ATK_target_yi = ATK_target_yi
        self.Img_channels, self.Img_height, self.Img_width = (
            original_image.shape[1], original_image.shape[2], original_image.shape[3])
        self.Img_pixes = self.Img_channels * self.Img_width * self.Img_height
        self.RGB_chosen = []
        self.old_best_adv = V2D(self.Img_channels, self.Img_width, self.Img_height, self.args.initSign, label)
        if InitATKer is not None:
            self.old_best_adv.initV = (InitATKer.Img_result[2] - InitATKer.Img_result[1])[0].cuda()##############
            self.old_best_adv.sens = torch.ones(self.Img_channels, self.Img_height, self.Img_width, dtype=torch.float32).cuda()
            self.old_best_adv.adv_v = self.old_best_adv.initV.clone()
        self.aim_l2 = args.epsilon
        self.aim_ADB = self.aim_l2 / torch.norm(self.old_best_adv.adv_v, p=2).item()

        self.blocks = []
        self.chosen_block_i = 0
        self.chosen_theta = -1
        self.query = 0
        self.query_d = 0
        self.query_t = 0
        self.iter_n = iter_n
        self.iter_t = iter_n
        self.success = -1
        self.ACCquery = 0
        self.ACCiter_n = 0
        self.File_string = "none"
        self.Img_result = []
        self.heatmaps = []

        self.Nvs = []
        self.total_Kbridge = []

        self.Theta = []

        self.L2_line = [[0, self.Img_cpu.numel()**0.5]]
        self.Linf_line = [[0, 1.0]]
        self.blocks_l2sq_sum1 = 0
        self.blocks_l2sq_sum0 = 0
        self.blocks_s1 = 0
        self.blocks_s0 = 0
        if self.args.RGB_mod == 1:
            if 'R' in args.RGB:
                self.RGB_chosen.append(0)
            if 'G' in args.RGB:
                self.RGB_chosen.append(1)
            if 'B' in args.RGB:
                self.RGB_chosen.append(2)
            if self.Img_channels == 1:
                self.RGB_chosen = [0]
                self.blocks.extend(Block2D(0, 0, self.Img_width - 1, 0, self.Img_height - 1,
                                           self.old_best_adv.adv_v, 1.0, 1).cut_block(self.old_best_adv.adv_v))
            else:
                for i, channel in enumerate(self.RGB_chosen):
                    self.blocks.append(Block2D(channel, 0, self.Img_width - 1, 0, self.Img_height - 1,
                                               self.old_best_adv.adv_v,1.0, 1))
        elif self.args.RGB_mod == 0:
            self.RGB_chosen = [0]
            self.blocks.extend(Block2D(0, 0, self.Img_width - 1, 0, self.Img_height - 1,
                                       self.old_best_adv.adv_v, 1.0, 1).cut_block(self.old_best_adv.adv_v))



    def if_atk_success(self, F_x_add_adv):
        if self.args.targeted == 0:
            if F_x_add_adv != self.label_origin.item():
                return True
            else:
                return False
        if self.args.targeted == 1:
            if F_x_add_adv == self.ATK_target_yi:
                return True
            else:
                return False

    def show_message(self):
        sys.stdout.write(
            f'\rImg{self.Img_i} Query{self.query :.0f}'
            f'\tIter{self.iter_n :.0f}'
            f'\tl2=({self.old_best_adv.RealL2:.4f})'
            f'\tLAB={self.label_origin.item():.0f}->{self.old_best_adv.label_attack.item():.0f}')
        sys.stdout.flush()
        return

    def select_important_blocks(self):
        self.blocks_l2sq_sum1 = 0
        self.blocks_l2sq_sum0 = 0
        self.blocks_s1 = 0
        self.blocks_s0 = 0
        for b in self.blocks:
            b.Scorer(self.old_best_adv.adv_v,self.args.RGB_mod)
        self.blocks.sort(key=lambda x: x.score, reverse=True)
        self.chosen_block_i = 0
        for i in range(len(self.blocks)):
            if i == 0:
                self.blocks[i].if_chosen = 1
                self.blocks_l2sq_sum1 += self.blocks[i].block_advv_l2 **2
                self.blocks_s1 += self.blocks[i].S
            else:
                self.blocks[i].if_chosen = 0
                self.blocks_l2sq_sum0 += self.blocks[i].block_advv_l2 **2
                self.blocks_s0 += self.blocks[i].S
        return self.chosen_block_i


    def reset_perturbation(self, inV, new_theta, original_theta):
        inV_orig_l2 = torch.norm(inV.adv_v, p=2).item()
        b = self.blocks[0]
        k1 = new_theta / original_theta
        inV.theta = new_theta
        inV.ADBmin = 0
        if self.args.RGB_mod == 1:
            inV.adv_v[b.channel][b.y1:b.y2 + 1,b.x1:b.x2 + 1] *= k1
        elif self.args.RGB_mod == 0:
            inV.adv_v[:,b.y1:b.y2 + 1,b.x1:b.x2 + 1] *= k1
        inV_theta_l2 = torch.norm(inV.adv_v, p=2).item()
        repair_v = inV_orig_l2/inV_theta_l2
        inV.adv_v *= repair_v
        return self


    def compare_two_perturbations(self, VtestOld, VtestNew, tol=1e-5, binaryM=1):
        query = 0
        chosen_v = 0

        AdvvNew = VtestNew.adv_v.cuda()
        VtestNew.perturbed_image = torch.clamp(self.Img_cuda + VtestOld.ADBmax * AdvvNew, 0.0, 1.0)
        VtestNew.label_attack = self.model.predict_label(VtestNew.perturbed_image).cpu()
        query += 1
        if self.if_atk_success(VtestNew.label_attack) == 0:
            return query, chosen_v
        elif VtestOld.ADBmin > 0:
            VtestNew.perturbed_image = torch.clamp(self.Img_cuda + VtestOld.ADBmin * AdvvNew, 0.0, 1.0)
            label_attack2 = self.model.predict_label(VtestNew.perturbed_image).cpu()
            query += 1
            if self.if_atk_success(label_attack2) == 1:
                chosen_v = 1
                VtestNew.ADBmax = VtestOld.ADBmin
                return query, chosen_v

        AdvvOld = VtestOld.adv_v.cuda()
        low, high = copy.deepcopy(VtestOld.ADBmin), copy.deepcopy(VtestOld.ADBmax)
        while high - low > tol:
            ADB = DataTools.next_binary_rref(low, high, self.old_best_adv.ADBmax, binaryM)##############
            VtestNew.perturbed_image = torch.clamp(self.Img_cuda + ADB * AdvvNew, 0.0, 1.0)
            VtestOld.perturbed_image = torch.clamp(self.Img_cuda + ADB * AdvvOld, 0.0, 1.0)
            Newlabel_attack = self.model.predict_label(VtestNew.perturbed_image).cpu()
            Oldlabel_attack = self.model.predict_label(VtestOld.perturbed_image).cpu()
            query = query + 2

            if self.if_atk_success(Newlabel_attack) == 1 and self.if_atk_success(Oldlabel_attack) == 1:
                high, VtestNew.ADBmax, VtestOld.ADBmax = ADB, ADB, ADB
                VtestNew.label_attack, VtestOld.label_attack = Newlabel_attack, Oldlabel_attack
                chosen_v = 0
            elif self.if_atk_success(Newlabel_attack) == 0 and self.if_atk_success(Oldlabel_attack) == 0:
                low, VtestNew.ADBmin, VtestOld.ADBmin = ADB, ADB, ADB
            elif self.if_atk_success(Newlabel_attack) == 1 and self.if_atk_success(Oldlabel_attack) == 0:
                chosen_v = 1
                VtestNew.label_attack = Newlabel_attack
                VtestNew.ADBmax = ADB
                break
            elif self.if_atk_success(Newlabel_attack) == 0 and self.if_atk_success(Oldlabel_attack) == 1:
                chosen_v = 0
                VtestOld.label_attack = Oldlabel_attack
                VtestOld.ADBmax = ADB
                break
        return query, chosen_v

    def binary_Theta_search_old(self, thetaN=2):
        a, b = 0.0, 90.0

        """"""
        original_theta = (180.0 / math.pi) * math.atan(
            math.sqrt(self.blocks_l2sq_sum1 / (self.blocks_l2sq_sum0)))
        newTheta, oldTheta = 0, copy.deepcopy(original_theta)
        self.old_best_adv.theta = original_theta

        # newTheta总是尝试更靠左。如果oldTheta靠左，则newTheta靠右
        if original_theta >= (a + b) / 2:
            newTheta = (a + original_theta) / 2
        elif original_theta < (a + b) / 2:
            newTheta = (original_theta + b) / 2
        if newTheta > original_theta * 1.4:# limit the size of newTheta to maximum 2original_theta
            newTheta = original_theta * 1.4
            b = original_theta * 3

        VtestOld = copy.deepcopy(self.old_best_adv)
        VtestNew = copy.deepcopy(self.old_best_adv)
        iter_theta = 0
        while iter_theta < thetaN:
            self.reset_perturbation(VtestNew, newTheta, VtestNew.theta)
            query_plus, chosen_v = self.compare_two_perturbations(VtestOld, VtestNew)
            self.query += query_plus
            self.query_t += query_plus
            iter_theta = iter_theta + 1
            if chosen_v == 1:
                VtestOld, VtestNew = VtestNew, VtestOld
            VtestNew.ADBmax, VtestNew.ADBmin = self.old_best_adv.ADBmax, 0

            if oldTheta >= (a + b) / 2 and chosen_v == 0:
                a, oldTheta, b = newTheta, oldTheta, b
            elif oldTheta >= (a + b) / 2 and chosen_v == 1:
                a, oldTheta, b = a, newTheta, oldTheta
            elif oldTheta < (a + b) / 2 and chosen_v == 0:
                a, oldTheta, b = a, oldTheta, newTheta
            elif oldTheta < (a + b) / 2 and chosen_v == 1:
                a, oldTheta, b = oldTheta, newTheta, b

            if oldTheta >= (a + b) / 2:
                newTheta = (a + oldTheta) / 2
            elif oldTheta < (a + b) / 2:
                newTheta = (oldTheta + b) / 2
            if self.query >= self.args.budget2:
                break

        self.iter_n += 1
        self.iter_t += 1
        self.old_best_adv = VtestOld
        self.old_best_adv.real_R(self.Img_cuda)
        self.show_message()
        self.Theta.append([original_theta,self.old_best_adv.theta,self.old_best_adv.theta/original_theta])

    def binary_Theta_search(self, thetaN=2):
        a, b = 0.0, 90.0

        original_theta = (180.0 / math.pi) * math.atan(
            math.sqrt(self.blocks_l2sq_sum1 / (self.blocks_l2sq_sum0)))
        self.old_best_adv.theta = original_theta
        ThetaLow, ThetaHigh = self.args.k1 * original_theta, self.args.k2 * original_theta#ThetaLow, ThetaHigh = 0.2 * original_theta, 1.5 * original_theta
        VtestLow, VtestHigh = copy.deepcopy(self.old_best_adv), copy.deepcopy(self.old_best_adv)
        self.reset_perturbation(VtestLow, ThetaLow, original_theta)
        self.reset_perturbation(VtestHigh, ThetaHigh, original_theta)

        query_plus, chosen_v = self.compare_two_perturbations(self.old_best_adv, VtestHigh)
        self.query += query_plus
        self.query_t += query_plus

        if chosen_v == 1:
            self.old_best_adv = VtestHigh
        if chosen_v == 0:
            query_plus, chosen_v = self.compare_two_perturbations(self.old_best_adv, VtestHigh)
            self.query += query_plus
            self.query_t += query_plus
            if chosen_v == 1:
                self.old_best_adv = VtestHigh

        self.iter_n += 1
        self.iter_t += 1
        self.old_best_adv.real_R(self.Img_cuda)
        self.show_message()
        self.Theta.append([original_theta,self.old_best_adv.theta,self.old_best_adv.theta/original_theta])

    def attack(self):
        while self.query < self.args.budget2:
            self.chosen_block_i = self.select_important_blocks()
            self.L2_line.append([self.query, self.old_best_adv.RealL2])
            self.Linf_line.append([self.query, self.old_best_adv.RealLinf])
            self.binary_Theta_search()
            self.L2_line.append([self.query, self.old_best_adv.RealL2])
            self.Linf_line.append([self.query, self.old_best_adv.RealLinf])

            self.blocks.extend(self.blocks[self.chosen_block_i].cut_block(self.old_best_adv.adv_v))
            self.blocks.pop(self.chosen_block_i)
            if self.query >= self.args.budget2:
                break
            if self.success == -1 and self.old_best_adv.ADBmax <= self.aim_ADB:
                self.success = 1
                self.ACCquery, self.ACCiter_n = self.query, self.iter_n
            if self.args.early == 1 and self.success == 1:
                break

        real_out_label = self.model.predict_label(self.old_best_adv.adversarial_image.cuda()).item()
        """"""
        self.File_string = (str(self.args.dataset) + ",Img" + str(self.Img_i) + ",I-Q[" + str(self.iter_n) + "-" + str(
            self.args.budget2) + "],Label[" + str(self.label_origin.item()) + "-" + str(real_out_label) + "],ADB{:.4f}".format(
            self.old_best_adv.ADBmax) + ",T" + str(datetime.now().strftime("%H-%M-%S"))
                            )

        advimg = 0.5 * (1 + (self.old_best_adv.ADBmax / self.old_best_adv.RealLinf) * self.old_best_adv.adv_v)
        sens_array = (self.old_best_adv.adv_v / (self.old_best_adv.initV+1e-8)).cpu()
        sens_array = (sens_array-sens_array.min()) / (sens_array.max()-sens_array.min())
        sens_array = sens_array / torch.quantile(sens_array, 0.97)
        cmap = cm.get_cmap('coolwarm')
        self.Img_result = [self.Img_cuda,
                           advimg,
                           torch.tensor(cmap(sens_array.mean(dim=0))[:, :, :3]).permute(2, 0, 1).float() * 0.5 + self.Img_cpu * 0.5,
                           self.old_best_adv.adversarial_image]
        self.heatmaps = []
        for channel in range(self.Img_channels):
            heatmap = self.Img_cpu[0] * 0.5 + torch.tensor(cmap(sens_array[channel])[:, :, :3]).permute(2, 0, 1).float() * 0.5
            self.heatmaps.append(heatmap)
        return
