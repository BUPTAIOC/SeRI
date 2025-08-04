import copy
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
from scipy.fftpack import dct, idct
import math
import sys
import DataTools

def clip_image_values(x, minv, maxv):
    x = torch.max(x, minv)
    x = torch.min(x, maxv)
    return x


def valid_bounds(img, delta=1.0):
    im = copy.deepcopy(np.asarray(img))
    im = im.astype(np.float32)
    # General valid bounds [0, 255]
    valid_lb = np.zeros_like(im)
    valid_ub = np.ones_like(im)
    # Compute the bounds
    lb = im - delta
    ub = im + delta
    # Validate that the bounds are in [0, 255]
    lb = np.maximum(valid_lb, np.minimum(lb, im))
    ub = np.minimum(valid_ub, np.maximum(ub, im))
    return lb, ub


def inv_tf(x, mean, std):
    for i in range(len(mean)):
        x[i] = np.multiply(x[i], std[i], dtype=np.float32)
        x[i] = np.add(x[i], mean[i], dtype=np.float32)
    x = np.swapaxes(x, 0, 2)
    x = np.swapaxes(x, 0, 1)
    return x


def inv_tf_pert(r):

    pert = np.sum(np.absolute(r), axis=0)
    pert[pert != 0] = 1
    return pert


def get_label(x):
    s = x.split(' ')
    label = ''
    for l in range(1, len(s)):
        label += s[l] + ' '
    return label


def nnz_pixels(arr):
    return np.count_nonzero(np.sum(np.absolute(arr), axis=0))

class V(object):
    def __init__(self):
        self.ADBmax = 0
        self.RealL2 = 0
        self.RealLinf = 0

class Attacker():
    def __init__(self, args, model, im_orig, imgi,
                 tar_img = None, tar_label = None, attack_method = 'CGBA', dim_reduc_factor=4,
                 iteration=93, initial_query=30, tol=0.0001, sigma=0.0002,
                 verbose_control='Yes'):
        self.model = model
        self.x0 = im_orig.cuda()
        self.x0_label = model.predict_label(self.x0).cpu().item()
        self.tar_img = tar_img
        if tar_img != None:
            self.tar_label = tar_label
        self.dim_reduc_factor = dim_reduc_factor
        if im_orig.shape[3] < 224:
            self.dim_reduc_factor = 1
        self.Max_iter = iteration
        self.Init_query = initial_query

        lb, ub = valid_bounds(im_orig, 1.0)
        self.lb = torch.from_numpy(lb).cuda()
        self.ub = torch.from_numpy(ub).cuda()
        self.tol = tol
        self.sigma = sigma
        self.grad_estimator_batch_size = 40
        self.verbose_control = verbose_control
        self.attack_method = attack_method
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.args = args
        self.Img_i = imgi
        self.old_best_adv = V()
        self.File_string = "CGBA"
        self.Img_result = []
        self.heatmaps = self.Img_result
        self.L2_line = [[0, self.x0.numel() ** 0.5]]
        self.Linf_line = [[0, 1.0]]
        self.success = -1
        self.queries = 0
        self.ACCquery = 0
        self.ACCiter_n = 1

        self.total_Kbridge = []

    def is_adversarial(self, image):
        predict_label = self.model.predict_label(image).cpu().item()
        self.queries += 1
        if self.tar_img == None:
            is_adv = predict_label != self.x0_label
        else:
            is_adv = predict_label == self.tar_label
        if is_adv:
            return 1
        else:
            return -1

    def find_init_adversarial(self, image):
        num_calls = 1
        step = 0.02
        perturbed = image
        while self.is_adversarial(perturbed) == -1:
            if self.args.budget / 10 <= num_calls <= self.args.budget / 10 + 2:
                perturbed = torch.ones(image.shape).to(self.device)
                if self.is_adversarial(perturbed) == -1:
                    perturbed = torch.zeros(image.shape).to(self.device)
            else:
                pert = torch.randn(image.shape).cuda()
                perturbed = torch.clamp(image + num_calls * step * pert, 0.0, 1.0)
                perturbed = perturbed.to(self.device)
                num_calls += 1
            sys.stdout.write(f'\rImg{self.Img_i} query{self.queries :.0f} \treal_d={torch.norm(perturbed):.6f} ')
            sys.stdout.flush()
        return perturbed, num_calls

    def bin_search(self, x_0, x_random):
        num_calls = 0
        adv = x_random
        cln = x_0
        while True:
            mid = (cln + adv) / 2.0
            num_calls += 1           
            if self.is_adversarial(mid)==1:
                adv = mid
            else:
                cln = mid
            if torch.norm(adv-cln).cpu().numpy()<self.tol or num_calls>=100:
                break       
        return adv, num_calls

    def normal_vector_approximation_batch(self, x0, q_max):
        random_noises = None
        if self.dim_reduc_factor < 1.0:
            raise Exception(
                "The dimension reduction factor should be greater than 1 for reduced dimension, and should be 1 for Full dimensional image space.")
        if self.dim_reduc_factor > 1.0:
            fill_size = int(x0.shape[-1] / self.dim_reduc_factor)
            random_noises = torch.zeros(q_max, int(x0.shape[-3]), int(x0.shape[-2]), int(x0.shape[-1]), dtype=torch.float32).cuda()
            for i in range(q_max):
                random_noises[i][ :, 0:fill_size, 0:fill_size] = torch.randn(x0.shape[0], x0.shape[1], fill_size, fill_size)
                random_noises[i] = torch.from_numpy(idct(idct(random_noises[i].cpu().numpy(), axis=2, norm='ortho'), axis=1, norm='ortho'))
        else:
            random_noises = torch.randn(q_max, self.x0.shape[-3], self.x0.shape[-2], self.x0.shape[-1], dtype=torch.float32).cuda()

        labels_out = []
        for i in range(q_max):
            labels_out.append(self.model.predict_label(x0 + self.sigma * random_noises[i]))
        self.queries += q_max

        z = []  # sign of grad_tmp
        for i, predict_label in enumerate(labels_out):
            if self.tar_img == None:
                if predict_label == self.x0_label:
                    z.append(-1)
                    random_noises[i] *= -1
                else:
                    z.append(1)
            if self.tar_img != None:
                if predict_label != self.tar_label:
                    z.append(-1)
                    random_noises[i] *= -1
                else:
                    z.append(1)
        normal_v = sum(random_noises)
        return normal_v, sum(z)/q_max

    def go_to_boundary_CGBA_H(self, x_s, eta_o, x_b):   
        num_calls = 1
        eta = eta_o/torch.norm(eta_o)
        v = (x_b - x_s)/torch.norm(x_b - x_s)
        theta = torch.acos(torch.dot(eta.reshape(-1), v.reshape(-1)))  
        while True:
            m = (torch.sin(theta)*torch.cos(theta/(pow(2, num_calls)))/torch.sin(theta/(pow(2,num_calls)))-torch.cos(theta)).item()
            zeta = (eta + m*v)/torch.norm(eta + m*v)
            perturbed = x_s + zeta*torch.norm(x_b-x_s)*torch.dot(zeta.reshape(-1), v.reshape(-1)) 
            perturbed = clip_image_values(perturbed, self.lb, self.ub)
            num_calls += 1
            if self.is_adversarial(perturbed) == 1:
                break
            if num_calls >=40:
                return x_b, num_calls
        perturbed, bin_query = self.bin_search(self.x0, perturbed)
        return perturbed, num_calls-1 + bin_query

    def go_to_boundary_CGBA(self, x_s, eta_o, x_b):
        num_calls = 1
        eta = eta_o/torch.norm(eta_o)
        v = (x_b - x_s)/torch.norm(x_b - x_s)
        theta = torch.acos(torch.dot(eta.reshape(-1), v.reshape(-1)))
        while True:
            m = (torch.sin(theta.cpu())*torch.cos(torch.tensor([(math.pi/2)])*(1 - 1/pow(2, num_calls)))/torch.sin(torch.tensor([(math.pi/2)])*(1 - 1/pow(2, num_calls)))-torch.cos(theta.cpu())).item()
            zeta = (eta + m*v)/torch.norm(eta + m*v)
            p_near_boundary = x_s + zeta*torch.norm(x_b-x_s)*torch.dot(v.reshape(-1), zeta.reshape(-1)) 
            p_near_boundary = clip_image_values(p_near_boundary, self.lb, self.ub)
            if self.is_adversarial(p_near_boundary) == -1:
                break
            num_calls += 1
            if num_calls>=40:
                return x_b, num_calls
        perturbed, n_calls = self.SemiCircular_boundary_search(x_s, x_b, p_near_boundary)
        return perturbed, num_calls+n_calls

    def get_attempt_unit_x(self, boundary_v_l2, boundary_v, normal_v, x0, k):
        attempt_v = (1-k) * boundary_v + k * normal_v
        attempt_l2 = torch.norm(attempt_v)
        attempt_v_unit = (boundary_v_l2/attempt_l2) * attempt_v
        attempt_x_unit = x0 + attempt_v_unit
        return attempt_x_unit

    def SemiCircular_boundary_search(self, x_0, x_b, p_near_boundary):
        num_calls = 0
        norm_dis = torch.norm(x_b-x_0)
        boundary_dir = (x_b-x_0)/torch.norm(x_b-x_0)
        clean_dir = (p_near_boundary - x_0)/torch.norm(p_near_boundary - x_0)
        adv_dir = boundary_dir
        adv = x_b.clone()
        clean = x_0
        while True:
            mid_dir = adv_dir + clean_dir
            mid_dir = mid_dir/torch.norm(mid_dir)
            theta = torch.acos(torch.dot(boundary_dir.reshape(-1), mid_dir.reshape(-1))/ (torch.linalg.norm(boundary_dir)*torch.linalg.norm(mid_dir)))
            d = torch.cos(theta)*norm_dis
            x_mid = x_0 + mid_dir*d
            num_calls +=1
            if self.is_adversarial(x_mid)==1:
                adv_dir = mid_dir
                adv = x_mid  
            else:
                clean_dir = mid_dir  
                clean = x_mid                             
            if torch.norm(adv-clean).cpu().numpy()<self.tol:
                break
            if num_calls >100:
                break      
        return adv, num_calls


    def attack(self):
        x_random, query_random = self.tar_img, 0
        if self.tar_img == None:
            x_random, query_random = self.find_init_adversarial(self.x0)
        x_boundary, query_b = self.bin_search(self.x0, x_random)
        best_adv_v = x_boundary - self.x0
        q_num = query_random + query_b
        norm_l2 = torch.norm(best_adv_v)
        sys.stdout.write(f'\rImg{self.Img_i} query{self.queries :.0f} \tIter{0 :.0f} \treal_d={norm_l2:.6f}')
        sys.stdout.flush()

        total_ratios = []
        for i in range(self.Max_iter):
            q_opt = int(self.Init_query * np.sqrt(i + 1))
            grad_oi, ratios = self.normal_vector_approximation_batch(x_boundary, q_opt)
            total_ratios.append(ratios)

            q_num = q_num + q_opt
            if self.attack_method == 'CGBA':
                x_boundary_new, qs = self.go_to_boundary_CGBA(self.x0, grad_oi, x_boundary)
            if self.attack_method == 'CGBA-H':
                x_boundary_new, qs = self.go_to_boundary_CGBA_H(self.x0, grad_oi, x_boundary)
            sim_cos = DataTools.cosine_similarity(x_boundary - self.x0, x_boundary_new - self.x0)
            x_boundary = x_boundary_new

            q_num = q_num + qs
            #assert self.queries == q_num
            best_adv_v = x_boundary - self.x0
            norm_l2 = torch.norm(best_adv_v)
            norm_linf = torch.max(torch.abs(best_adv_v))

            sys.stdout.write(f'\rImg{self.Img_i} query{self.queries :.0f} \tIter{i + 1:.0f} '
                             f'\treal_d={norm_l2:.6f} cos={sim_cos:.3f} \tNVn={ratios:.3f}')
            sys.stdout.flush()

            self.old_best_adv.RealL2 = min(torch.norm(self.x0).item(), norm_l2.item())
            self.old_best_adv.RealLinf = min(1, norm_linf.item())
            self.old_best_adv.ADBmax = min(torch.norm(self.x0).item(), norm_l2.item())
            self.L2_line.append([self.queries, self.old_best_adv.RealL2])
            self.Linf_line.append([self.queries, self.old_best_adv.RealLinf])
            if self.queries >= self.args.budget:
                break
            if self.success == -1 and self.old_best_adv.RealL2 <= self.args.epsilon:
                self.success = 1
                self.ACCquery, self.ACCiter_n = self.queries, i
            if self.args.early == 1 and self.success == 1:
                break

        advimg = 0.5 * (1.0 + best_adv_v / self.old_best_adv.RealLinf)
        self.Img_result = [advimg, self.x0, self.x0 + best_adv_v]
        self.heatmaps = self.Img_result
        self.File_string = (str(self.args.dataset) + ",Img" + str(self.Img_i) + ",I-Q[" + str(self.ACCiter_n) + "-" +
                            str(self.queries) + "], ADB{:.4f}".format(self.old_best_adv.ADBmax))
        print(f' --AVG_ra={sum(total_ratios)/len(total_ratios):.3f}', end = "")
        return
    
  
    
  
    
  
    
  
