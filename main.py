# coding:utf-8
import DataTools
import ADBA
import SeRI
import CGBA
import os
from datetime import datetime
import argparse
import line_profiler as lp
import torch
import random
import dataset

bridge_bestKs = []
def main():
    profile = lp.LineProfiler()
    # ###################################################################################
    torch_model, test_loader = None, None
    parser = argparse.ArgumentParser(description='Hard Label Attacks')
    parser.add_argument('--dataset', default='mnsit-cnn', type=str, help='Dataset')
    parser.add_argument('--targeted', default=0, type=int, help='targeted-1 or untargeted-0')
    parser.add_argument('--norm', default='CGBA', type=str, help='Norm for attack, k or l2')
    parser.add_argument('--zip', default='SeRI', type=str, help='Decision-based Sensitive Region Aware noize zip approach')
    parser.add_argument('--epsilon', default=0.3, type=float, help='attack strength')
    parser.add_argument('--budget', default=1600, type=int, help='Maximum query for the attack norm k')
    parser.add_argument('--budget2', default=400, type=int, help='Maximum query for the Sensitive Region Aware')
    parser.add_argument('--early', default=0, type=int, help='early stopping (stop attack once the adversarial example is found)')
    parser.add_argument('--remember', default=0, type=int, help='if remember adversarial examples.')
    parser.add_argument('--imgnum', default=5, type=int, help='Number of samples to be attacked from test dataset.')
    parser.add_argument('--beginIMG', default=0, type=int, help='begin test img number')
    parser.add_argument('--RGB_mod', default=0, type=int, help='0 for x,y; 1 for channels,x,y')
    parser.add_argument('--RGB', default='RGB', type=str, help='List of RGB channels (e.g., RG)')
    parser.add_argument('--binaryM', default=1, type=int, help='binary search mod, mid 0 or median 1.')
    parser.add_argument('--initSign', default=1, type=int, help='initial adv sign, 1,-1,and 0 for random, 2 for 1-11-1...')
    parser.add_argument('--RandResizePad', default=0, type=int, help='if use RandResizePad defense model')
    parser.add_argument('--Folder', default='Results', type=str, help='Result files folder Name')
    parser.add_argument('--k1', default=0.2, type=float, help='perturbation enhancement k1')
    parser.add_argument('--k2', default=1.8, type=float, help='perturbation enhancement k2')

    args = parser.parse_args()
    print(args)

    Folder_name =  str(args.dataset) + "_" + str(args.norm) + "_" + str(args.zip) + "_aimR" + str(args.epsilon) +\
                        "_target" + str(args.targeted) +\
                        "_budget" + str(args.budget) +\
                        "_md" + str(args.binaryM) +\
                        "_Early" + str(args.early) +\
                        "_IMG" + str(args.beginIMG) + "+" + str(args.imgnum) +\
                        "_T" + str(datetime.now().strftime("%H-%M-%S"))
    if not os.path.exists("results_record/" + Folder_name):
        os.makedirs("results_record/" + Folder_name)
    args.Folder = Folder_name
    # ###############################################################################
    test_loader, torch_model = dataset.load_dataset_model(args.dataset, args)
    Out = dataset.OutResult(args, Folder_name)

    Attacker = None
    Attacker2 = None
    for imgi, (original_image, label) in enumerate(test_loader):
        if Out.ImgNum_origin_right >= args.imgnum:
            break
        if imgi < args.beginIMG:
            continue
        Out.ImgNum_total_tested = Out.ImgNum_total_tested + 1
        out_label = torch_model.predict_label(original_image.cuda()).cpu().item()
        real_label = label.item()
        if out_label == real_label:
            ATK_target_xi, ATK_target_yi = None, None
            if args.targeted == 1:
                random_index = random.randint(0, len(test_loader) - 1)
                ATK_target_xi, ATK_target_yi = test_loader.dataset[random_index]
                ATK_target_xi = ATK_target_xi.cuda()
                ATK_target_F = torch_model.predict_label(ATK_target_xi).cpu()
                while label.item() == ATK_target_yi or ATK_target_F.item() != ATK_target_yi:
                    random_index = random.randint(0, len(test_loader) - 1)
                    ATK_target_xi, ATK_target_yi = test_loader.dataset[random_index]
                    ATK_target_xi = ATK_target_xi.cuda()
                    ATK_target_F = torch_model.predict_label(ATK_target_xi).cpu()

            if args.norm == "ADBA":
                Attacker = ADBA.Attacker(torch_model, args)
                Attacker.attack(x=original_image.cuda(), y=label.cuda(), Img_i=imgi, target=None, query_limit=args.budget)
            elif args.norm == "CGBA":
                Attacker = CGBA.Attacker(args, torch_model, original_image, imgi, ATK_target_xi, ATK_target_yi, "CGBA")
                Attacker.attack()
            elif args.norm == "CGBA-H":
                Attacker = CGBA.Attacker(args, torch_model, original_image, imgi, ATK_target_xi, ATK_target_yi, "CGBA-H")
                Attacker.attack()
            elif args.norm == "SeRI":
                Attacker = SeRI.Attacker(args, torch_model, original_image, imgi, label, ATK_target_xi, ATK_target_yi, Attacker)
                Attacker.attack()
            else:
                print("norm is wrong: "+args.norm)
                return

            print("")
            if args.budget2!=0 and args.zip != "NA":
                if args.zip == "SeRI":
                    Attacker = SeRI.Attacker(args, torch_model, original_image, imgi, label, ATK_target_xi, ATK_target_yi, Attacker)
                    Attacker.attack()
                else:
                    print("zip approach is wrong: " + args.zip)
                    return

            Out.add1Result(Attacker)
            Out.Summary()
            #if Out.ImgNum_origin_right %10 == 1:
            #    Out.Summary()
            if args.remember == 1:
                combined_file = DataTools.save_images([Attacker.Img_result, Attacker.heatmaps], "results_record/" + Folder_name, Attacker.File_string)
            print("")
        else:
            print(f"IMG{imgi} Originally classify incorrect")
    Out.Summary()
    print(args)
    print(f"NATURAL ACCURACY RATE={Out.NATURAL_ACCURACY_RATE:.4f}")
    print(f"ATTACK SUCCESS RATE = {Out.ATTACK_SUCCESS_RATE:.4f}")
    print(f"ROBUST ACCURACY RATE={Out.ROBUST_ACCURACY_RATE:.4f}")
    print(f"AVG(MID)-AccQuery  = {Out.AccQuery_avg:.1f}({Out.AccQuery_mid:.1f})")
    print(f"FINAL AVG(MID)-l2 = {torch.mean(Out.L2_LINE_sum[:, -1]).item():.3f}"
          f"({torch.median(Out.L2_LINE_sum[:, -1]).item():.3f})")
    query = [int(args.budget/5), int(args.budget/2), int(args.budget)]
    for q in query:
        print(f"AVG(MID)-l2 after {q} queries : {torch.mean(Out.L2_LINE_sum[:, q]).item():.3f}"
              f"({torch.median(Out.L2_LINE_sum[:, q]).item():.3f})")
    """"""
    print(f"l2(linf)-midAUC = {Out.AUC_l2:.1f}({Out.AUC_linf:.1f})")
    print(f"AVG(MID)-ADB = {Out.EndADB_avg}({Out.EndADB_mid})")
    print(f"AVG(MID)-linf= {Out.Endlinf_avg}()")
    print(f"AVG-Kbridge= {sum(Out.Kbridge)/(len(Out.Kbridge)+0.01)}")

if __name__ == "__main__":
    main()