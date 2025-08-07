import os, multiprocessing
# os.environ["OMP_NUM_THREADS"] = str(multiprocessing.cpu_count())
# os.environ["MKL_NUM_THREADS"] = str(multiprocessing.cpu_count())
import sys
sys.argv = [sys.argv[0]] 
import pandas as pd
import random
import torch
import argparse
import numpy as np
from pathlib import Path
import logging
import wandb
from tqdm import tqdm
import time
import json

from utils.Hyperopt import Hyperoptimizer
from utils.logger import LoggerMixin, init_fancy_logging
from exp.exp_pop_pre_train import Exp_Pop_Main
from exp.exp_patient_specific import Exp_PS_Main
from exp.exp_data_specific import Exp_DS_Main
from exp.exp_xgboost import Exp_xgboost
from exp.exp_test import Exp_Test
from resp_db.client import RpmDatabaseClient
from main_configs import Benchmark_args



init_fancy_logging(log_file = "/mnt/nas-wang/nas-ssd/Results/RMP/log/PsExp.log")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

hyperopt_search_dict = {
        "model": {'LSTM', 'MLP', 'TransformerTSFv2', 'DLinear','XGBoostTSF'},
        "project": {'PL','PS-4DCT', 'PS', 'test'},
        "pred_len": {6, 12, 18, 24}
    }

def get_pids(args):
        client = RpmDatabaseClient(db_filepath=args.db_root)
        with client:
                raw_pairs = client.get_pids_fraction_of_test_dataset(project='test')
                pairs = list(raw_pairs)
        return pairs

if __name__ == "__main__":
    #### Setp 1: Load configs
    args_parser = Benchmark_args()
    args = args_parser.parse_args()
    fix_seed = args.random_seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed) 

    start_time = time.time()
    #### Step 2: hyper-parameters optimaztion. 遍历models和pred_len.
    args.project = 'PS-4DCT' # 'FT-4DCT', 'patient-specific'
    project_copy = args.project # 'FT-4DCT', 'patient-specific'

    args.is_fine_tuning = False
    is_use_pre_train_model = False
    learning_rate_fine_tuning = 0.1
    args.model = 'DLinear' # 'LSTM', 'MLP', 'TransformerTSFv2', 'DLinear','XGBoostTSF'
    args.pred_len= int(6) # 6, 12, 18 ,24

    args.patience = 10
    train_epochs = 200
    max_trials = 50 
    max_epochs = 50
    pids = get_pids(args)
    settings = '{}_pl{}'.format(
                args.model,
                args.pred_len)
    use_wandb = False
    try:
        project_name = args.project + ("_FT" if args.is_fine_tuning else "")
        wandb.init(
            project=f"{project_name}",
            dir= os.path.join(args.checkpoints, args.project + ("_fine_tuning" if args.is_fine_tuning else ""), settings),
            anonymous="allow",
            name=f"{project_name}_{settings}",
        )
        use_wandb = True
    except wandb.errors.Error:
        logger.warning(
            "Cannot import weights and biases. Consider creating a free account and login for advanced results tracking!"
        )

    rmse_list = []
    re_rmse_list = []
    inference_time_list = []
    training_cost_time_list = []
    patient_re_rmse_list = []
    vali_loss_list = []
    model_params_list = []
    macs_list = []
    params_list = []
    hyper_para_list = []
    patient_id_list = []

    for i, (patient_id, f_num) in enumerate(tqdm(pids, total=len(pids), ncols=100, position=0)):
        percent = int((i + 1) / len(pids) * 100)
        args.patient_id = patient_id
        args.f_num = f_num
        pid_settings = 'pid{}_fnum{}'.format(
                args.patient_id,
                args.f_num)
        tqdm.write(f"Step {i+1}/{len(pids)} ({percent}%) - Patient ID: {patient_id}")
        logger.info(f"----------- Start building {args.project} model for {pid_settings} ---------")
        if not args.is_fine_tuning:
            logger.info(f"Starting hyperopt searching")
            logger.info(f"Working space {os.path.join(args.checkpoints, args.project)}")
            hyperopt = Hyperoptimizer(args,settings)
            result_dir = os.path.join(args.checkpoints, args.project, settings, pid_settings)
            logger.info(f"-------------------------------------")
            logger.info(f"Current patient id {patient_id}")
            logger.info(f"Current model and prediction length {settings}")
            # max_epochs = args.train_epochs # except XGBoostTSF
            hyper_para_1 = hyperopt.run_trials(result_dir,max_trials,max_epochs)
            best_trial = hyperopt.read_hyperopt_object(result_dir, settings)
            hyper_para = best_trial['result']['applied_hyper_paras']
            training_cost_time_trials = best_trial['result']['training_cost_time']
            vali_loss_trials = best_trial['result']['loss']
            logger.info(f"Best hyper-parameters: {hyper_para}")
            logger.info(f"-------------------------------------")
            current_args = hyperopt._merge_hyperparameters(args, hyper_para) # Apply optimzed parameters
            current_args.is_save_checkpoint = True
            current_args.train_epochs = train_epochs
            ps_exp = Exp_PS_Main(current_args)
            logger.info(f"------------ End of hyperopt optimaztion ------------------")
            logger.info(f"------------ Start of training patient-specific model ------------------")
            trained_model, vali_loss , training_cost_time = ps_exp.train(settings)
            patient_id, rmse , relative_rmse, avg_inference_time, model_params, macs, params  = ps_exp.test(settings, model=trained_model)
            logger.info("=== Trials Summary ===")
            logger.info(f"Min vali_loss in trials: {vali_loss_trials:.4f}")
            logger.info(f"Best vali_loss (final): {vali_loss:.4f}")
            logger.info(f"Min training_cost_time in trials: {training_cost_time_trials:.2f} sec")
            logger.info(f"Best training_cost_time (final): {training_cost_time:.2f} sec")
        else:
            logger.info(f"Starting fine tuning")
            logger.info(f"Current patient id {patient_id}")
            logger.info(f"Current model and prediction length {settings}")
            if is_use_pre_train_model:
                pre_trained_path = os.path.join(args.checkpoints, 'PL' , settings)
                hyperopt = Hyperoptimizer(args,settings)
                best_trial = hyperopt.read_hyperopt_object(pre_trained_path, settings)
                hyper_para = best_trial['result']['applied_hyper_paras']
                current_args = hyperopt._merge_hyperparameters(args, hyper_para) # Apply optimzed parameters
                
                load_model_path = os.path.join(pre_trained_path, 'checkpoint.pth')
                logger.info(f"------------ Loading pre-trained PL model from {pre_trained_path}------------ ")
            else:
                logger.info(f"------------ Retraining pre-trained PL model using optimized paras of PS ------------ ")
                hyperopt = Hyperoptimizer(args,settings)
                result_dir = os.path.join(args.checkpoints, args.project, settings, pid_settings)
                best_trial = hyperopt.read_hyperopt_object(result_dir, settings)
                hyper_para = best_trial['result']['applied_hyper_paras']
                training_cost_time_trials = best_trial['result']['training_cost_time']
                vali_loss_trials = best_trial['result']['loss']
                logger.info(f"Best hyper-parameters: {hyper_para}")
                logger.info(f"-------------------------------------")
                current_args = hyperopt._merge_hyperparameters(args, hyper_para) # Apply optimzed parameters
                current_args.is_save_checkpoint = False
                current_args.train_epochs = train_epochs
                current_args.patience = 20
                current_args.project = "PL"
                pre_train_pl_exp = Exp_Pop_Main(current_args)
                pre_trained_model, vali_loss, training_cost_time = pre_train_pl_exp.train(settings)
                load_model_path = os.path.join(result_dir, 'pre_trained_checkpoint.pth')
                torch.save(pre_trained_model, load_model_path)
                # with open(result_dir + '/' + 'pre_trained_checkpoint.pth', 'wb') as f:
                #     torch.save(pre_trained_model, f)
                #     f.flush()                  
                #     os.fsync(f.fileno()) 
                logger.info(f"------------ End of pre training PL model ------------ ")
                current_args.project = project_copy

            logger.info(f"------------ Start of fine-tuning ------------ ")
            current_args.learning_rate = current_args.learning_rate * learning_rate_fine_tuning
            current_args.is_save_checkpoint = True
            current_args.train_epochs = train_epochs
            current_args.patience = 20
            ps_exp = Exp_PS_Main(current_args)
            fine_tuned_model, vali_loss , training_cost_time = ps_exp.train(settings, load_model_path = load_model_path)
            patient_id, rmse , relative_rmse, avg_inference_time, model_params, macs, params  = ps_exp.test(settings, model=fine_tuned_model)
            logger.info(f"------------ End of fine-tuning testing ------------------")
        
        cost_time_each_patient = time.time() - start_time
        logger.info(f"Total cost time for each patient: {cost_time_each_patient}")
        print(f"Patient ID: {patient_id}, RMSE: {rmse:.4f}, Relative RMSE: {relative_rmse:.4f}, Avg Inference Time: {avg_inference_time:.4f}s")
        logger.info(f"Patient ID: {patient_id}, RMSE: {rmse:.4f}, Relative RMSE: {relative_rmse:.4f}, Avg Inference Time: {avg_inference_time:.4f}s")
        
        wandb.log({"percent":percent,
                   "pid": patient_id,
                   "total_cost_time": training_cost_time,
                   "best_vali_loss":vali_loss,
                   "inference_time":avg_inference_time})
        patient_re_rmse_list.append((patient_id,relative_rmse))
        rmse_list.append(rmse)
        re_rmse_list.append(relative_rmse)
        inference_time_list.append(avg_inference_time)
        training_cost_time_list.append(training_cost_time)
        vali_loss_list.append(vali_loss)
        patient_id_list.append(patient_id)
        model_params_list.append(model_params)
        macs_list.append(macs)
        params_list.append(params)
        hyper_para_list.append(hyper_para.copy())
        
    print("\n======== Plotting statistical results  ========")
    logger.info("\n======== Plotting statistical results ========")
    test_result_folder_path = os.path.join(args.checkpoints, args.project  + ("_fine_tuning" if args.is_fine_tuning else ""), settings, 'test_results')
    rmse_path = os.path.join(test_result_folder_path, "rmse_list.npy")
    re_rmse_path = os.path.join(test_result_folder_path, "re_rmse_list.npy")
    np.save(rmse_path, np.array(rmse_list))
    np.save(re_rmse_path, np.array(re_rmse_list))

    inference_time_mean = np.mean(inference_time_list)
    std_inference_time = np.std(inference_time_list)
    vali_loss_mean = np.mean(vali_loss_list)
    training_cost_time_mean = np.mean(training_cost_time_list)
    rmse_mean = np.mean(rmse_list)
    rmse_std = np.std(rmse_list)
    rmse_median = np.median(rmse_list)
    rmse_q1, rmse_q3 = np.percentile(rmse_list, [25, 75])

    re_rmse_mean = np.mean(re_rmse_list)
    re_rmse_std = np.std(re_rmse_list)
    re_rmse_median = np.median(re_rmse_list)
    re_rmse_q1, re_rmse_q3 = np.percentile(re_rmse_list, [25, 75])

    sorted_by_re_rmse = sorted(patient_re_rmse_list, key=lambda x: x[1])  
    top5_best = sorted_by_re_rmse[:5]
    top5_worst = sorted_by_re_rmse[-5:][::-1]

    suffix = "_fine_tuning" if args.is_fine_tuning else ""
    result_path = os.path.join(test_result_folder_path, f"result{suffix}.txt")
    with open(result_path, 'a') as f:
        print("\n=== Inference time Statistics ===")
        print(f'Average inference time {avg_inference_time}')
        print(f'Standard devation inference time {std_inference_time}')

        print("\n=== Patient-wise RMSE Statistics ===")
        print(f"Mean ± Std: {rmse_mean:.7f} ± {rmse_std:.7f}")
        print(f"Median: {rmse_median:.7f}")
        print(f"IQR: [{rmse_q1:.7f}, {rmse_q3:.7f}]")

        print("\n=== Patient-wise Relative RMSE Statistics ===")
        print(f"Mean ± Std: {re_rmse_mean:.7f} ± {re_rmse_std:.7f}")
        print(f"Median: {re_rmse_median:.7f}")
        print(f"IQR: [{re_rmse_q1:.7f}, {re_rmse_q3:.7f}]")

        f.write("\n=== Inference time Statistics ===\n")
        f.write(f"Average: {avg_inference_time:.7f} ms\n")  
        f.write(f"Std Dev: {std_inference_time:.7f} ms\n")
        f.write(f"Mean training cost time: {training_cost_time_mean:.2f} s\n")
        logger.info(f'Mean inference time {avg_inference_time}')
        logger.info(f'Mean training cost time: {training_cost_time_mean}')

        f.write("\n=== Patient-wise RMSE Statistics ===\n")
        f.write(f"Mean ± Std: {rmse_mean:.7f} ± {rmse_std:.7f}\n")
        f.write(f"Median: {rmse_median:.7f}\n")
        f.write(f"IQR: [{rmse_q1:.7f}, {rmse_q3:.7f}]\n\n")
        logger.info(f'Patient-wise RMSEe Mean ± Std: {rmse_mean:.7f} ± {rmse_std:.7f}\n')

        f.write("=== Patient-wise Relative RMSE Statistics ===\n")
        f.write(f"Mean ± Std: {re_rmse_mean:.7f} ± {re_rmse_std:.7f}\n")
        f.write(f"Median: {re_rmse_median:.7f}\n")
        f.write(f"IQR: [{re_rmse_q1:.7f}, {re_rmse_q3:.7f}]\n\n")
        logger.info(f'Patient-wise Relative RMSE Mean ± Std: {re_rmse_mean:.7f} ± {re_rmse_std:.7f}\n')

        print("\nTop 5 best patients (lowest relative RMSE):")
        f.write("=== Top 5 Patients (Lowest Relative RMSE) ===\n")
        logger.info("=== Top 5 Patients (Lowest Relative RMSE) ===")
        for i, (pid, rermse) in enumerate(top5_best, 1):
            print(f"# {i}, Patient ID: {pid}, Relative RMSE: {rermse:.4f}")
            f.write(f"# {i}, Patient ID: {pid}, Relative RMSE: {rermse:.7f}\n")
            logger.info(f"# {i}, Patient ID: {pid}, Relative RMSE: {rermse:.7f}")

        print("\nWorst 5 patients (highest relative RMSE):")
        f.write("\n=== Worst 5 Patients (Highest Relative RMSE) ===\n")
        logger.info("=== Worst 5 Patients (Highest Relative RMSE) ===")
        for i, (pid, rermse) in enumerate(top5_worst, 1):
            print(f"# {i}, Patient ID: {pid}, Relative RMSE: {rermse:.4f}")
            f.write(f"# {i}, Patient ID: {pid}, Relative RMSE: {rermse:.7f}\n")
            logger.info(f"# {i}, Patient ID: {pid}, Relative RMSE: {rermse:.7f}")

    result_json_dir = os.path.join(args.checkpoints, args.project, "result.json")
    result_data = {
        "project": args.project,
        "settings": settings,
        "hyper_para":hyper_para_list,
        "patient_id":patient_id_list,
        "rmse_mean": rmse_mean,
        "re_rmse_mean": re_rmse_mean,
        "training_cost_time":training_cost_time_mean,
        "avg_inference_time": avg_inference_time,
        "model_params": model_params_list,
        "macs": macs_list,
        "params":params_list
    }
    try:
        with open(result_json_dir, 'r') as f:
            all_results = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        all_results = {"experiments": []}  

    all_results["experiments"].append(result_data)

    with open(result_json_dir, 'w') as f:
        json.dump(all_results, f, indent=4)
        

    wandb.log({
            "rmse_mean": rmse_mean,
            "re_rmse_mean":re_rmse_mean,
            "inference_time": inference_time_mean,
            "vali_loss":vali_loss_mean,
            "training_cost_time":training_cost_time_mean,}, commit=True)
    logger.info(f"-------------------End testing-------------------------")
    print(f"-------------------End testing-------------------------")
    if use_wandb:
        wandb.finish()


    # ### Step 2: hyper-parameters optimaztion. 遍历models和pred_len.
    # args.is_save_checkpoint = False
    # args.is_fine_tuning = True
    # use_wandb = False
    # try:
    #     project_name = args.project + ("_fine_tuning" if args.is_fine_tuning else "")
    #     wandb.init(
    #         project=f"{args.project}",
    #         dir= os.path.join(args.checkpoints, args.project + ("_fine_tuning" if args.is_fine_tuning else ""), settings),
    #         anonymous="allow",
    #         name=f"{project_name}_{settings}",
    #     )
    #     use_wandb = True
    # except wandb.errors.Error:
    #     logger.warning(
    #         "Cannot import weights and biases. Consider creating a free account and login for advanced results tracking!"
    #     )

    # rmse_list = []
    # re_rmse_list = []
    # inference_time_list = []
    # training_cost_time_list = []
    # patient_re_rmse_list = []
    # vali_loss_list = []
    # for i, (patient_id, f_num) in enumerate(tqdm(pids, total=len(pids), ncols=100, position=0)):
    #     percent = int((i + 1) / len(pids) * 100)
    #     args.patient_id = patient_id
    #     args.f_num = f_num
    #     pid_settings = 'pid{}_fnum{}'.format(
    #             args.patient_id,
    #             args.f_num)
        
    #     logger.info(f"----------- Start building model for {pid_settings} ---------")
    #     if not args.is_fine_tuning:
    #         logger.info(f"Starting hyperopt searching")
    #         logger.info(f"Working space {os.path.join(args.checkpoints, args.project)}")
            
    #         hyperopt = Hyperoptimizer(args,settings)
    #         result_dir = os.path.join(args.checkpoints, args.project, settings, pid_settings)
    #         logger.info(f"-------------------------------------")
    #         logger.info(f"Current patient id {patient_id}")
    #         logger.info(f"Current model and prediction length {settings}")
    #         # max_epochs = args.train_epochs # except XGBoostTSF
    #         hyper_para = hyperopt.run_trials(result_dir,max_trials,max_epochs)
    #         best_trial = hyperopt.read_hyperopt_object(result_dir, settings)
    #         hyper_para = best_trial['result']['applied_hyper_paras']
    #         training_cost_time = best_trial['result']['training_cost_time']
    #         vali_loss = best_trial['result']['loss']
    #         logger.info(f"Best hyper-parameters: {hyper_para}")
    #         logger.info(f"-------------------------------------")
    #         current_args = hyperopt._merge_hyperparameters(args, hyper_para) # Apply optimzed parameters
    #         current_args.is_save_checkpoint = True
    #         current_args.train_epochs = train_epochs
    #         ps_exp = Exp_PS_Main(current_args)
    #         trained_model, vali_loss , training_cost_time = ps_exp.train(settings)
    #         patient_id, rmse , relative_rmse, avg_inference_time = ps_exp.test(settings, model=trained_model)
    #     else:
    #         pre_trained_path = os.path.join(args.checkpoints, 'population-level' , settings)
    #         hyperopt = Hyperoptimizer(args,settings)
    #         best_trial = hyperopt.read_hyperopt_object(pre_trained_path, settings)
    #         hyper_para = best_trial['result']['applied_hyper_paras']
    #         current_args = hyperopt._merge_hyperparameters(args, hyper_para) # Apply optimzed parameters
    #         current_args.learning_rate = current_args.learning_rate * 0.5
    #         load_model_path = os.path.join(pre_trained_path + '/checkpoint.pth')
    #         current_args.is_save_checkpoint = True
    #         current_args.train_epochs = train_epochs
    #         current_args.patience = 20
    #         ps_exp = Exp_PS_Main(current_args)
    #         fine_tuned_model, vali_loss , training_cost_time = ps_exp.train(settings, load_model_path = load_model_path)
    #         patient_id, rmse , relative_rmse, avg_inference_time = ps_exp.test(settings, model=fine_tuned_model)
    #     cost_time_each_patient = time.time() - start_time
    #     logger.info(f"Total cost time for each patient: {cost_time_each_patient}")
    #     print(f"Patient ID: {patient_id}, RMSE: {rmse:.4f}, Relative RMSE: {relative_rmse:.4f}, Avg Inference Time: {avg_inference_time:.4f}s")
    #     logger.info(f"Patient ID: {patient_id}, RMSE: {rmse:.4f}, Relative RMSE: {relative_rmse:.4f}, Avg Inference Time: {avg_inference_time:.4f}s")
    #     wandb.log({"percent":percent,
    #                "pid": patient_id,
    #         "total_cost_time": training_cost_time,
    #         "best_vali_loss":vali_loss,
    #         "inference_time":avg_inference_time})
    #     patient_re_rmse_list.append((patient_id,relative_rmse))
    #     rmse_list.append(rmse)
    #     re_rmse_list.append(relative_rmse)
    #     inference_time_list.append(avg_inference_time)
    #     training_cost_time_list.append(training_cost_time)
    #     vali_loss_list.append(vali_loss)

    # print("\n======== Plotting statistical results  ========")
    # logger.info("\n======== Plotting statistical results ========")
    # test_result_folder_path = os.path.join(args.checkpoints, args.project + ("_fine_tuning" if args.is_fine_tuning else ""), settings, 'test_results')
    # rmse_path = os.path.join(test_result_folder_path, "rmse_list.npy")
    # re_rmse_path = os.path.join(test_result_folder_path, "re_rmse_list.npy")
    # np.save(rmse_path, np.array(rmse_list))
    # np.save(re_rmse_path, np.array(re_rmse_list))

    # inference_time_mean = np.mean(inference_time_list)
    # std_inference_time = np.std(inference_time_list)
    # vali_loss_mean = np.mean(vali_loss_list)
    # training_cost_time_mean = np.mean(training_cost_time_list)
    # rmse_mean = np.mean(rmse_list)
    # rmse_std = np.std(rmse_list)
    # rmse_median = np.median(rmse_list)
    # rmse_q1, rmse_q3 = np.percentile(rmse_list, [25, 75])

    # re_rmse_mean = np.mean(re_rmse_list)
    # re_rmse_std = np.std(re_rmse_list)
    # re_rmse_median = np.median(re_rmse_list)
    # re_rmse_q1, re_rmse_q3 = np.percentile(re_rmse_list, [25, 75])

    # sorted_by_re_rmse = sorted(patient_re_rmse_list, key=lambda x: x[1])  
    # top5_best = sorted_by_re_rmse[:5]
    # top5_worst = sorted_by_re_rmse[-5:][::-1]

    # suffix = "_fine_tuning" if args.is_fine_tuning else ""
    # result_path = os.path.join(test_result_folder_path, f"result{suffix}.txt")
    # with open(result_path, 'a') as f:
    #     print("\n=== Inference time Statistics ===")
    #     print(f'Average inference time {avg_inference_time}')
    #     print(f'Standard devation inference time {std_inference_time}')

    #     print("\n=== Patient-wise RMSE Statistics ===")
    #     print(f"Mean ± Std: {rmse_mean:.7f} ± {rmse_std:.7f}")
    #     print(f"Median: {rmse_median:.7f}")
    #     print(f"IQR: [{rmse_q1:.7f}, {rmse_q3:.7f}]")

    #     print("\n=== Patient-wise Relative RMSE Statistics ===")
    #     print(f"Mean ± Std: {re_rmse_mean:.7f} ± {re_rmse_std:.7f}")
    #     print(f"Median: {re_rmse_median:.7f}")
    #     print(f"IQR: [{re_rmse_q1:.7f}, {re_rmse_q3:.7f}]")

    #     f.write("\n=== Inference time Statistics ===\n")
    #     f.write(f"Average: {avg_inference_time:.7f} ms\n")  
    #     f.write(f"Std Dev: {std_inference_time:.7f} ms\n")
    #     logger.info(f'Average Inference time {avg_inference_time}')

    #     f.write("\n=== Patient-wise RMSE Statistics ===\n")
    #     f.write(f"Mean ± Std: {rmse_mean:.7f} ± {rmse_std:.7f}\n")
    #     f.write(f"Median: {rmse_median:.7f}\n")
    #     f.write(f"IQR: [{rmse_q1:.7f}, {rmse_q3:.7f}]\n\n")
    #     logger.info(f'Patient-wise RMSEe Mean ± Std: {rmse_mean:.7f} ± {rmse_std:.7f}\n')

    #     f.write("=== Patient-wise Relative RMSE Statistics ===\n")
    #     f.write(f"Mean ± Std: {re_rmse_mean:.7f} ± {re_rmse_std:.7f}\n")
    #     f.write(f"Median: {re_rmse_median:.7f}\n")
    #     f.write(f"IQR: [{re_rmse_q1:.7f}, {re_rmse_q3:.7f}]\n\n")
    #     logger.info(f'Patient-wise Relative RMSE Mean ± Std: {re_rmse_mean:.7f} ± {re_rmse_std:.7f}\n')

    #     print("\nTop 5 best patients (lowest relative RMSE):")
    #     f.write("=== Top 5 Patients (Lowest Relative RMSE) ===\n")
    #     logger.info("=== Top 5 Patients (Lowest Relative RMSE) ===")
    #     for i, (pid, rermse) in enumerate(top5_best, 1):
    #         print(f"# {i}, Patient ID: {pid}, Relative RMSE: {rermse:.4f}")
    #         f.write(f"# {i}, Patient ID: {pid}, Relative RMSE: {rermse:.7f}\n")
    #         logger.info(f"# {i}, Patient ID: {pid}, Relative RMSE: {rermse:.7f}")

    #     print("\nWorst 5 patients (highest relative RMSE):")
    #     f.write("\n=== Worst 5 Patients (Highest Relative RMSE) ===\n")
    #     logger.info("=== Worst 5 Patients (Highest Relative RMSE) ===")
    #     for i, (pid, rermse) in enumerate(top5_worst, 1):
    #         print(f"# {i}, Patient ID: {pid}, Relative RMSE: {rermse:.4f}")
    #         f.write(f"# {i}, Patient ID: {pid}, Relative RMSE: {rermse:.7f}\n")
    #         logger.info(f"# {i}, Patient ID: {pid}, Relative RMSE: {rermse:.7f}")
            
    # wandb.log({
    #         "rmse_mean": rmse_mean,
    #         "re_rmse_mean":re_rmse_mean,
    #         "inference_time": inference_time_mean,
    #         "vali_loss":vali_loss_mean,
    #         "training_cost_time":training_cost_time_mean,}, commit=True)
    # logger.info(f"-------------------End testing-------------------------")
    # print(f"-------------------End testing-------------------------")
    # if use_wandb:
    #     wandb.finish()