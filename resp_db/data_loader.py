from __future__ import annotations

import logging
from pathlib import Path
from random import randrange
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
from typing import Union, Tuple
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import time
from torch.utils.data import Subset
from resp_db.client import RpmDatabaseClient
from resp_db.common_types import PathLike
from utils.logger import LoggerMixin, init_fancy_logging
from resp_db.time_series_utils import add_white_noise_to_signal, fourier_smoothing, time_series_scaling
torch.multiprocessing.set_sharing_strategy('file_system')

class RpmSignals(Dataset, LoggerMixin):
    def __init__(
        self,
        db_root: PathLike,
        project: str,
        set_name: str | None = None,
        fourier_smoothing_hz: int | None = 1,
        white_noise_db: int | None = 30,
        sampling_rate_hz: int = 25,
        apply_denoise: bool = True,
        apply_noise: bool = True,
        apply_scaling: bool = True,
        scaling_period_s: tuple[float, float] | None = (0, 20),
        seq_len: int = 50,
        label_len: int = 0,
        pred_len: int = 0,
        only_beam_on: bool = False,
        remove_offset: bool = False,
        min_length_s: int = 1,
    ):
        super().__init__()
        self.db_root = Path(db_root)
        self.white_noise_db = white_noise_db
        self.fourier_cutoff = fourier_smoothing_hz
        self.sampling_rate = sampling_rate_hz
        self.apply_denoise = apply_denoise
        self.apply_noise = apply_noise
        self.apply_scaling = apply_scaling
        self.scaling_period_s = scaling_period_s
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.only_beam_on = only_beam_on
        self.remove_offset = remove_offset
        self.project = project
        self.set_name = set_name

        self.sequence_infos = []  # list of (signal, start_index)
        client = RpmDatabaseClient(db_filepath=db_root)

        if not set_name and self.project == "test":
            with client:
                query = client.get_all_project_signals_of_dl_dataset(
                    project=project
                )
        else:
            with client:
                query = client.get_signals_of_dl_dataset(
                    dl_dataset=set_name, project=project
                )
        len_unfiltered = len(query)
        self.query = []
        for signal in query:
            df_signal = client.preprocess_signal(
                signal.df_signal,
                sampling_rate=self.sampling_rate,
                only_beam_on=self.only_beam_on,
                remove_offset=self.remove_offset,
            )
            if self.apply_scaling:
                df_signal.amplitude, scaler = time_series_scaling(
                time_series=df_signal.amplitude.values,
                feature_range=(-1, 1),
                scaler="MinMax",
                scaling_period_s=self.scaling_period_s,
                return_scaler=True,
                )
            #if df_signal.time.max() > min_length_s:
            self.query.append((signal,df_signal))

        if self.project in ["PL", "PS", "test" , "PS-4DCT"]:
            for signal, df_signal in self.query:  
                max_index = len(df_signal.amplitude.values) - (self.seq_len + self.pred_len)
                for start in range(int(max_index) + 1):
                    self.sequence_infos.append((signal, df_signal, start))
        else:
            signal, df_signal = self.query[0]
            # for signal, df_signal in self.query:  
            max_index = len(df_signal.amplitude.values) - (self.seq_len + self.pred_len)
            for start in range(max_index + 1):
                self.sequence_infos.append((signal,df_signal,start))

    def __len__(self):
        return len(self.sequence_infos)

    # def get_entire_sequence(self):
    #     if self.project in ["population-level", "patient-specific"]:
    #         return {signal: df_signal for signal, df_signal in self.query}
    #     else:
    #         signal, df_signal = self.query[0]
    #         return df_signal
    
    def __getitem__(self, index: int) -> dict[str, Tensor]:
        signal,df_signal, start = self.sequence_infos[index]
        research_number, modality, fraction = (
            signal.research_number,
            signal.modality,
            signal.fraction,
        )
        time_series = df_signal.amplitude.values
        time = df_signal.time.values    
        if self.apply_denoise:
            time_series = fourier_smoothing(
                time_series=time_series,
                freq_threshold_hz=self.fourier_cutoff,
                sampling_rate=self.sampling_rate,
                return_spectrum=False,
            )
        if self.apply_noise:
            time_series_noisy = add_white_noise_to_signal(
                target_snr_db=self.white_noise_db, signal=time_series
            )
        else:
            time_series_noisy = time_series

        name = f"{research_number}_{modality}_{fraction}"

        #time_series_noisy = time_series  # Default to no noise

        i = start
        x_series = time_series[i : i + self.seq_len]
        x_noisy = time_series_noisy[i : i + self.seq_len]
        x_time = time[i : i + self.seq_len]

        y_series = time_series[i + self.seq_len : i + self.seq_len + self.pred_len]
        y_noisy = time_series_noisy[i + self.seq_len : i + self.seq_len + self.pred_len]
        y_time = time[i + self.seq_len : i + self.seq_len + self.pred_len]

        # print("Patient IDs:", name)
        # print("Input shape:", torch.from_numpy(x_series.astype(np.float32)).shape)
        # print("Target shape:", torch.from_numpy(y_series.astype(np.float32)).shape)
        # print("Time shape:", torch.from_numpy(x_time.astype(np.float32)).shape)
        
        return {
            "index": i,
            "name": name,
            "time_x": torch.from_numpy(x_time.astype(np.float32)).unsqueeze(-1),
            "series_x": torch.from_numpy(x_series.astype(np.float32)).unsqueeze(-1),
            "series_noisy_x": torch.from_numpy(x_noisy.astype(np.float32)).unsqueeze(-1),
            "time_y": torch.from_numpy(y_time.astype(np.float32)).unsqueeze(-1),
            "series_y": torch.from_numpy(y_series.astype(np.float32)).unsqueeze(-1),
            "series_noisy_y": torch.from_numpy(y_noisy.astype(np.float32)).unsqueeze(-1),
        }


def Create_dataset_population(args) -> Tuple[DataLoader, DataLoader]:
    train_ds = RpmSignals(
        db_root=args.db_root,
        project="PL",
        set_name="Pop-pre-train",
        fourier_smoothing_hz=args.fourier_smoothing_hz, 
        white_noise_db=args.white_noise_db,
        sampling_rate_hz=args.sampling_rate_hz,
        apply_denoise=args.apply_denoise,
        apply_noise=args.apply_noise,
        apply_scaling = args.apply_scaling,
        scaling_period_s= args.scaling_period_s,
        seq_len=args.seq_len,
        label_len=args.label_len,
        pred_len=args.pred_len,
        only_beam_on=args.only_beam_on,
        remove_offset=args.remove_offset,
    )

    val_ds = RpmSignals(
        db_root=args.db_root,
        project="PL",
        set_name="Pop-val",
        fourier_smoothing_hz=args.fourier_smoothing_hz,
        white_noise_db=args.white_noise_db,
        sampling_rate_hz=args.sampling_rate_hz,
        apply_denoise=args.apply_denoise,
        apply_noise=args.apply_noise,
        apply_scaling = args.apply_scaling,
        scaling_period_s= args.scaling_period_s,
        seq_len=args.seq_len,
        label_len=args.label_len,
        pred_len=args.pred_len,
        only_beam_on=args.only_beam_on,
        remove_offset=args.remove_offset,
    )
    return (
        DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=args.data_shuffle,
            drop_last=True,
            num_workers=args.num_workers,
            pin_memory=True,
            prefetch_factor=args.prefetch_factor
        ),
        DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=args.data_shuffle,
            drop_last=True,
            num_workers=args.num_workers,
            pin_memory=True,
            prefetch_factor=args.prefetch_factor
        )
    )

def Create_dataset_individual(
    args, patient_id: int | None = None, f_num: int | None = None, project: str | None = None
) -> Union[DataLoader, Tuple[DataLoader, DataLoader]]:
    assert hasattr(args, 'project'), "Missing project in args"
    project = project or args.project
    # shuffle = args.project not in ['test', 'data-specific']
    if project == 'PS-4DCT':
        if patient_id is None:
            raise ValueError("patient_id is required for PS-4DCT project")
        set_name = f"fine-tune_{patient_id}"
        #print('1111111:',set_name)
        ds = _create_single_dataset(args, set_name, project='PS-4DCT')

    elif project == 'PS':
        if patient_id is None or f_num is None:
            raise ValueError("patient_id or f_num are required for patient-specific project")
        #f_num = int(Fx_flag.replace('Fx', ''))  
        datasets: list[Dataset] = []
        # 4DCT part
        set_name_4DCT = f"fine-tune_{patient_id}"
        ps_4DCT = _create_single_dataset(args, set_name_4DCT, project='PS-4DCT')
        datasets.append(ps_4DCT)
        # fractions
        for i in range(1, f_num ):
            frac = f"f{i}"
            set_name = f"ps_{args.patient_id}_f{frac}"
            try:
                ds_frac = _create_single_dataset(args, set_name)
                datasets.append(ds_frac)
            except FileNotFoundError:
                print(f"Dataset {set_name} not found, skipping")
        ds = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]

    elif project == 'DS':
        if patient_id is None or f_num is None:
            raise ValueError("patient_id or f_num are required for data-specific project")
        #f_num = int(Fx_flag.replace('Fx', ''))  
        set_name = f"test_{patient_id:03d}_f{f_num}"
        ds = _create_single_dataset(args, set_name, project='test')

    elif project == 'test':
        ds = RpmSignals(
        db_root=args.db_root,
        project="test",
        fourier_smoothing_hz=args.fourier_smoothing_hz, 
        white_noise_db=args.white_noise_db,
        sampling_rate_hz=args.sampling_rate_hz,
        apply_denoise=args.apply_denoise,
        apply_noise=args.apply_noise,
        apply_scaling = args.apply_scaling,
        scaling_period_s= args.scaling_period_s,
        seq_len=args.seq_len,
        label_len=args.label_len,
        pred_len=args.pred_len,
        only_beam_on=args.only_beam_on,
        remove_offset=args.remove_offset,
    )
        return DataLoader(
            ds,
            batch_size=1, 
            shuffle=False,
            drop_last=True,
            num_workers=args.num_workers,
            pin_memory=True,
            prefetch_factor=args.prefetch_factor
        )
    else:
        raise ValueError(f"Unknown project type: {args.project}")
    
    if project == 'DS':
        total_samples = len(ds)
        train_samples = int(args.data_specific_train * args.sampling_rate_hz)
        train_end = int(train_samples * args.val_ratio)
        val_samples = train_samples -  train_end
        # print("val_samples",val_samples)
        # print("seq_len",args.seq_len)
        train_indices = list(range(0, train_end))
        val_indices = list(range(train_end, train_samples))
        test_indices = list(range(train_samples, total_samples))
        train_ds = Subset(ds, train_indices)
        val_ds = Subset(ds, val_indices)
        test_ds = Subset(ds, test_indices)

    else: # FT-4DCT and patient-specific
        total_samples = len(ds)
        split_idx = int(args.val_ratio* total_samples) 
        train_indices = list(range(split_idx))
        val_indices = list(range(split_idx, total_samples))
        train_ds = Subset(ds, train_indices)
        val_ds = Subset(ds, val_indices)
        test_set_name=f"test_{patient_id:03d}_f{f_num}"
        test_ds = _create_single_dataset(args, test_set_name,project = 'test')

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=args.data_shuffle,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=args.prefetch_factor
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,  
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=args.prefetch_factor
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=1, 
        shuffle=False,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=args.prefetch_factor
    )
    return train_loader, val_loader, test_loader
    

def _create_single_dataset(args, set_name: str, project: str | None = None) -> RpmSignals:
    return RpmSignals(
        db_root=args.db_root,
        project=project or args.project,
        set_name=set_name,
        fourier_smoothing_hz=args.fourier_smoothing_hz,
        white_noise_db=args.white_noise_db,
        sampling_rate_hz=args.sampling_rate_hz,
        apply_denoise=args.apply_denoise,
        apply_noise=args.apply_noise,
        apply_scaling = args.apply_scaling,
        scaling_period_s= args.scaling_period_s,
        seq_len=args.seq_len,
        label_len=args.label_len,
        pred_len=args.pred_len,
        only_beam_on=args.only_beam_on,
        remove_offset=args.remove_offset,
    )

def move_to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=True)
    elif isinstance(obj, (list, tuple)):
        return type(obj)(move_to_device(o, device) for o in obj)
    elif isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    else:
        return obj


def benchmark_dataloader(dataloader, device="cuda", num_batches=100):
    import time
    start_time = time.time()

    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break
        batch = move_to_device(batch, device)

    end_time = time.time()
    avg_time = (end_time - start_time) / num_batches
    print(f"Average time per batch: {avg_time:.6f} seconds")

def inspect_dataloader(dataloader, n=5):
    for i, batch in enumerate(dataloader):
        print(f"\nBatch {i}:")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                continue
                #print(value)
            elif isinstance(value, list):
                print(f"  {key}: list of length {len(value)}")
                print(value)
            else:
                print(f"  {key}: type={type(value)}, value={value}")
        if i >= n - 1:
            break

if __name__ == "__main__":
    init_fancy_logging()
    logging.getLogger("resp_db").setLevel(logging.DEBUG)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    def parse_args():
        parser = argparse.ArgumentParser(description="个体化数据集配置")
        # 必须参数
        parser.add_argument('--db-root', type=Path, default="/mnt/nas-wang/nas-ssd/Scripts/respiratory-signal-database/open_access_rpm_signals_master.db")
        parser.add_argument('--project', choices=['population-level','FT-4DCT', 'patient-specific', 'test', 'data-specific'], default='test')
        parser.add_argument('--patient_id', type=str, default='418', help="格式如 '010', 3 digits, Only available for project FT-4DCT patient-specific test data-specific")
        # 条件参数
        #parser.add_argument('--flag', type=str, default='train', help="train,val,test")
        parser.add_argument('--Fx_flag', type=str, default='Fx5', help="仅patient-specific项目使用,格式如 Fx2")
        # input size
        parser.add_argument('--seq-len', type=int, default=64, help="历史序列长度")
        parser.add_argument('--label-len', type=int, default=32, help="标签序列长度,可为0")
        parser.add_argument('--pred-len', type=int, default=12, help="预测序列长度,可为0")
        parser.add_argument('--only-beam-on', action='store_true', default=False, help="仅保留beam-on时段数据")
        parser.add_argument('--remove-offset', action='store_true', default=False, help="移除信号偏移量")
        # 通用
        parser.add_argument('--batch_size', type=int, default=256)
        parser.add_argument('--fourier_smoothing_hz', type=int, default=1)
        parser.add_argument('--white_noise_db', type=int, default=27)
        parser.add_argument('--sampling_rate_hz', type=int, default=25)
        parser.add_argument('--apply_denoise', action='store_true', default=True)
        parser.add_argument('--apply_noise', action='store_true', default=True)
        parser.add_argument('--num_workers', type=int, default=8)
        return parser.parse_args()

    args = parse_args()
    if args.project == 'population-level':
        dataloader, val_ds = Create_dataset_population(args)
        benchmark_dataloader(dataloader)
        inspect_dataloader(dataloader, n=5)
    #logger.info("=== Visualizing Population Dataloader ===")

    # ----------------- individual loader -------------------
    else:
        dataloader = Create_dataset_individual(args)
        benchmark_dataloader(dataloader)
    

        inspect_dataloader(dataloader, n=25)