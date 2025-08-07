from __future__ import annotations

import pickle
from pathlib import Path
from peewee import JOIN
import numpy as np
import pandas as pd
import peewee
from peewee import IntegrityError
from tqdm import tqdm

from collections import defaultdict, Counter

from resp_db import sql

from resp_db.common_types import MaybeSequence
from utils.logger import LoggerMixin
from resp_db.time_series_utils import find_peaks, resample_time_series
import re


class RpmDatabaseClient(LoggerMixin):
    """database client to perform querying and preprocessing."""

    def __init__(self, db_filepath: Path = "rpm_signals.db"):
        self.db_filepath = db_filepath
        if not self.db_filepath.is_file():
            raise FileNotFoundError(f"{self.db_filepath} is not a valid database file!")
        sql.database.init(database=self.db_filepath)
        self._create_tables()

    def __repr__(self):
        return f"<{self.__class__.__name__}(datapath={self.db_filepath!r})>"

    def __enter__(self):
        self.logger.debug(f"{self} successfully connected!")
        sql.database.connect(reuse_if_open=True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _create_tables(self):
        with self:
            sql.database.create_tables(
                [
                    sql.Signal,
                    sql.ResearchNumber,
                    sql.Patient,
                    sql.RespiratoryStats,
                    sql.DeepLearningDataset,
                ],
                safe=True,
            )

    @staticmethod
    def _combination_repr(
        research_number: int, modality: str, fraction: int, origin: str
    ) -> str:
        combination_repr = (
            f"\n"
            f"\033[4mENTERED INPUT PARAMETERS\033[0m \n"
            f"ORIGIN : {origin} \n"
            f"RESEARCH NUMBER: {research_number} \n"
            f"MODALITY: {modality} \n"
            f"FRACTION: {fraction}"
            f"\n"
        )
        return combination_repr

    @staticmethod
    def _check_function_inputs(
        research_number: int, modality: str | None, fraction: int | None
    ):
        possible_modalities = ["4DCT", "CBCT", "LINAC"]
        if modality and modality not in possible_modalities:
            raise ValueError(f"{modality=} has to be one of {possible_modalities}.")
        if fraction and not 0 <= fraction < 11:
            raise ValueError(
                f"{fraction=} is an invalid input. It has to be between 0 and 10"
            )

        research_number_str = str(research_number)
        if not any(
   38,         [research_number_str.startswith("57"), len(research_number_str) == 7]
        ):
            raise ValueError(f"{research_number} has to start with 57 and has 7 digits")

    @staticmethod
    def get_specific_signal(
        research_number: int,
        modality: str | None,
        fraction: int | None,
        origin: str = "UKE",
        return_only_query: bool = False,
    ) -> tuple[MaybeSequence[pd.DataFrame], MaybeSequence[sql.Signal]] | MaybeSequence[
        sql.Signal
    ]:

        RpmDatabaseClient._check_function_inputs(research_number, modality, fraction)
        query = (
            sql.Signal.select(sql.Signal)
            .join(
                sql.ResearchNumber,
                on=(sql.Signal.research_number == sql.ResearchNumber.id),
            )
            .join(sql.Patient, on=(sql.ResearchNumber.patient_id == sql.Patient.id))
            .where(
                sql.ResearchNumber.id == research_number,
                sql.Patient.origin == origin,
            )
        )
        if fraction is not None:
            query = query.where(sql.Signal.fraction == fraction)
        if modality:
            query = query.where(sql.Signal.modality == modality)
        if query.count() == 0:
            raise FileNotFoundError(
                f"No signal found in database for the following combination:"
                f"{RpmDatabaseClient._combination_repr(research_number, modality, fraction, origin)}"
            )
        if return_only_query:
            return query
        if query.count() > 1 and fraction is not None and modality is not None:
            raise IntegrityError(
                f"Database fuck-up. {query.count()} signals were found for an unique combination. \n"
                f"Please review and clean database for:"
                f"{RpmDatabaseClient._combination_repr(research_number, modality, fraction, origin)}"
            )
        df_signals = [pickle.loads(signal.df_signal) for signal in query]
        if query.count() == 1:
            return df_signals[0], query[0]
        return df_signals, query

    @staticmethod
    def preprocess_signal(
        df_signal: pd.DataFrame | bytes,
        only_beam_on: bool = True,
        sampling_rate: int = 26,
        remove_offset: bool = True,
    ) -> pd.DataFrame:
        """Performs preprocessing by.

        - only using first to last beam on point (excluding potential acquisition errors)
        - resampling to given sampling_rate
        - shifting raw signal that first three minima are at zero.
        :param df_signal:
        :param only_beam_on:
        :param sampling_rate:
        :param remove_offset:
        :return: pd.Dataframe
        """
        if isinstance(df_signal, bytes):
            df_signal = pickle.loads(df_signal)
        if not isinstance(df_signal, pd.DataFrame):
            raise ValueError(
                f"df_signal should be a Dataframe but is type {type(df_signal)}"
            )
        if not {"time", "amplitude", "beam_on"}.issubset(df_signal.columns):
            raise ValueError(
                f"Dataframe does not contain all mandatory columns; {df_signal.columns}"
            )
        if (
            any(df_signal.amplitude.isna())
            or any(df_signal.time.isna())
            or any(df_signal.beam_on.isna())
        ):
            raise ValueError("Contain invalid data")
        if only_beam_on:
            beam_on_idx = np.where(df_signal.beam_on == 0)[0]
            first_beam_on, last_beam_on = min(beam_on_idx), max(beam_on_idx)
            df_signal = df_signal[first_beam_on:last_beam_on]
            df_signal.reset_index(inplace=True, drop=True)
            time_offset = df_signal.time.min()
            df_signal[:]["time"] -= time_offset
        if int(sampling_rate) != 26:
            t_new, a_new = resample_time_series(
                signal_time_secs=df_signal.time.values,
                signal_amplitude=df_signal.amplitude.values,
                target_samples_per_second=sampling_rate,
            )
            df_signal = pd.DataFrame.from_dict(
                dict(time=t_new, amplitude=a_new), dtype=float
            )
        if remove_offset:
            signal_subset = -1 * df_signal.amplitude[df_signal.time < 50]
            number_minima = 3
            minima_idx = find_peaks(x=signal_subset.values)
            minima = df_signal.amplitude[minima_idx].values
            df_signal.loc[:, "amplitude"] = (
                df_signal.amplitude - minima[:number_minima].mean()
            )
        return df_signal

    @staticmethod
    def get_signals_of_dl_dataset(dl_dataset: str, project: str) -> peewee.ModelSelect:
        # if dl_dataset not in ["Pop-pre-train","Pop-val", "Ft-4DCT", "test"]:
        #     raise ValueError(f"Invalid dataset: {dl_dataset}")

        query = (
            sql.Signal.select(sql.Signal)
            .join(
                sql.DeepLearningDataset,
                on=(sql.DeepLearningDataset.signal == sql.Signal.id),
            )
            .where(
                sql.DeepLearningDataset.set == dl_dataset,
                sql.DeepLearningDataset.project == project,
                sql.Signal.is_corrupted == 0, 
            )
        )
        return query
    @staticmethod
    def get_signals_of_dl_dataset_benchmark(dl_dataset: str) -> peewee.ModelSelect:
        # if dl_dataset not in ["Pop-pre-train","Pop-val", "Ft-4DCT", "test"]:
        #     raise ValueError(f"Invalid dataset: {dl_dataset}")

        query = (
            sql.Signal.select(sql.Signal)
            .join(
                sql.DeepLearningDataset,
                on=(sql.DeepLearningDataset.signal == sql.Signal.id),
            )
            .where(
                sql.DeepLearningDataset.set == dl_dataset,
                sql.Signal.is_corrupted == 0, 
            )
        )
        return query
    
    @staticmethod
    def get_all_project_signals_of_dl_dataset(project: str) -> peewee.ModelSelect:
        # if dl_dataset not in ["Pop-pre-train","Pop-val", "Ft-4DCT", "test"]:
        #     raise ValueError(f"Invalid dataset: {dl_dataset}")

        query = (
            sql.Signal.select(sql.Signal)
            .join(
                sql.DeepLearningDataset,
                on=(sql.DeepLearningDataset.signal == sql.Signal.id),
            )
            .where(
                sql.DeepLearningDataset.project == project,
                sql.Signal.is_corrupted == 0, 
            )
        )
        return query
    
    @staticmethod
    def get_pids_fraction_of_test_dataset(project: str) -> set[tuple[int, int]]:
        query = (
        sql.DeepLearningDataset
        .select(sql.DeepLearningDataset.set)
        .where(sql.DeepLearningDataset.project == project)
        )

        pairs = set()
        for item in query:
            match = re.match(r'test_(\d+)_f(\d+)', item.set)
            if match:
                pid = int(match.group(1))
                f_num = int(match.group(2))
                test_set_name=f"test_{pid:03d}_f{f_num}"
                pid = str(pid).zfill(3)
                pairs.add((pid, f_num))
        return pairs
    
    @staticmethod
    def get_signals_by_modality(
        modalities: list[str],
        origin: str = "UKE",
        return_dataframe: bool = True
    ) -> tuple[list[pd.DataFrame], list[sql.Signal]] | peewee.ModelSelect:

        possible_modalities = ["4DCT", "CBCT", "LINAC"]
        for mod in modalities:
            if mod not in possible_modalities:
                raise ValueError(f"{mod} has to be one of {possible_modalities}")

        query = (
            sql.Signal.select(sql.Signal)
            .join(sql.ResearchNumber)
            .join(sql.Patient)
            .where(
                sql.Signal.modality.in_(modalities),
                sql.Patient.origin == origin,
                sql.Signal.is_corrupted == 0, 
            )
            .order_by(sql.Signal.research_number, sql.Signal.fraction)
        )
        
        if return_dataframe:
            signals = []
            dfs = []
            for signal in query:
                df = pickle.loads(signal.df_signal)
                dfs.append(df)
                signals.append(signal)
            return dfs, signals
        return query
    
    @staticmethod
    def get_modality_distribution():
    # 查询原始 signal，带上 patient_id 和 modality
        signal_records = (
            sql.Signal
            .select(
                sql.ResearchNumber.patient_id.alias('patient_id'),
                sql.Signal.modality
            )
            .join(sql.ResearchNumber)
            .where(
                sql.Signal.is_corrupted == 0,
                sql.Signal.modality.in_(["4DCT", "CBCT", "LINAC"]),
                sql.ResearchNumber.patient_id != 25  # 排除 patient_id 为 25 的数据
            )
        )

        # 每个 patient_id -> Counter of modalities
        patient_modality_counts = defaultdict(Counter)

        for row in signal_records.dicts():
            pid = row["patient_id"]
            mod = row["modality"]
            patient_modality_counts[pid][mod] += 1

        # 初始化频次分布
        modality_distribution = {
            "4DCT": Counter(),
            "CBCT": Counter(),
            "LINAC": Counter(),
        }

        for counts in patient_modality_counts.values():
            for modality in ["4DCT", "CBCT", "LINAC"]:
                count = counts.get(modality, 0)
                if modality == "4DCT":
                    key = count if count <= 3 else ">3"
                elif modality == "CBCT":
                    key = count if count <= 8 else ">8"
                elif modality == "LINAC":
                    key = count if count <= 10 else ">10"
                modality_distribution[modality][key] += 1
        
        count_no_cbct_linac = sum(
        1 for v in patient_modality_counts.values()
        if v.get("CBCT", 0) == 0 and v.get("LINAC", 0) == 0
        )
        return modality_distribution, count_no_cbct_linac
        
    @staticmethod
    def get_pid_by_research_number(research_number: str) -> str:
        query = (
            sql.Patient.select(sql.Patient.id)
            .join(sql.ResearchNumber) 
            .where(
                sql.ResearchNumber.research_number == research_number,
            )
        )
        result = query.first()
        if result:
            return result.pid
        return None
        
    @staticmethod
    def get_patients_without_cbct_and_linac():
        # 查询所有的 signal 数据，获取 patient_id 和 modality
        signal_records = (
            sql.Signal
            .select(
                sql.ResearchNumber.patient_id.alias('patient_id'),
                sql.Signal.modality
            )
            .join(sql.ResearchNumber)
            .where(
                sql.Signal.is_corrupted == 0,
                sql.Signal.modality.in_(["4DCT", "CBCT", "LINAC"])
            )
        )

        # 用来记录每个 patient 拥有的 modalities
        patient_modality_counts = defaultdict(set)

        # 填充每个 patient 对应的 modalities（不去重 fraction）
        for row in signal_records.dicts():
            pid = row["patient_id"]
            mod = row["modality"]
            patient_modality_counts[pid].add(mod)

        # 筛选出没有 CBCT 和 LINAC 的 patient_id
        patients_without_cbct_and_linac = [
            pid for pid, mods in patient_modality_counts.items()
            if "CBCT" not in mods and "LINAC" not in mods
        ]

        return patients_without_cbct_and_linac

    @staticmethod
    def get_patient_modality_stats():
        # 查询所有的 signal 数据，获取 patient_id 和 modality
        signal_records = (
            sql.Signal
            .select(
                sql.ResearchNumber.patient_id.alias('patient_id'),
                sql.Signal.modality
            )
            .join(sql.ResearchNumber)
            .where(
                sql.Signal.is_corrupted == 0,
                sql.Signal.modality.in_(["4DCT", "CBCT", "LINAC"]),
                sql.ResearchNumber.patient_id != 25  # 排除 patient_id 为 25 的数据
            )  
        )

        # 用来记录每个 patient 拥有的 modalities 数量
        patient_modality_counts = defaultdict(Counter)

        # 填充每个 patient 对应的 modalities（不去重 fraction）
        for row in signal_records.dicts():
            pid = row["patient_id"]
            mod = row["modality"]
            patient_modality_counts[pid][mod] += 1

        # 构造返回的数据
        patient_data = [
            {
                "id": f"PID{int(patient_id):03d}",
                "ct": modality_counts.get("4DCT", 0),
                "cbct": modality_counts.get("CBCT", 0),
                "linac": modality_counts.get("LINAC", 0),
            }
            for patient_id, modality_counts in patient_modality_counts.items()
        ]

        return patient_data


    def close(self):
        sql.database.close()
        self.logger.debug(f"{self.db_filepath} was disconnected.")





############

#     @staticmethod
#     def get_patient_signal_stats():
#         # 统计原始 Signal 表中每条记录的 patient_id 和 modality
#         signal_records = (
#             sql.Signal
#             .select(
#                 sql.ResearchNumber.patient_id.alias('patient_id'),
#                 sql.Signal.modality
#             )
#             .join(sql.ResearchNumber)
#             .where(
#                 sql.Signal.is_corrupted == 0,
#                 sql.Signal.modality.in_(["4DCT", "CBCT", "LINAC"])
#             )
#         )

#         # 用来统计每个 patient 的各类 modality 数量（不去重！）
#         patient_modality_counter = defaultdict(Counter)

#         for row in signal_records.dicts():
#             pid = row["patient_id"]
#             mod = row["modality"]
#             patient_modality_counter[pid][mod] += 1

#         # 汇总总量
#         total_signals = 0
#         total_by_modality = Counter()

#         for pid, mod_counts in patient_modality_counter.items():
#             total_signals += sum(mod_counts.values())
#             total_by_modality += mod_counts

#         return patient_modality_counter, total_signals, total_by_modality