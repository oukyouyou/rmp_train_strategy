import random
from pathlib import Path
import peewee
from peewee import fn
from tqdm import tqdm
import re
import numpy as np
from peewee import SqliteDatabase


from resp_db import sql
from resp_db.client import RpmDatabaseClient

class DeepLearningDatasetBuilder:
    def __init__(self, args, new_table_name):
        self.args = args
        self.db_root = self.args.db_root
        self.table_name = new_table_name
        self.client = RpmDatabaseClient(db_filepath=self.db_root)
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)
        self.dataset = self.ensure_table_exists()
        self._classify_patients()

    def ensure_table_exists(self):
        class_name = self.table_name
 
        model_class = type(
            class_name,
            (sql.BaseModel,),  
            {
                "project": peewee.CharField(index=True),
                "set": peewee.CharField(index=True),
                "signal": peewee.ForeignKeyField(sql.Signal, backref=f'dl_datasets'),
                "Meta": type("Meta", (), {
                    "indexes": ((("project", "set", "signal"), True),),
                }),
            }
        )
        setattr(sql, class_name, model_class)
        
        if model_class.table_exists():
            # print(f"Table {class_name} exists! Dropping and recreating.")
            sql.database.drop_tables([model_class])
            sql.database.create_tables([model_class])
        else:
            # print(f"Table {class_name} does not exist, creating now.")
            sql.database.create_tables([model_class])
            
        return model_class

    # ======================
    # 核心逻辑函数
    # ======================
    def _classify_patients(self):
        # (1) 获取基础数据集
        all_patients = (
        sql.Patient.select(sql.Patient.id)
        .where(
            (sql.Patient.origin == "UKE") &
            (sql.Patient.id != 25) &
            (sql.Patient.id.in_(  # 只选择有有效信号的患者
                sql.ResearchNumber.select(sql.ResearchNumber.patient_id)
                .join(sql.Signal)
                .where(
                    (sql.Signal.is_corrupted == 0) &
                    (sql.Signal.modality.in_(["4DCT", "CBCT", "LINAC"]))
                )
                .distinct()
            ))
        )
        )
        # print(f"总患者数（排除25后）: {all_patients.count()}")

        # (2) 分类患者：有且仅有4DCT的患者 vs 三模态患者
        patients_withonly_4dct = []
        patients_with_under_5_LINAC = []
        patients_with_all = []
        
        #for patient in tqdm(all_patients, desc="分类患者"):
        for patient in tqdm(all_patients):
            has_cbct_or_linac = (
                sql.Signal.select()
                .join(sql.ResearchNumber)
                .where(
                    sql.ResearchNumber.patient == patient,
                    sql.Signal.modality.in_([ "LINAC"]),
                    sql.Signal.is_corrupted == 0
                )
                .exists()
            )
            if not has_cbct_or_linac:
                patients_withonly_4dct.append(patient.id)
            else:
                # 获取所有LINAC信号
                linac_count = (
                    sql.Signal.select()
                    .join(sql.ResearchNumber)
                    .where(
                        sql.ResearchNumber.patient == patient,
                        sql.Signal.modality == "LINAC",
                        sql.Signal.is_corrupted == 0
                    )
                    .count()
                )

                if linac_count > 5:
                    ## print(f"Patient ID: {patient.id}, total LINAC signals: {linac_count}")
                    patients_with_all.append(patient.id)
                else:
                    patients_with_under_5_LINAC.append(patient.id)

        # (3) 拆分全模态患者为两组
        # 假设 patients_with_all 是已经准备好的患者列表
        total_size = len(patients_with_under_5_LINAC)
        group1_size = int(total_size * self.args.split_ratio)  # data for pop train default 70%

        random.shuffle(patients_with_under_5_LINAC)  

        group1 = patients_with_under_5_LINAC[group1_size:]  # 前30%
        group2 = patients_with_under_5_LINAC[:group1_size]  # 后70%
        # print(f"全模态分组结果: Group1({len(group1)}) | Group2({len(group2)})")

        # (4) 生成 population-level 数据集
        population_patients = group1 + patients_withonly_4dct + patients_with_all
        random.shuffle(population_patients)
        split_idx = int(len(population_patients) * self.args.val_ratio)
        pop_train = population_patients[:split_idx]
        pop_val = population_patients[split_idx:]


        # (5) 处理 PS-4DCT 数据（group2的4DCT信号，按患者分组）
        ft_data = []
        #for pid in tqdm(group2, desc="收集PS-4DCT数据"):
        for pid in tqdm(group2):
            signals = (
                sql.Signal.select(sql.Signal.id)
                .join(sql.ResearchNumber)
                .where(
                    sql.ResearchNumber.patient == pid,
                    sql.Signal.modality == "4DCT",
                    sql.Signal.is_corrupted == 0
                )
            )
            for signal in signals:
                ft_data.append({
                    "project": "PS-4DCT",
                    "set": f"ps_4dct_{pid:03d}",  # 添加患者ID
                    "signal_id": signal.id
                })

        # (6) 处理 Test 和 PS 数据
        test_signals = []
        ps_signals = []
        #for pid in tqdm(group2, desc="处理测试数据"):
        for pid in group2:
            # 获取该患者所有非4DCT信号
            candidates = list(
                sql.Signal.select(sql.Signal.id, sql.Signal.modality, sql.Signal.fraction)
                .join(sql.ResearchNumber)
                .where(
                    sql.ResearchNumber.patient == pid,
                    sql.Signal.modality.in_(["LINAC"]),
                    sql.Signal.is_corrupted == 0
                )
            )
            if not candidates:
                continue
            
            # 随机选择一个作为测试
            test_signal = random.choice(candidates)
            test_signals.append({
                "patient_id": pid,
                "signal_id": test_signal.id,
                "fraction": test_signal.fraction 
            })
            
            # 剩余作为PS数据
            ps_signals.extend([
                {"patient_id": pid, "signal_id": s.id,  "fraction":s.fraction}
                for s in candidates if s.id != test_signal.id
            ])
        print(f"仅有4DCT患者数: {len(patients_withonly_4dct)}")
        print(f"LINAC<6患者数: {len(patients_with_under_5_LINAC)}")
        print(f"LINAC>6患者数: {len(patients_with_all)}")
        print(f"pop_train的患者数: {len(pop_train)}")
        print(f"pop_val的患者数: {len(pop_val)}")
        print(f"ps_4dct的患者数: {len(ft_data)}")

        # ======================
        # 写入数据库
        # ======================
        with sql.database.atomic():
            # 清空旧数据
            self.dataset.delete().execute()
            
            # 批量插入函数
            def batch_insert(project, set_name, signal_ids):
                data = [{"project": project, "set": set_name, "signal_id": sid} for sid in signal_ids]
                for i in range(0, len(data), 500):
                    self.dataset.insert_many(data[i:i+500]).execute()

            # (1) 插入population-level
            pop_train_signals = self.get_patient_signals(pop_train)
            pop_val_signals = self.get_patient_signals(pop_val)
            batch_insert("PL", "Pop-pre-train", pop_train_signals)
            batch_insert("PL", "Pop-val", pop_val_signals)
            print(f"Insert PL train dataset: {len(pop_train_signals)}, done")
            print(f"Insert PL vali dataset: {len(pop_val_signals)}, done")
            # (2) 插入PS-4DCT
            if ft_data:
                batch_size = 500
                for i in range(0, len(ft_data), batch_size):
                    batch = ft_data[i:i+batch_size]
                    self.dataset.insert_many([
                        {"project": item["project"], "set": item["set"], "signal_id": item["signal_id"]}
                        for item in batch
                    ]).execute()
            print(f"Insert PS-4DCT dataset: {len(ft_data)}, done")     
            # (3) 插入PS (rest)
            for entry in ps_signals:
                self.dataset.create(
                    project="PS",
                    set=f"ps_{entry['patient_id']:03d}_f{entry['fraction']}",
                    signal=entry["signal_id"]
                )
            print(f"Insert PS dataset: {len(ps_signals)}, done")   
            # (4) 插入test
            for entry in test_signals:
                self.dataset.create(
                    project="test",
                    set=f"test_{entry['patient_id']:03d}_f{entry['fraction']}",
                    signal=entry["signal_id"]
                )

            print(f"Insert test dataset: {len(test_signals)}, done")   
    def get_patient_signals(self,patient_ids):
        """获取指定患者的所有有效信号ID"""
        return [
            s.id for s in 
            sql.Signal.select(sql.Signal.id)
            .join(sql.ResearchNumber)
            .where(
                sql.ResearchNumber.patient.in_(patient_ids),
                sql.Signal.is_corrupted == 0
            )
        ]

if __name__ == "__main__":
    db_root = Path("/mnt/nas-wang/nas-ssd/Scripts/respiratory-signal-database/open_access_rpm_signals_master.db")
    client = RpmDatabaseClient(db_filepath=db_root)
    

