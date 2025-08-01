# generate_partitioned_data.py

import pandas as pd
import os
import numpy as np
from faker import Faker
import random


def create_partitioned_parquet_files(base_path="data_lake", table_name="employees", partition_date="20250730",
                                     num_records=10000, num_files=3):
    """
    生成大量、真实的模拟员工数据，并将其分割成多个 Parquet 文件，
    存储在指定的分区目录中。
    """
    # 初始化 Faker
    fake = Faker('zh_CN')  # 使用中文数据，也可以用 'en_US'

    # --- 1. 创建分区目录 ---
    partition_path = os.path.join(base_path, table_name, f"date={partition_date}")
    if not os.path.exists(partition_path):
        os.makedirs(partition_path)
        print(f"Created partition directory: {partition_path}")

    # --- 2. 生成大量模拟数据 ---
    print(f"Generating {num_records} mock employee records...")
    data = []
    departments = ["Engineering", "Marketing", "Human Resources", "Data Science", "Product", "Sales", "Finance",
                   "Operations"]
    genders = ["Male", "Female"]

    for _ in range(num_records):
        gender = random.choice(genders)
        name = fake.name_male() if gender == "Male" else fake.name_female()

        # 模拟薪资，工程和数据科学部门薪资稍高
        department = random.choice(departments)
        if department in ["Engineering", "Data Science"]:
            base_salary = random.randint(150000, 400000)
        else:
            base_salary = random.randint(80000, 250000)
        salary = round(base_salary, -3)  # 四舍五入到千位

        data.append({
            "employee_id": fake.uuid4(),
            "employee_name": name,
            "age": random.randint(22, 60),
            "gender": gender,
            "department": department,
            "salary": salary,
            "hire_date": fake.date_between(start_date="-10y", end_date="today").strftime('%Y-%m-%d'),
            "bio": fake.text(max_nb_chars=200)  # 生成一段随机文本作为简介
        })

    # 创建完整的 DataFrame
    full_df = pd.DataFrame(data)
    print(f"Successfully generated DataFrame with {len(full_df)} records.")
    print("Sample of generated data:")
    print(full_df.head())
    print("\n")

    # --- 3. 分割 DataFrame 并保存为多个 Parquet 文件 ---
    # 使用 np.array_split 将数据大致平均地分成 num_files 份
    split_dfs = np.array_split(full_df, num_files)

    for i, df_part in enumerate(split_dfs):
        # 生成符合 Spark 规范的文件名
        file_name = f"part-{i:05d}-{fake.uuid4()}.snappy.parquet"
        file_path = os.path.join(partition_path, file_name)

        print(f"Saving file {i + 1}/{num_files}: {file_path} with {len(df_part)} records...")

        try:
            # index=False 表示不将 DataFrame 的索引写入文件
            # compression='snappy' 是 Spark 默认的压缩格式之一，速度快
            df_part.to_parquet(file_path, engine='pyarrow', index=False, compression='snappy')
        except Exception as e:
            print(f"An error occurred while saving {file_path}: {e}")

    print("\nProcess complete!")
    print(f"All data files for partition date={partition_date} have been saved in '{partition_path}'")


if __name__ == "__main__":
    create_partitioned_parquet_files()