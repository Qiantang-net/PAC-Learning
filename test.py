import subprocess

# 需要循环执行的命令列表
commands = [
    # SWaT
    "python main.py --anormly_ratio 0.1 --num_epochs 100 --batch_size 32 --mode train --dataset SWAT --data_path /home/stu2023/qtt/project/MEMTO-main/dataset/SWAT/ --input_c 51 --output_c 51 --n_memory 10 --lambd 0.01 --lr 1e-4 --memory_initial False --phase_type None",
    "python main.py --anormly_ratio 0.1 --num_epochs 100 --batch_size 32 --mode memory_initial --dataset SWAT --data_path /home/stu2023/qtt/project/MEMTO-main/dataset/SWAT/ --input_c 51 --output_c 51 --n_memory 10 --lambd 0.01 --lr 5e-5 --memory_initial True --phase_type second_train",
    "python main.py --anormly_ratio 0.1 --num_epochs 100 --batch_size 32 --mode test --dataset SWAT --data_path /home/stu2023/qtt/project/MEMTO-main/dataset/SWAT/ --input_c 51 --output_c 51 --n_memory 10 --memory_initial False --phase_type test",

    # SMD
    "python main.py --anormly_ratio 0.5 --num_epochs 100 --batch_size 32 --mode train --dataset SMD --data_path /home/stu2023/qtt/project/MEMTO-main/dataset/SMD/ --input_c 38 --output_c 38 --n_memory 10 --lambd 0.01 --lr 1e-4 --memory_initial False --phase_type None",
    "python main.py --anormly_ratio 0.5 --num_epochs 100 --batch_size 32 --mode memory_initial --dataset SMD --data_path /home/stu2023/qtt/project/MEMTO-main/dataset/SMD/ --input_c 38 --output_c 38 --n_memory 10 --lambd 0.01 --lr 5e-5 --memory_initial True --phase_type second_train",
    "python main.py --anormly_ratio 0.5 --num_epochs 10 --batch_size 32 --mode test --dataset SMD --data_path /home/stu2023/qtt/project/MEMTO-main/dataset/SMD/ --input_c 38 --output_c 38 --n_memory 10 --memory_initial False --phase_type test",

    # SMAP
    "python main.py --anormly_ratio 1.0 --num_epochs 100 --batch_size 32 --mode train --dataset SMAP --data_path /home/stu2023/qtt/project/MEMTO-main/dataset/SMAP/ --input_c 25 --output_c 25 --n_memory 10 --lambd 0.01 --lr 1e-4 --memory_initial False --phase_type None",
    "python main.py --anormly_ratio 1.0 --num_epochs 100 --batch_size 32 --mode memory_initial --dataset SMAP --data_path /home/stu2023/qtt/project/MEMTO-main/dataset/SMAP/ --input_c 25 --output_c 25 --n_memory 10 --lambd 0.01 --lr 5e-5 --memory_initial True --phase_type second_train",
    "python main.py --anormly_ratio 1.0 --num_epochs 10 --batch_size 32 --mode test --dataset SMAP --data_path /home/stu2023/qtt/project/MEMTO-main/dataset/SMAP/ --input_c 25 --output_c 25 --n_memory 10 --memory_initial False --phase_type test",

    # PSM
    "python main.py --anormly_ratio 1.0 --num_epochs 100 --batch_size 32 --mode train --dataset PSM --data_path /home/stu2023/qtt/project/MEMTO-main/dataset/PSM/ --input_c 25 --output_c 25 --n_memory 10 --lambd 0.01 --lr 1e-4 --memory_initial False --phase_type None",
    "python main.py --anormly_ratio 1.0 --num_epochs 100 --batch_size 32 --mode memory_initial --dataset PSM --data_path /home/stu2023/qtt/project/MEMTO-main/dataset/PSM/ --input_c 25 --output_c 25 --n_memory 10 --lambd 0.01 --lr 5e-5 --memory_initial True --phase_type second_train",
    "python main.py --anormly_ratio 1.0 --num_epochs 10 --batch_size 32 --mode test --dataset PSM --data_path /home/stu2023/qtt/project/MEMTO-main/dataset/PSM/ --input_c 25 --output_c 25 --n_memory 10 --memory_initial False --phase_type test",

    # MSL
    "python main.py --anormly_ratio 1.0 --num_epochs 100 --batch_size 32 --mode train --dataset MSL --data_path /home/stu2023/qtt/project/MEMTO-main/dataset/MSL/ --input_c 55 --output_c 55 --n_memory 10 --lambd 0.01 --lr 1e-4 --memory_initial False --phase_type None",
    "python main.py --anormly_ratio 1.0 --num_epochs 100 --batch_size 32 --mode memory_initial --dataset MSL --data_path /home/stu2023/qtt/project/MEMTO-main/dataset/MSL/ --input_c 55 --output_c 55 --n_memory 10 --lambd 0.01 --lr 5e-5 --memory_initial True --phase_type second_train",
    "python main.py --anormly_ratio 1.0 --num_epochs 10 --batch_size 32 --mode test --dataset MSL --data_path /home/stu2023/qtt/project/MEMTO-main/dataset/MSL/ --input_c 55 --output_c 55 --n_memory 10 --memory_initial False --phase_type test",

    # NIPS TS_Swan
    "python main.py --anormly_ratio 1.0 --num_epochs 100 --batch_size 32 --mode train --dataset NIPS_TS_Swan --data_path /home/stu2023/qtt/project/MEMTO-main/dataset/NIPS_TS_Swan/ --input_c 38 --output_c 38 --n_memory 10 --lambd 0.01 --lr 1e-4 --memory_initial False --phase_type None",
    "python main.py --anormly_ratio 1.0 --num_epochs 100 --batch_size 32 --mode memory_initial --dataset NIPS_TS_Swan --data_path /home/stu2023/qtt/project/MEMTO-main/dataset/NIPS_TS_Swan/ --input_c 38 --output_c 38 --n_memory 10 --lambd 0.01 --lr 5e-5 --memory_initial True --phase_type second_train",
    "python main.py --anormly_ratio 1.0 --num_epochs 10 --batch_size 32 --mode test --dataset NIPS_TS_Swan --data_path /home/stu2023/qtt/project/MEMTO-main/dataset/NIPS_TS_Swan/ --input_c 38 --output_c 38 --n_memory 10 --memory_initial False --phase_type test",

    # NIPS_TS_Water
    "python main.py --anormly_ratio 1.0 --num_epochs 100 --batch_size 32 --mode train --dataset NIPS_TS_Water --data_path /home/stu2023/qtt/project/MEMTO-main/dataset/NIPS_TS_Water/ --input_c 9 --output_c 9 --n_memory 10 --lambd 0.01 --lr 1e-4 --memory_initial False --phase_type None",
    "python main.py --anormly_ratio 1.0 --num_epochs 100 --batch_size 32 --mode memory_initial --dataset NIPS_TS_Water --data_path /home/stu2023/qtt/project/MEMTO-main/dataset/NIPS_TS_Water/ --input_c 9 --output_c 9 --n_memory 10 --lambd 0.01 --lr 5e-5 --memory_initial True --phase_type second_train",
    "python main.py --anormly_ratio 1.0 --num_epochs 10 --batch_size 32 --mode test --dataset NIPS_TS_Water --data_path /home/stu2023/qtt/project/MEMTO-main/dataset/NIPS_TS_Water/ --input_c 9 --output_c 9 --n_memory 10 --memory_initial False --phase_type test"
]

# 运行每个命令 10 次
for i in range(10):
    print(f"Starting iteration {i + 1} of 10...")
    for command in commands:
        subprocess.run(command, shell=True)
    print(f"Iteration {i + 1} complete.")
