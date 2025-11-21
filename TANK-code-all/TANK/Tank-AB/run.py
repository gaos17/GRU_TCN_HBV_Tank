import subprocess
import config
import os

for station in config.ZHANMING_LIST:
    for time_val in config.TIME_VALUES:
        print(f"\n{'=' * 50}")
        print(f"开始处理站点: {station} | time参数: {time_val}")
        print(f"{'=' * 50}\n")

        # 创建必要目录（修改为跨平台兼容写法）
        os.makedirs(f"./model/{station}", exist_ok=True)
        os.makedirs(f"./result/{station}", exist_ok=True)
        os.makedirs(f"./picture/{station}", exist_ok=True)
        os.makedirs(f"./hydrograph/{station}", exist_ok=True)

        # 运行训练（完整参数）
        train_result = subprocess.run(
            ["python", "train.py", "--zhanming", station, "--time", str(time_val)],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            env={**os.environ, 'PYTHONIOENCODING': 'utf-8'}
        )

        # 处理训练结果（添加详细日志）
        if train_result.returncode != 0:
            print(f"训练失败！站点: {station} | time: {time_val}")
            print("标准错误输出:")
            print(train_result.stderr)
            print("标准输出:")
            print(train_result.stdout)
            continue

        # 运行测试（完整参数）
        test_result = subprocess.run(
            ["python", "test.py", "--zhanming", station, "--time", str(time_val)],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            env={**os.environ, 'PYTHONIOENCODING': 'utf-8'}
        )

        # 处理测试结果
        if test_result.returncode != 0:
            print(f"测试失败！站点: {station} | time: {time_val}")
            print("标准错误输出:")
            print(test_result.stderr)
            print("标准输出:")
            print(test_result.stdout)