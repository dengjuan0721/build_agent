import os,sys
def save_it(text, output_filename, output_directory):
    """
    :param output_filename: file.md
    :return:
    """

    # 2. 创建目录（如果它不存在）
    try:
        # exist_ok=True 确保如果目录已存在，代码不会报错
        os.makedirs(output_directory, exist_ok=True)
    except OSError as e:
        print(f"错误: 无法创建目录 '{output_directory}'. 原因: {e}")
        sys.exit(1) # 如果目录创建失败，则退出脚本

    file_path = os.path.join(output_directory, output_filename)

    # 4. 将 final_plan 的内容写入文件
    try:
        # 使用 'with open' 语句来安全地写入文件
        # 'w' 表示写入模式，会覆盖已存在的文件
        # encoding='utf-8' 确保能正确处理中文字符
        with open(file_path, 'w', encoding='utf-8') as f:
            # 最好检查一下 final_plan 是否有内容，避免写入一个空文件
            if text:
                f.write(text)
                print(f"\n✅ 成功！最终计划已保存到: {file_path}")
            else:
                print("\n⚠️ 警告: text 为空，没有内容可以写入文件。")
    except IOError as e:
        print(f"\n❌ 错误: 无法写入文件 '{file_path}'. 原因: {e}")
