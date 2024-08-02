
def decrement_number_in_path(path):
    import re
    pattern = re.compile(r'(\d+)')
    
    def replace_with_decrement(match):
        number = int(match.group(0)) - 1
        return str(number)

    updated_path = re.sub(pattern, replace_with_decrement, path)

    return updated_path

# 示例使用
original_path = "./work_dirs/modelscope/discriminator_2/checkpoint-discriminator-final/discriminator.pth"
updated_path = decrement_number_in_path(original_path)
print("Updated path:", updated_path)