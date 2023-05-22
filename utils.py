import os


def get_all_files(path, pattern):
    files = []
    lsdir = os.listdir(path)
    dirs = [i for i in lsdir if os.path.isdir(os.path.join(path, i))]
    if dirs:
        for i in dirs:
            files += get_all_files(os.path.join(path, i), pattern)
    files += [os.path.abspath(os.path.join(path, i)) for i in lsdir if os.path.isfile(
        os.path.join(path, i)) and os.path.splitext(i)[1] == pattern and os.path.getsize(os.path.join(path, i))]
    return files


def count_runtime(formatter: str):
    def wrapper(func):
        def do_wrapper(*args, **kwargs):
            import time
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            print(formatter.format(end - start))
            return result
        return do_wrapper
    return wrapper


def save_csv(filepath: str, header: list, data: list):
    import csv
    with open(filepath, 'w', encoding='utf8') as f:
        csv_writer = csv.writer(f)
        for row in header + data:
            csv_writer.writerow(row)


def sort(input, output, length, index=1, sep=",", worker_num=1, limit=10000, reverse=False):
    '''
    排序，并返回排序后的结果
    len是估计的长度，如果小于limit，则不作归并处理
    '''
    if length > limit:
        cmd = f'./sort.sh {worker_num} {index} {limit} {sep} {input} {output}'
        if reverse:
            cmd + 'r'
    else:
        cmd = f'sort --parallel={worker_num} -t {sep} -nk{index} {input} > {output}'
        if reverse:
            cmd = f'sort --parallel={worker_num} -t {sep} -nrk{index} {input} > {output}'
    os.system(cmd)
    return output
