import inspect


def get_init_params(cls):
    """
    获取类的可填写的全部参数，包括从父类继承的参数。

    参数:
        cls: 要查询的Python类

    返回:
        list: 包含元组的列表，每个元组为 (参数名, 参数类型)，如果无类型则为None
    """
    # 获取类的MRO（方法解析顺序）
    mro = inspect.getmro(cls)[::-1]
    func_param = dict()

    # 遍历MRO，找到第一个定义了__init__方法的类
    for base in mro:
        if '__init__' in base.__dict__:
            # 获取__init__方法
            init_method = base.__dict__['__init__']
            # 获取方法签名
            sig = inspect.signature(init_method)
            # 获取参数列表
            params = sig.parameters
            func_param.update(params)

    return func_param


if __name__ == '__main__':
    from mmdet.datasets.transforms.transforms import Pad
    from mmdet.models.dense_heads.rtmdet_ins_head import RTMDetInsSepBNHead

    res = get_init_params(RTMDetInsSepBNHead)
    # 格式化输出参数名和类型
    for name, param in res.items():
        param_str = str(param) if param is not None else 'None'
        print(f"{name:<20} {param_str:<40}")
