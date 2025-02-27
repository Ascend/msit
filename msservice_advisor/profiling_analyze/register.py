# 创建一个全局的注册表，注册为分析函数
REGISTRY = {}

ANSWERS = dict(env=dict(), config=dict())


def register_analyze(analyze_name=None):
    def decorator(func):
        name = analyze_name if analyze_name is not None else func.__name__
        REGISTRY[name] = func
        return func

    return decorator


def cached():
    # 缓存函数结果，反正所有输入都是一样的
    cache = {}

    def decorator(func):
        name = func.__name__

        def wrapper(*args, **kwargs):
            if name in cache:
                return cache[name]
            result = func(*args, **kwargs)
            cache[name] = result
            return result

        return wrapper

    return decorator


def answer(env=None, config=None, action=None, reason=""):
    if env:
        ANSWERS["env"].setdefault(env, [])
        ANSWERS["env"][env].append((action, reason))
    if config:
        ANSWERS["config"].setdefault(config, [])
        ANSWERS["config"][config].append((action, reason))

def print_answer():
    print("\n<answer>")
    for name, items in ANSWERS.get("env", dict()).items():
        for action, reason in items:
            print("env {} {} . reason: {}".format(name, action, reason))

    for name, items in ANSWERS.get("config", dict()).items():
        for action, reason in items:
            print("config {} {} . reason: {}".format(name, action, reason))

    print("</answer>")