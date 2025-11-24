# Licensed under the TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/main/LICENSE
#
# Unless and only to the extent required by applicable law, the Tencent Hunyuan works and any
# output and results therefrom are provided "AS IS" without any express or implied warranties of
# any kind including any warranties of title, merchantability, noninfringement, course of dealing,
# usage of trade, or fitness for a particular purpose. You are solely responsible for determining the
# appropriateness of using, reproducing, modifying, performing, displaying or distributing any of
# the Tencent Hunyuan works or outputs and assume any and all risks associated with your or a
# third party's use or distribution of any of the Tencent Hunyuan works or outputs and your exercise
# of rights and permissions under this agreement.
# See the License for the specific language governing permissions and limitations under the License.

import torch

from simpletuner.helpers.models.hunyuanvideo.commons.infer_state import get_infer_state


def torch_compile_wrapper():
    """返回一个装饰器，延迟决定是否使用torch.compile"""

    def decorator(func):
        def wrapper(*args, **kwargs):
            if get_infer_state() and get_infer_state().enable_torch_compile:
                compiled_func = torch.compile(func)
                return compiled_func(*args, **kwargs)
            else:
                return func(*args, **kwargs)

        return wrapper

    return decorator
