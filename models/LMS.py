
import numpy as np
import pywt

def wlms_ar(x, wavelet='db4', p=2, mu=0.01, levels=3):
    """
    wLMS算法实现
    :param x: 输入信号
    :param wavelet: 小波类型（如'db4'）
    :param p: AR模型阶数
    :param mu: LMS学习率
    :param levels: 小波分解层数
    :return: 预测信号
    """
    # 小波分解
    coeffs = pywt.wavedec(x, wavelet, level=levels)
    
    # 对各子带应用LMS-AR
    pred_coeffs = []
    for coeff in coeffs:
        w, _ = lms_ar(coeff, p=p, mu=mu)  # 使用前文的LMS-AR函数
        # 预测子带未来值（此处简化：预测1步）
        x_pred = np.dot(w, coeff[-p:][::-1])
        pred_coeffs.append(np.append(coeff, x_pred))  # 扩展子带
    
    # 小波重构
    return pywt.waverec(pred_coeffs, wavelet)

# 示例
data = np.random.randn(100)  # 模拟非平稳信号
pred_signal = wlms_ar(data, wavelet='db4', p=2, mu=0.01)
