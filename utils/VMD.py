import numpy as np
from scipy.signal import hilbert



def mapmaxmin(data,ymax, ymin):
    xmax = np.max(data)
    xmin = np.min(data)
    new_data = (ymax-ymin)*(data-xmin) / (xmax - xmin) + ymin
    return new_data


class Vmd:
    def __init__(self, K, alpha, tau, tol=1e-7, maxIters=200, eps=1e-9):
        """
        :param K: 模态数
        :param alpha: 每个模态初始中心约束强度
        :param tau: 对偶项的梯度下降学习率
        :param tol: 终止阈值
        :param maxIters: 最大迭代次数
        :param eps: eps
        """
        self.K =K
        self.alpha = alpha
        self.tau = tau
        self.tol = tol
        self.maxIters = maxIters
        self.eps = eps

    def __call__(self, f):
        T = f.shape[0]
        t = np.linspace(1, T, T) / T
        omega = t - 1. / T
        # 转换为解析信号
        f = hilbert(f)
        f_hat = np.fft.fft(f)
        u_hat = np.zeros((self.K, T), dtype=np.complex)
        omega_K = np.zeros((self.K,))
        lambda_hat = np.zeros((T,), dtype=np.complex)
        # 用以判断
        u_hat_pre = np.zeros((self.K, T), dtype=np.complex)
        u_D = self.tol + self.eps

        # 迭代
        n = 0
        while n < self.maxIters and u_D > self.tol:
            for k in range(self.K):
                # u_hat
                sum_u_hat = np.sum(u_hat, axis=0) - u_hat[k, :]
                res = f_hat - sum_u_hat
                u_hat[k, :] = (res - lambda_hat / 2) / (1 + self.alpha * (omega - omega_K[k]) ** 2)

                # omega
                u_hat_k_2 = np.abs(u_hat[k, :]) ** 2
                omega_K[k] = np.sum(omega * u_hat_k_2) / np.sum(u_hat_k_2)

            # lambda_hat
            sum_u_hat = np.sum(u_hat, axis=0)
            res = f_hat - sum_u_hat
            lambda_hat -= self.tau * res

            n += 1
            u_D = np.sum(np.abs(u_hat - u_hat_pre) ** 2)
            u_hat_pre[::] = u_hat[::]

        # 重构，反傅立叶之后取实部
        u = np.real(np.fft.ifft(u_hat, axis=-1))

        omega_K = omega_K * T
        idx = np.argsort(omega_K)
        omega_K = omega_K[idx]
        u = u[idx, :]
        return u, omega_K


if __name__=="__main__":

    T = 1000
    fs = 1./T
    # t = np.linspace(0, 1, 1000,endpoint=True)
    data = open('D:/noncontact_spo2_pro/data/DEMO4/1/EVM/4_1_60/rightface_b.txt')
    f = data.readlines()
    f = np.array(list(map(float, f)))[100:599]
    f = mapmaxmin(f, 1, -1)
    K = 5
    alpha = 2000
    tau = 0
    vmd2 = Vmd(K, alpha, tau, tol=1e-7, maxIters=2000, eps=1e-9)
    u, omega_K = vmd2(f)
    print(omega_K)
    # array([0.85049797, 10.08516203, 50.0835613, 100.13259275]))

    # plt.figure(figsize=(5,7), dpi=200)
    # plt.subplot(5,1,1)
    # plt.title('vmd第一阶段')
    # plt.plot(u[0,:], linewidth=0.2, c='r')
    #
    # plt.subplot(5,1,2)
    # plt.plot(u[1,:], linewidth=0.2, c='r')
    #
    # plt.subplot(5,1,3)
    # plt.plot(u[2,:], linewidth=0.2, c='r')
    #
    # plt.subplot(5,1,4)
    # plt.plot(u[3,:], linewidth=0.2, c='r')
    # plt.subplot(5,1,5)
    # plt.plot(u[4,:], linewidth=0.2, c='r')
    # plt.show()
    # plt.figure()
    # plt.figure(figsize=(6,3), dpi=150)
    # plt.title('vmd第一阶段结束')

    new_data0 = f-u[0,:]-u[1,:]

    K = 8
    alpha = 2000
    tau = 1
    vmd1 = Vmd(K, alpha, tau, tol=1e-7, maxIters=200, eps=1e-9)
    u, omega_K = vmd1(new_data0)

