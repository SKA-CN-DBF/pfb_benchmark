#!/usr/bin/env python3
import numpy as np
from scipy.special import erfc
from scipy.signal import lfilter
from numpy.fft import fft, ifft, fftfreq, fftshift
import matplotlib.pylab as plt
'''
ref:
https://www.mathworks.com/matlabcentral/fileexchange/15813-near-perfect-reconstruction-polyphase-filterbank
'''

default_coeffs={
    8:    4.853,
    10:   4.775,
    12:   5.257,
    14:   5.736,
    16:   5.856,
    18:   7.037,
    20:   6.499,
    22:   6.483,
    24:   7.410,
    26:   7.022,
    28:   7.097,
    30:   7.755,
    32:   7.452,
    48:   8.522,
    64:   9.396,
    96:   10.785,
    128:  11.5, 
    192:  11.5,
    256:  11.5
}

def rrerf(F, K, M):
    x=K*(2*M*F-0.5)
    return np.sqrt(erfc(x)/2)

def coeff(N, L, K):
    K=default_coeffs[L] if L in default_coeffs and K is None else 8
    M=N//2
    F=np.arange(L*M)/(L*M)
    A=rrerf(F, K, M)
    N1=len(A)
    n=np.arange(N1//2-1)
    A[N1-n-1]=np.conj(A[1+n])
    A[N1//2]=0
    print(A)
    B=ifft(A).real
    B=fftshift(B)
    B=B/np.sum(B)
    print(B)
    return np.reshape(B, (L, M)).T

def analysis(coeff, x):
    #2倍超采样分析pfb
    x=np.squeeze(x).astype(np.complex)
    N=np.shape(coeff)[0]
    M=len(x)//N
    #x1=np.reshape(x, (N,M), order='F')
    x1=np.reshape(x, (M, N)).T
    x2=x1.copy()
    for i in range(N):
        x2[i,:]*=np.exp(1j*np.pi*i/N)
        x2[i,1::2]*=-1
    
    coeff=coeff[:, ::-1]
    for i in range(N):
        x1[i,:]=lfilter(coeff[i,:], 1, x1[i,:])
        x2[i,:]=lfilter(coeff[i,:], 1, x2[i,:])

    x1=ifft(x1, axis=0)*N
    x2=ifft(x2, axis=0)*N

    result=np.empty((2*N, M), dtype=x.dtype)
    result[::2, :]=x1
    result[1::2, :]=x2

    return result

def synthesis(coeff, x):
    #2倍超采样合成pfb
    N=np.shape(coeff)[0]
    M=np.shape(x)[1]
    y1=x[::2, :]
    y2=x[1::2, :]
    y1=fft(y1, axis=0)*N
    y2=fft(y2, axis=0)*N
    for i in range(N):
        y1[i, :] = lfilter(coeff[i, :], 1, y1[i, :])
        y2[i, :] = lfilter(coeff[i, :], 1, y2[i, :])
    for i in range(N):
        y2[i, :] *= np.exp(-1j*np.pi*i/N)
        y2[i, 1::2]*=-1

    return np.reshape((y1-y2).T, M*N).real

def add_delay(x, d):
    '''
    添加时延，d是添加进的时延
    '''
    nch=np.shape(x)[0] #通道个数
    y=x.copy()
    freq=fftfreq(nch)
    pf=np.exp(2.0j*np.pi*freq*d) #将时延转换为相因子
    for i in range(nch):
        y[i,:]*=pf[i] #把相因子乘到信道化后的数据上

    return y



def compare(x, d, tap=12, nch=1024):
    '''
    x: 输入信号，白噪声
    tap:每个通道的滤波器的阶数
    nch: 信道化个数
    d: 用于比较频域时延算法的延迟量，取整数
    本程序用频域时延算法计算一个输入白噪声的延迟后的结果，并以时域整数阶作为基准，对比效果。
    时域整数阶时延的结果是理想的，而频域时延是存在误差的。
    两者的差值反映了频域延迟算法的精度和质量
    '''
    L=tap
    N=nch
    K=3.8
    c=coeff(N, L, K) #design the filter
    #信道化
    y=analysis(c, x)
    #施加相位
    y=add_delay(y, d)
    #合成还原回时域
    z=synthesis(c, y)

    #filter delay是系统的内禀时延量
    filter_delay=nch//2*(tap-1)
    total_delay=filter_delay+d

    plt.plot(x[:-total_delay])
    plt.plot(z[total_delay:])
    plt.title("frequency and time domain delay result")
    plt.show()

    plt.scatter(x[:-total_delay],z[total_delay:])
    plt.title("frequency vs time domain delay result")
    plt.show()

    plt.plot(x[:-total_delay]-z[total_delay:])
    plt.title("error")
    plt.show()

    diff=x[:-total_delay]-z[total_delay:]
    print("relative residual={0}".format(np.std(diff)/np.std(x)))
