import numpy as np
import torch

# suspect_window_length 这个值比较小, fft段异常的注入也要受到这个限制的影响...
# -- 针对上面这个，可以考虑在全局变换后截取最后一段？-- 但长度的改变只是分辨率的改变，不一定有明显作用？
# 周期的缩放这类确实是频域上更好操作，还有类似的异常种类么
def seasonal_shift(
    x: torch.Tensor,
    tl: float,
    th: float,
) -> torch.tensor:
    # 两种周期上的变换方式：周期变大，周期变小
    # 随机生成一个变化的倍数，然后>1和<1分别处理，初步设置倍数在0.3~3之间，且不在1附近
    # <1 是截断高频部分，>1是高频后面补0
    import random
    from scipy.fftpack import fft, ifft

    xn = x.numpy()
    fft_yn = fft(xn - np.mean(xn))
    # 如果不考虑别的两个序列的拼接的话，离1太近不太行吧，可能会把正常的标成异常的；
    # 如果考虑两个序列的拼接，也就是类似coe_batch, 但感觉不一定需要这么强？
    multi = random.uniform(0.3, 3)
    while multi > tl and multi < th:
        multi = random.uniform(0.3, 3)
    print("multi is", multi)

    xlen = int(np.ceil(multi * x.shape[0]))
    print("xlen is", xlen)
    a = np.complex128(0 + 0j)
    fft_yn_new = a * np.arange(xlen)

    if multi < 1:
        ilceil = int(np.ceil(xlen / 2))
        fft_yn_new.real = np.concatenate((fft_yn.real[0:ilceil], fft_yn.real[ilceil - xlen:]))
        fft_yn_new.imag = np.concatenate((fft_yn.imag[0:ilceil], fft_yn.imag[ilceil - xlen:]))
        xn_ifft = ifft(fft_yn_new) + np.mean(xn)
        xt = np.tile(xn_ifft, 6)
        xc = xt[0:x.shape[0]]
        y = torch.from_numpy(xc)

    if multi > 1:
        iceil = int(np.ceil(x.shape[0] / 2))
        fft_yn_new.real = np.concatenate(
            (fft_yn.real[0:iceil], np.zeros(xlen - x.shape[0]), fft_yn.real[iceil - x.shape[0]:]), axis=-1)
        fft_yn_new.imag = np.concatenate(
            (fft_yn.imag[0:iceil], np.zeros(xlen - x.shape[0]), fft_yn.imag[iceil - x.shape[0]:]), axis=-1)
        xn_ifft = ifft(fft_yn_new) + np.mean(xn)
        xc = xn_ifft[0:x.shape[0]]
        y = torch.from_numpy(xc.real)

    return y

def with_noise(
    x: torch.Tensor,
) -> torch.tensor:
    # 实部/虚部 少部分随机扰动 -- 保守的注入
    # 有个疑虑 多大程度上的注入算是异常...prop从0.1开始可以么？or 0.01
    from scipy.fftpack import fft, ifft
    flag = np.random.randint(low=0, high=2)
    xn = x.numpy()
    fft_yn = fft(xn - np.mean(xn))
    a = np.complex128(0 + 0j)
    fft_yn_new = a * np.arange(x.shape[0])

    prop = np.random.uniform(0.01, 0.5)
    chnum = int(np.ceil(prop * x.shape[0]))
    halfnum = int(np.floor(0.5 * x.shape[0]))

    if flag==0: # 实部部分扰动, 高斯噪声, 虚部保留
        noise = np.random.randn(fft_yn.real[0:chnum].shape[0])
        fft_yn_new.real[0:chnum] = fft_yn.real[0:chnum] + noise
        fft_yn_new.real[chnum:halfnum] = fft_yn.real[chnum:halfnum]
        # 轴对称
        fft_yn_new.real[-halfnum:] = np.flip(fft_yn_new.real[0:halfnum], axis=-1)
        # 虚部
        fft_yn_new.imag = fft_yn.imag

    if flag==1: # 虚部部分扰动, 高斯噪声，实部保留
        noise = np.random.randn(fft_yn.imag[0:chnum].shape[0])
        fft_yn_new.imag[0:chnum] = fft_yn.imag[0:chnum] + noise
        fft_yn_new.imag[chnum:halfnum] = fft_yn.imag[chnum:halfnum]
        # 中心对称
        fft_yn_new.imag[-halfnum:] = -np.flip(fft_yn_new.imag[0:halfnum], axis=-1)
        # 实部
        fft_yn_new.real = fft_yn.real

    xn_ifft = ifft(fft_yn_new) + np.mean(xn)
    y = torch.from_numpy(xn_ifft.real)

    return y


def other_fftshift(
    x: torch.Tensor,
) -> torch.tensor:
    # 高低频变换(实部部分高低频变换、虚部部分高低频变换)，某些频率*mul -- 激进的注入
    from scipy.fftpack import fft, ifft
    flag = np.random.randint(low=0, high=4)
    xn = x.numpy()
    fft_yn = fft(xn - np.mean(xn))
    a = np.complex128(0 + 0j)
    fft_yn_new = a * np.arange(x.shape[0])

    prop = np.random.uniform(0.01, 0.25)
    chnum = int(np.ceil(prop * x.shape[0]))
    halfnum = int(np.ceil(0.5 * x.shape[0]))

    if flag==0: # 实部部分高低频互换
        fft_yn_new.real[0:chnum] = fft_yn.real[(halfnum-chnum):halfnum]
        fft_yn_new.real[(halfnum-chnum):halfnum] = fft_yn.real[0:chnum]
        # 轴对称
        fft_yn_new.real[-halfnum:] = np.flip(fft_yn_new.real[0:halfnum], axis=-1)
        fft_yn_new.imag = fft_yn.imag

    if flag==1: #虚部部分高低频互换
        fft_yn_new.imag[0:chnum] = fft_yn.imag[(halfnum - chnum):halfnum]
        fft_yn_new.imag[(halfnum - chnum):halfnum] = fft_yn.imag[0:chnum]
        # 中心对称
        fft_yn_new.imag[-halfnum:] = -np.flip(fft_yn_new.imag[0:halfnum], axis=-1)
        fft_yn_new.real = fft_yn.real

    if flag==2: # 实部部分频率*5
        mul = np.random.uniform(2, 5)
        fft_yn_new.real[0:chnum] = fft_yn.real[0:chnum]*mul
        fft_yn_new.real[chnum:halfnum] = fft_yn.real[chnum:halfnum]
        # 轴对称
        fft_yn_new.real[-halfnum:] = np.flip(fft_yn_new.real[0:halfnum], axis=-1)
        fft_yn_new.imag = fft_yn.imag

    if flag==3: # 虚部部分频率*5
        mul = np.random.uniform(2, 5)
        fft_yn_new.imag[0:chnum] = fft_yn.imag[0:chnum] * mul
        fft_yn_new.imag[chnum:halfnum] = fft_yn.imag[chnum:halfnum]
        # 中心对称
        fft_yn_new.imag[-halfnum:] = -np.flip(fft_yn_new.imag[0:halfnum], axis=-1)
        fft_yn_new.real = fft_yn.real

    xn_ifft = ifft(fft_yn_new) + np.mean(xn)
    y = torch.from_numpy(xn_ifft.real)

    return y


def ma2com(magnitude, angle):
    return magnitude * np.exp(1j * angle)

# from iad
def tsaug_freq_domain(
        ts_data,
        aug_type='both',  # "magnitude, phase, both"
        segwideth_ratio=0.1,
        seg_num=2,
        mag_rpl_mu='mean',  # 'mean' 'zero'
        mag_rpl_sigma2_times=1,  # 0, 0.1, 0.5
        phase_addnoise_sigma2=0.1,  # 0, 0.1, 0.5, 1
        topvaluekeep_ratio=0.1,
        twosidekeep_ratio=0.05,
        debug=False):

    # old code updating:
    rpl_mu = mag_rpl_mu
    rpl_sigma2_times = mag_rpl_sigma2_times
    rpl_phase_sigma2 = phase_addnoise_sigma2

    # remove mean value for spectra:  input ts_data
    ts_data = np.array(ts_data).flatten().reshape(-1, )
    data_mean = np.mean(ts_data)

    # freq_data len
    freq_len = int(len(ts_data) / 2)
    # find rpl seg with starting point in 0~valid_len
    rpl_len = int(segwideth_ratio * freq_len)
    rpl_len_half = int(np.floor(rpl_len / 2.0))
    valid_len = freq_len - rpl_len
    assert seg_num < int(valid_len/rpl_len_half), \
        "Value of seg_num is too large"
    # find random points of seg_num
    seg_start_pos = set()
    while len(seg_start_pos) < seg_num:
        tmp_pos = np.random.randint(0, valid_len)
        valid_candi = True
        for item in seg_start_pos:
            ltem_left = item - rpl_len_half
            ltem_right = item + rpl_len_half
            if ltem_left < tmp_pos < ltem_right:
                valid_candi = False
                break
        if valid_candi:
            seg_start_pos.add(tmp_pos)
    seg_start_pos = sorted(seg_start_pos)

    # seg_start_pos, rpl_len, ,rpl_mu,rpl_sigma2_times
    freq_data = rfft(ts_data -
                     data_mean)  # the length is about half of ts_data
    freq_mag = np.abs(freq_data)
    freq_phase = np.angle(freq_data)

    if rpl_mu == 'mean':  # 'mean' 'zero'
        mag_mean = np.mean(freq_mag)
    elif rpl_mu == 'zero':
        mag_mean = 0
    else:
        raise ValueError("rpl_mu can only be 'mean' or 'zero'.")
    mag_var = np.var(freq_mag)

    # augment mag or phase based aug_type
    updated_freq_mag = freq_mag.copy()
    updated_freq_phase = freq_phase.copy()
    if aug_type == "magnitude":
        for pos in seg_start_pos:
            updated_freq_mag[pos:pos+rpl_len] = \
                np.abs(np.random.normal(mag_mean, np.sqrt(
                    rpl_sigma2_times*mag_var), rpl_len))
    elif aug_type == "phase":
        for pos in seg_start_pos:
            # note: not need to use % np.pi, since ma2com would deal with it.
            updated_freq_phase[pos:pos+rpl_len] += \
                np.random.normal(0, np.sqrt(rpl_phase_sigma2), rpl_len)
    elif aug_type == "both":
        for pos in seg_start_pos:
            updated_freq_mag[pos:pos+rpl_len] = \
                np.abs(np.random.normal(mag_mean, np.sqrt(
                    rpl_sigma2_times*mag_var), rpl_len))
            updated_freq_phase[pos:pos+rpl_len] += \
                np.random.normal(0, np.sqrt(rpl_phase_sigma2), rpl_len)
    else:
        raise ValueError(
            "aug_type can only be 'magnitude', 'phase' or 'both'.")

    # recover the top value of original freq_data by topvaluekeep_ratio
    topvaluekeep_num = int(len(freq_data) * topvaluekeep_ratio)
    min_topvalue_num = 5
    if topvaluekeep_num < min_topvalue_num:
        topvaluekeep_num = min_topvalue_num
    topvaluekeep_index = np.argsort(freq_mag)[::-1][:topvaluekeep_num]
    topvalue_freq_mag = freq_mag[topvaluekeep_index]
    topvalue_freq_phase = freq_phase[topvaluekeep_index]
    updated_freq_mag[topvaluekeep_index] = topvalue_freq_mag
    updated_freq_phase[topvaluekeep_index] = topvalue_freq_phase

    updated_freq_data = ma2com(updated_freq_mag, updated_freq_phase)
    augmented_data = irfft(updated_freq_data) + data_mean

    # recover the edge signal of two sides due to severe distortion from aug
    sidekeep_num = int(len(ts_data) * twosidekeep_ratio) // 2
    min_sidekeep_num = 2
    if sidekeep_num < min_sidekeep_num:
        sidekeep_num = min_sidekeep_num
    augmented_data[:sidekeep_num] = ts_data[:sidekeep_num]
    augmented_data[-sidekeep_num:] = ts_data[-sidekeep_num:]

    if debug:
        from matplotlib import pyplot as plt
        figsize = (9, 3)
        freq_x_axis = np.arange(len(freq_mag))
        plt.figure(figsize=figsize)
        plt.plot(freq_x_axis, updated_freq_mag, 'r', freq_x_axis, freq_mag,
                 'b')
        plt.xlabel('Freq Index')
        plt.ylabel('Magnitude Spectra')
        plt.legend(['Augmented Magnitude', 'Original Magnitude'])

        plt.figure(figsize=figsize)
        plt.plot(freq_x_axis, updated_freq_phase, 'r', freq_x_axis, freq_phase,
                 'b')
        plt.xlabel('Freq Index')
        plt.ylabel('Phase Spectra')
        plt.legend(['Augmented Phase', 'Original Phase'])

        time_x_axis = np.arange(len(ts_data))
        plt.figure(figsize=figsize)
        plt.plot(time_x_axis, augmented_data, 'r', time_x_axis, ts_data, 'b')
        plt.xlabel('Time Index')
        plt.ylabel('Signal')
        plt.legend(['Augmented Time Series', 'Original Time Series'])
        plt.show()

    return augmented_data

# suspect_window_length 这个值比较小, fft段异常的注入也要受到这个限制的影响...
# 周期的缩放这类确实是频域上更好操作，还有类似的异常种类么
def fft_aug(
    x: torch.Tensor,
    y: torch.Tensor,
    coe_rate: float, # 某种异常的注入比例
    suspect_window_length: int,
    random_start_end: bool = True,
    method: str = "multi_sea",
) -> torch.Tensor:
    """Contextual Outlier Exposure.

    Args:
        x : Tensor of shape (batch, ts channels, time)
        y : Tensor of shape (batch, )
        coe_rate : Number of generated anomalies as proportion of the batch size.
        random_start_end : If True, a random subset within the suspect segment is permuted between time series;
            if False, the whole suspect segment is randomly permuted.
    """

    if coe_rate == 0:
        raise ValueError(f"coe_rate must be > 0.")
    batch_size = x.shape[0]
    ts_channels = x.shape[1]
    oe_size = int(batch_size * coe_rate)

    # Select indices
    idx_1 = torch.arange(oe_size)
    idx_2 = torch.arange(oe_size)
    while torch.any(idx_1 == idx_2):
        idx_1 = torch.randint(low=0, high=batch_size, size=(oe_size,)).type_as(x).long()
        idx_2 = torch.randint(low=0, high=batch_size, size=(oe_size,)).type_as(x).long()

    if ts_channels > 3:
        numb_dim_to_swap = np.random.randint(low=3, high=ts_channels, size=(oe_size))
    else:
        numb_dim_to_swap = np.ones(oe_size) * ts_channels

    x_oe = x[idx_1].clone()  # .detach()
    oe_time_start_end = np.random.randint(
        low=x.shape[-1] - suspect_window_length, high=x.shape[-1] + 1, size=(oe_size, 2)
    )
    oe_time_start_end.sort(axis=1)
    # for start, end in oe_time_start_end:
    for i in range(len(idx_2)):
        # obtain the dimensons to swap
        numb_dim_to_swap_here = int(numb_dim_to_swap[i])
        dims_to_swap_here = np.random.choice(
            range(ts_channels), size=numb_dim_to_swap_here, replace=False
        )

        # obtain start and end of swap
        start, end = oe_time_start_end[i]

        if method == 'multi_sea':
            # 下面这种写法考虑的是两个序列的拼接，也就是类似coe_batch, 但感觉不一定需要这么强？
            # 随机决定是两个序列拼接还是同一个序列变换后拼接？
            flg = np.random.randint(low=0, high=1)
            # 两个序列拼接
#             x_oe[i, dims_to_swap_here, start:end] = seasonal_shift(x=x[idx_2[i], dims_to_swap_here, start:end], tl=0.9, th=1.2)

            # 同一序列变换后拼接
            x_oe[i, dims_to_swap_here, start:end] = seasonal_shift(x=x[idx_1[i], dims_to_swap_here, start:end], tl=0.9, th=1.2)

        if method == 'noise':
            # 同一序列拼接
            x_oe[i, dims_to_swap_here, start:end] = with_noise(x=x[idx_1[i], dims_to_swap_here, start:end])

        if method == 'other':
            # 同一序列拼接
            x_oe[i, dims_to_swap_here, start:end] = other_fftshift(x=x[idx_1[i], dims_to_swap_here, start:end])
            
        if method == 'from_iad':
            # 同一序列拼接
            ts_data = x[idx_1[i], dims_to_swap_here, start:end]
            augmented_data = tsaug_freq_domain(
                                test_numpy,
                                aug_type='both',  # "magnitude, phase, both"
                                segwideth_ratio=0.1,
                                seg_num=2,
                                mag_rpl_mu='mean',  # 'mean' 'zero'
                                mag_rpl_sigma2_times = 1,   # 0, 0.05, 0.1, 0.5
                                phase_addnoise_sigma2 = 1, # 0, 0.05, 0.1, 0.5, 
                                topvaluekeep_ratio=0.1,
                                twosidekeep_ratio=0.05,
                                debug=False)
            x_oe[i, dims_to_swap_here, start:end] = torch.from_numpy(augmented_data)

    # Label as positive anomalies
    y_oe = torch.ones(oe_size).type_as(y)

    return x_oe, y_oe