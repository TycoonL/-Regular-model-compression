import h5py
import matplotlib.pyplot as plt
import numpy as np

f1 = h5py.File("./results10/vgg_c10_weights.h5", 'r')
f = h5py.File("./results2/vgg_c10_weights.h5", 'r')
#f = h5py.File("./results8/compressed_vgg_weights.h5", 'r')
# print(f['conv2d']['shape'][:])
#weight = f['conv2d_1']['weights'][:]
# weight=f['dense_2']['dense_2']['kernel:0'][:]
plt.figure(12,figsize=(25,4))
plt.xlabel('number')
plt.ylabel('value')
n=0
for key in ['conv2d_1','conv2d_2','conv2d_3','conv2d_4','conv2d_5','dense']:#,'dense_1']:
    print("key : " + key)
    n=n+1
    weight1 = f1[key][key]['kernel:0'][:]
    weight = f[key][key]['kernel:0'][:]
    #weight = f[key]['weights'][:]
    arr1 = [*weight1.flat]
    arr = [*weight.flat]
    ymin = min(min(arr),min(arr1))
    ymax = max(max(arr),max(arr1))
    x = np.linspace(ymin, ymax, 60)
    plt.subplot(170 + n)
    plt.title(key)
    kwargs = dict(histtype='stepfilled', alpha=0.7, density=False, bins=x)
    plt.hist(arr1, **kwargs, label="L1")#blue
    plt.hist(arr, **kwargs,label="VGG")
    plt.legend()
plt.show()


# ymin = min(arr)
# ymax = max(arr)
# x = np.linspace(ymin, ymax, 40)
# hist, bins = np.histogram(arr, x)
# yy, _, _ = plt.hist(arr, x)
# hist = hist / len(arr)
#对不同的分布特征的样本进行对比时，将histtype=‘stepfilled’与透明性设置参数alpha搭配使用效果好
# x1 = np.random.normal(0, 0.8, 1000)
# x2 = np.random.normal(-2, 1, 1000)
# x3 = np.random.normal(3, 2, 1000)
# kwargs = dict(histtype='stepfilled', alpha=0.3, density=True, bins=40)
# plt.hist(x1, **kwargs)
# plt.hist(x2, **kwargs)
# plt.hist(x3, **kwargs)
# plt.show()
