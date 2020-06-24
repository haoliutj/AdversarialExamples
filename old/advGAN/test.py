import matplotlib.pyplot as plt

fig = plt.figure(1)
a = [[1,3,4,4,9.8,7],[2,3,3,2,1,5]]
b = [[2,4,3,5,6,2],[5,6,3,8,2,9]]

plt.subplot(2,2,1)
plt.plot(a[0])
plt.plot(b[0])



plt.subplot(2,2,2)
plt.plot(a[1])
plt.plot(b[1])
plt.show()