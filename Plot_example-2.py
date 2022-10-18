import matplotlib.pyplot as plt
#matplotlib inline

xlabels = ["bar1","bar2", "bar3"]
costs = [13139, 16460, 33333]

bar = plt.bar(xlabels, costs, width=0.8, color=['blue', 'red', 'green'])
plt.xticks(range(len(xlabels)), xlabels)

plt.ylabel("Costs")
plt.xlabel("X label")
plt.legend((bar[0], bar[1], bar[2]), xlabels, loc="upper right")
plt.title("My chart")
plt.show()
