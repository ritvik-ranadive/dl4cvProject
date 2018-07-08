import matplotlib.pyplot as plt
import numpy as np

# Training Loss
# trainLossFile = open("trainLoss.txt", "r")
# trainLoss = trainLossFile.read()
# trainLoss = trainLoss.split("\n")
# trainLoss = np.array(trainLoss).astype(np.float)
# x = range(0, np.shape(trainLoss)[0], 1)
# x = np.array(x)
# plt.xlabel('Iterations')
# plt.ylabel('Loss')
# plt.title('Training Loss')
# plt.xticks(np.arange(x.min(), x.max(), 500))
# plt.yticks(np.arange(trainLoss.min(), 100, 10))
# plt.plot(x, trainLoss)
# plt.show()

# Validation Loss
valLossFile = open("valLoss.txt", "r")
valLoss = valLossFile.read()
valLoss = valLoss.split("\n")
valLoss = np.array(valLoss).astype(np.float)
x = range(0, np.shape(valLoss)[0], 1)
x = np.array(x)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Validation Loss')
plt.xticks(np.arange(x.min(), x.max(), 10))
plt.yticks(np.arange(0, 150, 10))
plt.plot(x, valLoss)
plt.show()

# valLossFile = open("valLoss.txt", "r")

# Training Accuracy
# trainAccFile = open("trainAcc.txt", "r")
# trainAcc = trainAccFile.read()
# trainAcc = trainAcc.split("\n")
# trainAcc = np.array(trainAcc).astype(np.float)
# x = range(0, np.shape(trainAcc)[0], 1)
# x = np.array(x)
# plt.xlabel('Iterations')
# plt.ylabel('Accuracy')
# plt.title('Training Accuracy')
# plt.xticks(np.arange(x.min(), x.max(), 500))
# plt.yticks(np.arange(trainAcc.min(), trainAcc.max(), 0.05))
# plt.plot(x, trainAcc)
# plt.show()

# Validation Accuracy
# valAccFile = open("valAcc.txt", "r")
# valAcc = valAccFile.read()
# valAcc = valAcc.split("\n")
# valAcc = np.array(valAcc).astype(np.float)
# x = range(0, 149, 1)
# x = np.array(x)
# plt.xlabel('Iterations')
# plt.ylabel('Accuracy')
# plt.title('Validation Accuracy')
# plt.xticks(np.arange(x.min(), x.max(), 10))
# plt.yticks(np.arange(valAcc.min(), valAcc.max(), 0.05))
# plt.plot(x, valAcc)
# plt.show()



# logFile = open("localizerLogs.txt")
# logs = logFile.read()
# logs = logs.split("Training")
# validationLogs = [log[log.index("V"):len(log)] for log in logs if "Validation" in log]
# # print(validationLogs)
# values = [log.split("Validation ") for log in validationLogs]
# # print(values)
# # print(values)
# valuesLoss = []
# valuesAcc = []
# # print("-----------------------------------------------------------------------------------------------")
# for value in values:
#     for text in value:
#         if "Loss" in text:
#             valuesLoss.append(text)
#         if "Accuracy" in text:
#             valuesAcc.append(text)
# print(valuesLoss)
# print(valuesAcc)
# valLoss = [value[6:len(value)] for value in valuesLoss]
# valAcc = [value[10:len(value)] for value in valuesAcc]
# print(valLoss)
# print(valAcc)
# valAccFile = open("valAcc.txt","w")
# valLossFile = open("valLoss.txt", "w")
# for num in valLoss:
#     valLossFile.write(str(num))
#     valLossFile.write("\n")
# for num in valAcc:
#     valAccFile.write(str(num))
#     valAccFile.write("\n")