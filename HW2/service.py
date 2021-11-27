import matplotlib.pyplot as plt


def plot_policy_rewards(epochs, crosses, zeros):
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, crosses, label='crosses')
    plt.plot(epochs, zeros, label='zeros')
    plt.legend(loc="lower right")
    plt.xlabel('Epochs')
    plt.ylabel('Rewards')
