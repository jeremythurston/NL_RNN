import matplotlib.pyplot as plt

def plot_history(history):
    plt.rcParams["font.sans-serif"] = "Helvetica"
    plt.rcParams["font.size"] = 14
    plt.rcParams["figure.figsize"] = (10,6)
    
    train_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    plt.plot(train_acc)
    plt.plot(val_acc)
    
    plt.legend(['Training','Validation'], loc="best")
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.show()
    
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.plot(train_acc)
    plt.plot(val_acc)
    
    plt.ylabel('MS Error')
    plt.xlabel('Epochs')
    plt.legend(['Training','Validation'])
    plt.show()