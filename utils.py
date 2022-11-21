import torch
def load_weights(folder_name):
    PATH = 'logs/'+folder_name+'/model_weights.pth'
    model.load_state_dict(torch.load(PATH))

def save_weights(model,folder_name):
    PATH = './logs/{}/model_weights.pth'.format(folder_name)
    torch.save(model.state_dict(),PATH)

def save_test_results(path,test_acc,test_loss,pruning,ratio):
    full_path = "./logs/{}/results".format(path)
    if pruning :
        full_path += "pruned_{}".format(int(ratio*100))
    full_path +=".txt"
    f= open(full_path,"w+")
    f.write(' -> Test Accuracy = {}'.format(test_acc))
    f.write('\n')
    f.write(' -> Test Loss     = {}'.format(test_loss))
    f.write('\n')

    f.close()

def save_train_results(path,val_acc,val_loss,epoch):
    f= open("./logs/{}/training_results.txt".format(path),"w+")
    f.write(' -> Epoch      Number   = {}'.format(epoch))
    f.write('\n')
    f.write(' -> Validation Accuracy = {}'.format(val_acc))
    f.write('\n')
    f.write(' -> Validation Loss     = {}'.format(val_loss))


    f.close()