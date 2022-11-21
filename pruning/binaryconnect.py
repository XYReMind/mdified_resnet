import torch.nn as nn
import torch
import numpy
from torch.autograd import Variable
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import utils

use_gpu = torch.cuda.is_available()
if use_gpu:
    torch.cuda.empty_cache()
    device = torch.device('cuda:0')
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device('cpu')

def train_one_epoch(bcmodel,trainloader,criterion,optimizer,epoch):
    ####create bar
    bar = tqdm(total=len(trainloader), desc="[Train]")

    ####initialize loss
    epoch_loss = 0

    #### learning process
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        #scheduler.step()
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad(set_to_none = True)
        # forward + backward + optimize
        #bcmodel.binarization()
        bcmodel.BWN()
        outputs = bcmodel.model(inputs)
        loss_step  = criterion(outputs, labels)
        loss_step.backward()
        bcmodel.restore()
        optimizer.step()
        bcmodel.clip()
        # print statistics
        running_loss = loss_step.item()
        epoch_loss+=running_loss
        bar.set_description("[Train] Loss = {:.4f}".format(round(running_loss, 4)))
        bar.update()

    epoch_loss = epoch_loss/len(trainloader)
    bar.close()

    return bcmodel,epoch_loss

def validate(bcmodel,validloader,criterion,epoch):
    bar = tqdm(total=len(validloader), desc="[Val]")
    val_loss = 0
    bcmodel.model.eval()
    total = 0
    correct = 0
    #bcmodel.binarization()
    bcmodel.BWN()
    for i, data in enumerate(validloader,0):

        #extract data
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        # forward pass but without grad

        with torch.no_grad():
            pred = bcmodel.model(inputs)

        # update loss, calculated by cpu
        val_loss += criterion(pred,labels).cpu().item()
        bar.set_description("[Val] Loss = {:.4f}".format(round(val_loss/len(validloader), 4)))
        bar.update()

        ## into tensorboard
        _, predicted = torch.max(pred, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    bcmodel.restore()

    val_loss = val_loss/len(validloader)
    val_acc = correct/total


    bar.close()

    return val_loss,val_acc

class BC():
    def __init__(self, model):

        # First we need to
        # count the number of Conv2d and Linear
        # This will be used next in order to build a list of all
        # parameters of the model

        count_targets = 0
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                count_targets = count_targets + 1

        start_range = 0
        end_range = count_targets-1
        self.bin_range = numpy.linspace(start_range,
                end_range, end_range-start_range+1)\
                        .astype('int').tolist()

        # Now we can initialize the list of parameters

        self.num_of_params = len(self.bin_range)
        self.saved_params = [] # This will be used to save the full precision weights

        self.target_modules = [] # this will contain the list of modules to be modified

        self.model = model # this contains the model that will be trained and quantified

        ### This builds the initial copy of all parameters and target modules
        index = -1
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                index = index + 1
                if index in self.bin_range:
                    tmp = m.weight.data.clone()
                    self.saved_params.append(tmp)
                    self.target_modules.append(m.weight)


    def save_params(self):

        ### This loop goes through the list of target modules, and saves the corresponding weights into the list of saved_parameters

        for index in range(self.num_of_params):
            self.saved_params[index].copy_(self.target_modules[index].data)

    def binarization(self):

        ### To be completed

        ### (1) Save the current full precision parameters using the save_params method

        self.save_params()


        ### (2) Binarize the weights in the model, by iterating through the list of target modules and overwrite the values with their binary version

        for index in range(self.num_of_params):
            self.target_modules[index].data.copy_(self.target_modules[index].data.sign())

    def BWN(self): # Binary Weight Network
        self.save_params()
        for index in range(self.num_of_params):
            E=self.target_modules[index].data.abs().mean()
            self.target_modules[index].data.copy_(self.target_modules[index].data.sign() *E)

    def restore(self):

        ### restore the copy from self.saved_params into the model

        for index in range(self.num_of_params):
            self.target_modules[index].data.copy_(self.saved_params[index])

    def clip(self):

        ## To be completed
        ## Clip all parameters to the range [-1,1] using Hard Tanh


        clip_scale=[]
        m=nn.Hardtanh(-1, 1)
        for index in range(self.num_of_params):
            clip_scale.append(m(Variable(self.target_modules[index].data)))
        for index in range(self.num_of_params):
            self.target_modules[index].data.copy_(clip_scale[index].data)

    def forward(self,x):

        ### This function is used so that the model can be used while training
        out = self.model(x)
        return out

    def train_model(self,trainloader,validloader,criterion,optimizer,scheduler,overfitting,epochs,name):

        '''
        Description :
        ------------
        Perform training and validation (at each epoch) for several epochs
        Parameters :
        ------------
        trainloader (object) : Dataloader of training set
        validloader (object) : Dataloader of validation set
        criterion (object) : loss to use for validation (see pytorch documentation)
        optimizer (object) : optimizer to use for validation (see pytorch documentation)
        epochs (int) : number of epochs for training
        name (str) : name of the experience
        Returns :
        ---------
        val_acc (float) : mean accuracy of the validation
        val_loss (float) : mean loss of the validation
        '''
        ## create tensorboard writer
        writer = SummaryWriter('logs/'+name)

        min_val_loss = 100000
        max_val_acc = 0
        end = 0

        for epoch in range(epochs):

            print('='*10 + ' epoch ' + str(epoch+1) + '/' + str(epochs) + ' ' + '='*10)
            self, training_loss = train_one_epoch(self,trainloader,criterion,optimizer,epoch)
            val_loss,val_acc = validate(self,validloader,criterion,epoch)
            scheduler.step()
            writer.add_scalars('Losses', {'val' : val_loss ,'train' : training_loss}  , epoch + 1)
            writer.add_scalar('Validation Accuracy', val_acc  , epoch + 1)
            writer.flush()

            if overfitting == 'accuracy':
                if max_val_acc < val_acc :
                    best_model = self
                    max_val_acc = val_acc
                    ## save model

                    self.save_weights(name)
                    end = epoch
                    print('==> best model saved <==')
                    self.save_train_results(name,val_acc,val_loss,end+1)
            elif overfitting == 'loss':
                if val_loss < min_val_loss and abs(val_loss-training_loss) < 0.2 :
                    best_model = self
                    min_val_loss = val_loss
                    ## save model
                    self.save_weights(name)
                    end = epoch
                    print('==> best model saved <==')
                    utils.save_train_results(name,val_acc,val_loss,end+1)
            print('  -> Training   Loss     = {}'.format(training_loss))
            print('  -> Validation Loss     = {}'.format(val_loss))
            print('  -> Validation Accuracy = {}'.format(val_acc))

    def test(self,testloader,criterion,device) :
        '''
        Description :
        ------------
        Perform test of the model
        Parameters :
        ------------
        model (object) : model to test
        testloader (object) : Dataloader of test set
        criterion (object) : loss to use for test (see pytorch documentation)
        optimizer (object) : optimizer to use for test (see pytorch documentation)
        Returns :
        ---------
        test_acc (float) : mean accuracy of the test
        test_loss (float) : mean loss of the test
        '''
        bar = tqdm(total=len(testloader), desc="[Test]")

        self.model = self.model.half()

        #### set model to eval mode
        self.model.eval()
        self.BWN()

        total = 0
        correct = 0
        test_loss = 0



        for i, data in enumerate(testloader):

            #extract data
            inputs, labels = data

            inputs = inputs.half().to(device)
            labels = labels.to(device)

            # set  loss
            running_loss = 0

            # forward pass but without grad
            with torch.no_grad():
                pred = self.model(inputs)


            # update loss, calculated by cpu
            running_loss = criterion(pred,labels).cpu().item()
            bar.set_description("[Test] Loss = {:.4f}".format(round(running_loss, 4)))
            bar.update()

            test_loss += criterion(pred,labels).cpu().item()

            _, predicted = torch.max(pred.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        self.restore()
        test_acc = 100 * correct / total
        test_loss = test_loss/len(testloader)

        bar.close()

        print(' -> Test Accuracy = {}'.format(test_acc))
        print(' -> Test Loss     = {}'.format(test_loss))


        return test_loss,test_acc

    def to(self,device):
        self.model.to(device)

        return self

    def load_weights(self,PATH):
        '''
        Description :
        ------------
        Load weights on model, with weight mask or not
        Parameters :
        ------------
        model (object) : model where to load wieghts
        path (str) : folder to find .pth file
        Returns :
        ---------
        model (model) : model with loaded weights
        '''

        PATH = 'logs/'+PATH+'/model_weights.pth'
        state_dict = torch.load(PATH)

        ## check if weight_mask and create if needed
        ## can't prune if binarization


        self.model.load_state_dict(state_dict)

        return self

    def save_weights(self,folder_name):
        PATH = './logs/{}/model_weights.pth'.format(folder_name)
        torch.save(self.model.state_dict(),PATH)