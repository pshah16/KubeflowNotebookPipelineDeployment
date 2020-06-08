import kfp.dsl as dsl
import kfp.components as comp
from collections import OrderedDict
from kubernetes import client as k8s_client


def dataprocessing(TRAIN_STEPS: int):

    import os
    import shutil
    from kale.utils import pod_utils
    from kale.marshal import resource_save as _kale_resource_save
    from kale.marshal import resource_load as _kale_resource_load

    _kale_data_directory = "/home/jovyan/data-vol-1/KubeflowNotebookPipelineDeployment/kale/examples/pytorch-classification/marshal"

    if not os.path.isdir(_kale_data_directory):
        os.makedirs(_kale_data_directory, exist_ok=True)

    import subprocess
    import sys

    reqs = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'])
    installed_packages = [r.decode().split('==')[0] for r in reqs.split()]

    if 'wget' in installed_packages:
        import wget
    else:
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'wget'])
        import wget

    if 'torch' in installed_packages:
        import torch.nn as nn
        import torch
    else:
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'torch==1.5.0',
                        '-f', 'https://download.pytorch.org/whl/torch_stable.html'])
        import torch.nn as nn
        import torch

    if 'torchvision' in installed_packages:
        import torchvision
        import torchvision.transforms as transforms
    else:
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'torchvision==0.6.0',
                        '-f', 'https://download.pytorch.org/whl/torch_stable.html'])
        import torchvision
        import torchvision.transforms as transforms

    import matplotlib.pyplot as plt
    import numpy as np
    import os

    try:
        from function_library.function_library import Net, imshow
    except:
        print("Installing function_library")
        url_function = 'https://raw.githubusercontent.com/pshah16/KubeflowNotebookPipelineDeployment/master/kale/examples/pytorch-classification/function_library/function_library.py'
        path = '/'
        wget.download(url_function, path)
        print("Function library installed")
        sys.path.append(path)
        import function_library
        print("Imported function_library")
        from function_library import Net, imshow
        print("Imported functions from function_library")
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    input_data_folder = "./data"

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root=input_data_folder, train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root=input_data_folder, train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=2)

    # -----------------------DATA SAVING START---------------------------------
    if "testloader" in locals():
        _kale_resource_save(testloader, os.path.join(
            _kale_data_directory, "testloader"))
    else:
        print("_kale_resource_save: `testloader` not found.")
    if "trainloader" in locals():
        _kale_resource_save(trainloader, os.path.join(
            _kale_data_directory, "trainloader"))
    else:
        print("_kale_resource_save: `trainloader` not found.")
    # -----------------------DATA SAVING END-----------------------------------


def train(TRAIN_STEPS: int):
    
    import os
    import shutil
    from kale.utils import pod_utils
    from kale.marshal import resource_save as _kale_resource_save
    from kale.marshal import resource_load as _kale_resource_load

    _kale_data_directory = "/home/jovyan/data-vol-1/KubeflowNotebookPipelineDeployment/kale/examples/pytorch-classification/marshal"

    if not os.path.isdir(_kale_data_directory):
        os.makedirs(_kale_data_directory, exist_ok=True)

    # -----------------------DATA LOADING START--------------------------------
    _kale_directory_file_names = [
        os.path.splitext(f)[0]
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f))
    ]

    if "trainloader" not in _kale_directory_file_names:
        raise ValueError("trainloader" + " does not exists in directory")

    _kale_load_file_name = [
        f
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f)) and
        os.path.splitext(f)[0] == "trainloader"
    ]
    if len(_kale_load_file_name) > 1:
        raise ValueError("Found multiple files with name " +
                         "trainloader" + ": " + str(_kale_load_file_name))
    _kale_load_file_name = _kale_load_file_name[0]
    trainloader = _kale_resource_load(os.path.join(
        _kale_data_directory, _kale_load_file_name))
    # -----------------------DATA LOADING END----------------------------------

    import subprocess
    import sys

    reqs = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'])
    installed_packages = [r.decode().split('==')[0] for r in reqs.split()]

    if 'wget' in installed_packages:
        import wget
    else:
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'wget'])
        import wget

    if 'torch' in installed_packages:
        import torch.nn as nn
        import torch
    else:
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'torch==1.5.0',
                        '-f', 'https://download.pytorch.org/whl/torch_stable.html'])
        import torch.nn as nn
        import torch

    if 'torchvision' in installed_packages:
        import torchvision
        import torchvision.transforms as transforms
    else:
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'torchvision==0.6.0',
                        '-f', 'https://download.pytorch.org/whl/torch_stable.html'])
        import torchvision
        import torchvision.transforms as transforms

    import matplotlib.pyplot as plt
    import numpy as np
    import os

    try:
        from function_library.function_library import Net, imshow
    except:
        print("Installing function_library")
        url_function = 'https://raw.githubusercontent.com/pshah16/KubeflowNotebookPipelineDeployment/master/kale/examples/pytorch-classification/function_library/function_library.py'
        path = '/'
        wget.download(url_function, path)
        print("Function library installed")
        sys.path.append(path)
        import function_library
        print("Imported function_library")
        from function_library import Net, imshow
        print("Imported functions from function_library")
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    import torch.optim as optim

    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(TRAIN_STEPS):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')

    # -----------------------DATA SAVING START---------------------------------
    if "net" in locals():
        _kale_resource_save(net, os.path.join(_kale_data_directory, "net"))
    else:
        print("_kale_resource_save: `net` not found.")
    # -----------------------DATA SAVING END-----------------------------------


def testontest(TRAIN_STEPS: int):

    import os
    import shutil
    from kale.utils import pod_utils
    from kale.marshal import resource_save as _kale_resource_save
    from kale.marshal import resource_load as _kale_resource_load

    _kale_data_directory = "/home/jovyan/data-vol-1/KubeflowNotebookPipelineDeployment/kale/examples/pytorch-classification/marshal"

    if not os.path.isdir(_kale_data_directory):
        os.makedirs(_kale_data_directory, exist_ok=True)

    # -----------------------DATA LOADING START--------------------------------
    _kale_directory_file_names = [
        os.path.splitext(f)[0]
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f))
    ]

    if "net" not in _kale_directory_file_names:
        raise ValueError("net" + " does not exists in directory")

    _kale_load_file_name = [
        f
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f)) and
        os.path.splitext(f)[0] == "net"
    ]
    if len(_kale_load_file_name) > 1:
        raise ValueError("Found multiple files with name " +
                         "net" + ": " + str(_kale_load_file_name))
    _kale_load_file_name = _kale_load_file_name[0]
    net = _kale_resource_load(os.path.join(
        _kale_data_directory, _kale_load_file_name))

    if "testloader" not in _kale_directory_file_names:
        raise ValueError("testloader" + " does not exists in directory")

    _kale_load_file_name = [
        f
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f)) and
        os.path.splitext(f)[0] == "testloader"
    ]
    if len(_kale_load_file_name) > 1:
        raise ValueError("Found multiple files with name " +
                         "testloader" + ": " + str(_kale_load_file_name))
    _kale_load_file_name = _kale_load_file_name[0]
    testloader = _kale_resource_load(os.path.join(
        _kale_data_directory, _kale_load_file_name))
    # -----------------------DATA LOADING END----------------------------------

    import subprocess
    import sys

    reqs = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'])
    installed_packages = [r.decode().split('==')[0] for r in reqs.split()]

    if 'wget' in installed_packages:
        import wget
    else:
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'wget'])
        import wget

    if 'torch' in installed_packages:
        import torch.nn as nn
        import torch
    else:
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'torch==1.5.0',
                        '-f', 'https://download.pytorch.org/whl/torch_stable.html'])
        import torch.nn as nn
        import torch

    if 'torchvision' in installed_packages:
        import torchvision
        import torchvision.transforms as transforms
    else:
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'torchvision==0.6.0',
                        '-f', 'https://download.pytorch.org/whl/torch_stable.html'])
        import torchvision
        import torchvision.transforms as transforms

    import matplotlib.pyplot as plt
    import numpy as np
    import os

    try:
        from function_library.function_library import Net, imshow
    except:
        print("Installing function_library")
        url_function = 'https://raw.githubusercontent.com/pshah16/KubeflowNotebookPipelineDeployment/master/kale/examples/pytorch-classification/function_library/function_library.py'
        path = '/'
        wget.download(url_function, path)
        print("Function library installed")
        sys.path.append(path)
        import function_library
        print("Imported function_library")
        from function_library import Net, imshow
        print("Imported functions from function_library")
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    dataiter = iter(testloader)
    images, labels = dataiter.next()

    # print images
    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join('%5s' %
                                    classes[labels[j]] for j in range(4)))
    outputs = net(images)
    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                                  for j in range(4)))

    # -----------------------DATA SAVING START---------------------------------
    if "net" in locals():
        _kale_resource_save(net, os.path.join(_kale_data_directory, "net"))
    else:
        print("_kale_resource_save: `net` not found.")
    if "testloader" in locals():
        _kale_resource_save(testloader, os.path.join(
            _kale_data_directory, "testloader"))
    else:
        print("_kale_resource_save: `testloader` not found.")
    # -----------------------DATA SAVING END-----------------------------------


def testwhole(TRAIN_STEPS: int):

    import os
    import shutil
    from kale.utils import pod_utils
    from kale.marshal import resource_save as _kale_resource_save
    from kale.marshal import resource_load as _kale_resource_load

    _kale_data_directory = "/home/jovyan/data-vol-1/KubeflowNotebookPipelineDeployment/kale/examples/pytorch-classification/marshal"

    if not os.path.isdir(_kale_data_directory):
        os.makedirs(_kale_data_directory, exist_ok=True)

    # -----------------------DATA LOADING START--------------------------------
    _kale_directory_file_names = [
        os.path.splitext(f)[0]
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f))
    ]

    if "net" not in _kale_directory_file_names:
        raise ValueError("net" + " does not exists in directory")

    _kale_load_file_name = [
        f
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f)) and
        os.path.splitext(f)[0] == "net"
    ]
    if len(_kale_load_file_name) > 1:
        raise ValueError("Found multiple files with name " +
                         "net" + ": " + str(_kale_load_file_name))
    _kale_load_file_name = _kale_load_file_name[0]
    net = _kale_resource_load(os.path.join(
        _kale_data_directory, _kale_load_file_name))

    if "testloader" not in _kale_directory_file_names:
        raise ValueError("testloader" + " does not exists in directory")

    _kale_load_file_name = [
        f
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f)) and
        os.path.splitext(f)[0] == "testloader"
    ]
    if len(_kale_load_file_name) > 1:
        raise ValueError("Found multiple files with name " +
                         "testloader" + ": " + str(_kale_load_file_name))
    _kale_load_file_name = _kale_load_file_name[0]
    testloader = _kale_resource_load(os.path.join(
        _kale_data_directory, _kale_load_file_name))
    # -----------------------DATA LOADING END----------------------------------

    import subprocess
    import sys

    reqs = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'])
    installed_packages = [r.decode().split('==')[0] for r in reqs.split()]

    if 'wget' in installed_packages:
        import wget
    else:
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'wget'])
        import wget

    if 'torch' in installed_packages:
        import torch.nn as nn
        import torch
    else:
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'torch==1.5.0',
                        '-f', 'https://download.pytorch.org/whl/torch_stable.html'])
        import torch.nn as nn
        import torch

    if 'torchvision' in installed_packages:
        import torchvision
        import torchvision.transforms as transforms
    else:
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'torchvision==0.6.0',
                        '-f', 'https://download.pytorch.org/whl/torch_stable.html'])
        import torchvision
        import torchvision.transforms as transforms

    import matplotlib.pyplot as plt
    import numpy as np
    import os

    try:
        from function_library.function_library import Net, imshow
    except:
        print("Installing function_library")
        url_function = 'https://raw.githubusercontent.com/pshah16/KubeflowNotebookPipelineDeployment/master/kale/examples/pytorch-classification/function_library/function_library.py'
        path = '/'
        wget.download(url_function, path)
        print("Function library installed")
        sys.path.append(path)
        import function_library
        print("Imported function_library")
        from function_library import Net, imshow
        print("Imported functions from function_library")
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))


dataprocessing_op = comp.func_to_container_op(
    dataprocessing, base_image='auroradevacr.azurecr.io/kubeflownotebook')


train_op = comp.func_to_container_op(
    train, base_image='auroradevacr.azurecr.io/kubeflownotebook')


testontest_op = comp.func_to_container_op(
    testontest, base_image='auroradevacr.azurecr.io/kubeflownotebook')


testwhole_op = comp.func_to_container_op(
    testwhole, base_image='auroradevacr.azurecr.io/kubeflownotebook')


@dsl.pipeline(
    name='cifar10-classification-gvm3v',
    description='Sequential PyTorch pipeline to train a network on the CIFAR10 dataset'
)
def auto_generated_pipeline(TRAIN_STEPS='2'):
    pvolumes_dict = OrderedDict()

#     marshal_vop = dsl.VolumeOp(
#         name="kale_marshal_volume",
#         resource_name="kale-marshal-pvc",
#         modes=dsl.VOLUME_MODE_RWM,
#         size="2Gi",
#         storage_class="standard",
#         annotations={'example':'cifar_pvc'}
#     )
#     pvolumes_dict['/marshal'] = marshal_vop.volume

#     dataprocessing_task = dataprocessing_op(TRAIN_STEPS)\
#         .add_pvolumes(pvolumes_dict)\
#         .after()
    example_path = "./KubeflowNotebookPipelineDeployment/kale/examples/pytorch-classification"
    dataprocessing_task = dataprocessing_op(TRAIN_STEPS).\
        add_volume(k8s_client.V1Volume(name='kubeflowpipelinedemo-pshah-azurefile',\
                                    host_path=k8s_client.V1HostPathVolumeSource(path=example_path))).after()

    dataprocessing_task.container.working_dir=\
    "/home/jovyan/data-vol-1/KubeflowNotebookPipelineDeployment/kale/examples/pytorch-classification"
    dataprocessing_task.container.set_security_context(
        k8s_client.V1SecurityContext(run_as_user=0))
    
    dataprocessing_task.add_resource_limit('memory','4G')
    dataprocessing_task.add_resource_limit('cpu','0.5')
    dataprocessing_task.set_memory_request('2G')
    dataprocessing_task.set_cpu_request('0.5')

#     train_task = train_op(TRAIN_STEPS)\
#         .add_pvolumes(pvolumes_dict)\
#         .after(dataprocessing_task)
    train_task = train_op(TRAIN_STEPS).\
    add_volume(k8s_client.V1Volume(name='kubeflowpipelinedemo-pshah-azurefile',\
    host_path=k8s_client.V1HostPathVolumeSource(path=example_path))).after(dataprocessing_task)
    train_task.container.working_dir=\
    "/home/jovyan/data-vol-1/KubeflowNotebookPipelineDeployment/kale/examples/pytorch-classification"
    train_task.container.set_security_context(
        k8s_client.V1SecurityContext(run_as_user=0))

    train_task.add_resource_limit('memory','4G')
    train_task.add_resource_limit('cpu','0.5')
    train_task.set_memory_request('2G')
    train_task.set_cpu_request('0.5')

#     testontest_task = testontest_op(TRAIN_STEPS)\
#         .add_pvolumes(pvolumes_dict)\
#         .after(train_task)
    testontest_task = testontest_op(TRAIN_STEPS)\
    .add_volume(k8s_client.V1Volume(name='workspace-kubeflowdemo',\
    host_path=k8s_client.V1HostPathVolumeSource(path=example_path))).after(train_task)
    testontest_task.container.working_dir=\
    "/home/jovyan/data-vol-1/KubeflowNotebookPipelineDeployment/kale/examples/pytorch-classification"
    testontest_task.container.set_security_context(
        k8s_client.V1SecurityContext(run_as_user=0))
    
    testontest_task.add_resource_limit('memory','4G')
    testontest_task.add_resource_limit('cpu','0.5')
    testontest_task.set_memory_request('2G')
    testontest_task.set_cpu_request('0.5')

#     testwhole_task = testwhole_op(TRAIN_STEPS)\
#         .add_pvolumes(pvolumes_dict)\
#         .after(testontest_task)
    testwhole_task = testwhole_op(TRAIN_STEPS)\
        .add_volume(k8s_client.V1Volume(name='workspace-kubeflowdemo',\
        host_path=k8s_client.V1HostPathVolumeSource(path=example_path))).after(testontest_task)
    testwhole_task.container.working_dir =\
    "/home/jovyan/data-vol-1/KubeflowNotebookPipelineDeployment/kale/examples/pytorch-classification"
    testwhole_task.container.set_security_context(
        k8s_client.V1SecurityContext(run_as_user=0))

    testwhole_task.add_resource_limit('memory','4G')
    testwhole_task.add_resource_limit('cpu','0.5')
    testwhole_task.set_memory_request('2G')
    testwhole_task.set_cpu_request('0.5')


if __name__ == "__main__":
    pipeline_func = auto_generated_pipeline
    pipeline_filename = pipeline_func.__name__ + '.pipeline.tar.gz'
    import kfp.compiler as compiler
    compiler.Compiler().compile(pipeline_func, pipeline_filename)

    # Get or create an experiment and submit a pipeline run
    import kfp
    client = kfp.Client()
    experiment = client.create_experiment('cifar')

    # Submit a pipeline run
    run_name = 'cifar10-classification-gvm3v_run'
    run_result = client.run_pipeline(
        experiment.id, run_name, pipeline_filename, {})
