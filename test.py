import argparse
import torch
from statistics import mean
import os
import gdown

from utility.losses import smooth_crossentropy
from models.utils import get_model
from datasets.utils import get_data
from models.ema_model import SAMMovingAverageModel



def load_pretrain(model, model_name, data_name, chkpt_path):
    if chkpt_path:
        checkpoint_path = chkpt_path
    else:
        model_name = model_name.lower()
        data_name = data_name.lower()

        checkpoint_path = f'./pretrained_weight/EMASAM_{model_name}_{data_name}.pth'
        
        WEIGHT_URLS = {
            "cifar100": {
                "wideresnet": "https://drive.google.com/uc?id=1VE6M79X8enh8-ZZqplpxGac5xdIy-y0E",
                "pyramidnet": "https://drive.google.com/uc?id=1F0N_OCzuVic6_M3UnZU5kcikjiIXH6Fy",
                "resnet18": "https://drive.google.com/uc?id=1RdoeNn2OeetW6fpieVyCzfgJMGjWI_Dy",
                "resnet50": "https://drive.google.com/uc?id=1POovOy4-QCCOejV_lNVu-u463LJlXtSX",
            },
            "cifar10": {
                "wideresnet": "https://drive.google.com/uc?id=14dJrEWZta10DI-x0awNxo04EjW4dveSD",
                "pyramidnet": "https://drive.google.com/uc?id=1O2RvkNbRw34GxOxYPqesc8W4H--5eXXi",
                "resnet18": "https://drive.google.com/uc?id=13zUmocRKBgRkAZPCRrXghrBP59wXUQVf",
                "resnet50": "https://drive.google.com/uc?id=1NuhN3f1wd2keCQ9szKnuzgD_NRUd5_fp",
            },
            "fashionmnist": {
                "wideresnet": "https://drive.google.com/uc?id=1JtVrctA8o1ZkyHDaJrCwQdNlh7xQMJ0W",
                "pyramidnet": "https://drive.google.com/uc?id=1BQQWUCESmST2P5ugzSLQMKFjY-InZriR",
                "resnet18": "https://drive.google.com/uc?id=1isgL_IVlPtVi01pzKXGv2vtgHuEeiuGC",
                "resnet50": "https://drive.google.com/uc?id=1g0xcJRpI9WBI5h6nOG4xOC-bp7Yo7MaG",
            },
            "emnist": {
                "wideresnet": "https://drive.google.com/uc?id=1Dohwd_4yyh03Kdzxws1i8Zhw5cI5sKHd",
                "pyramidnet": "https://drive.google.com/uc?id=1r6PqfIrpCtIqs2OXS-6E3QQEAmhd2hcM",
                "resnet18": "https://drive.google.com/uc?id=1HJkS9nXQjGhLk0tpmTk6yrf2Ok3F0Zdo",
                "resnet50": "https://drive.google.com/uc?id=1D6ygQL_BhwI_bH0brJ-6HFZ2dBInVb_D",
            },
            "imagenet": {
                "resnet50": "https://drive.google.com/uc?id=1q-r7uZyTVK46s7uDp-LnsXYKmgIbf-f9",
                "vit_b_16": "https://drive.google.com/uc?id=1NNhZKWV7kJ0B3xK-rkDtKSsthd_wooaQ",
            }
        }
        weight_url = WEIGHT_URLS[data_name][model_name]
                
        # Check if checkpoint exists
        if not os.path.exists(checkpoint_path):
            if not os.path.exists("./pretrained_weight/"):
                os.makedirs("./pretrained_weight/")
            print(f"Checkpoint not found at {checkpoint_path}\ndownloading pre-trained checkpoint...")
            gdown.download(weight_url, checkpoint_path, quiet=False)
            print(f"Checkpoint downloaded to {checkpoint_path}")
        
    checkpoint = torch.load(checkpoint_path)
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except:
        model.module.load_state_dict(checkpoint['model_state_dict'])
    print("\n===================================")
    print(f"Model {model_name}")
    print(f"Dataset {data_name}")
    print(f"Checkpoint Path {checkpoint_path}\n")
    
    return model
            

def test(model_name, dataset_name, gpu, threads, chkpt_path, tsubame_id):
    DATASET_NUM_CLASSES = {"cifar10": 10, "cifar100": 100, "fashionmnist": 10, "emnist": 47, "imagenet_1k": 1000}
    cfg_test = {}
    cfg_test["num_classes"] = DATASET_NUM_CLASSES[dataset_name]
    
    device = torch.device(gpu)

    model = get_model(model_name, dataset_name, **cfg_test)
    model = SAMMovingAverageModel(model) #model wrapper for EMASAM
    if dataset_name == "imagenet_1k":
        batch_size = 512
    elif dataset_name in ["cifar10", "cifar100", "fashionmnist", "emnist"]:
        batch_size = 32
    dataset = get_data(dataset_name, cfg_test["num_classes"], batch_size, threads, tsubame_id)
    
    model = model.to(device)
    model = load_pretrain(model, model_name, dataset_name, chkpt_path)
    
    model.eval()
    
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    with torch.no_grad():
        for batch in dataset.test:
            inputs, targets = (b.to(device) for b in batch)

            predictions = model(inputs)
            loss = smooth_crossentropy(predictions, targets)
            total_loss += loss.sum().item()

            correct = torch.argmax(predictions, 1) == targets
            correct_predictions += correct.sum().item()
            total_samples += inputs.size(0)
    average_loss = total_loss / total_samples
    accuracy = correct_predictions / total_samples

    print(f'Average Loss: {average_loss:.4f}')
    print(f'Accuracy: {accuracy:.4f}')
                

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Model Architecture")
    parser.add_argument("--dataset", type=str, help="Dataset for training")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device to use.")
    parser.add_argument("--threads", type=int, default=4, help="Number of threads for data loading.")
    parser.add_argument("--chkpt_path", default=None, help="Path to a pre-trained checkpoint")
    parser.add_argument("--tsubame_id", type=str, default=None, help="leave it = None, used only when running on ScienceTokyo's Tsubame supercomputer")
    
    args = parser.parse_args()
    
    test(args.model, args.dataset, args.gpu, args.threads, args.chkpt_path, args.tsubame_id)
    